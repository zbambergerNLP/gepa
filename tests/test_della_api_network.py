"""Exercise the private OpenRouter tunnel without external network requests."""

import importlib.util
import socket
import tempfile
import threading
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "della_api_network", Path(__file__).parents[1] / "scripts/della/api_network.py"
)
network = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(network)


@pytest.fixture
def tunnel():
    """Serve the same Unix socket used by the namespace bridge."""
    with tempfile.TemporaryDirectory(prefix="api-test-") as directory:
        path = str(Path(directory) / "proxy.sock")
        with network.UnixServer(path, network.OpenRouterConnect) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield path
            finally:
                server.shutdown()
                thread.join(timeout=2)


def read_header(connection):
    """Read through the CONNECT response without consuming tunneled data."""
    response = b""
    while not response.endswith(b"\r\n\r\n"):
        response += connection.recv(1)
    return response


@pytest.mark.parametrize(
    "wire_request",
    [
        b"CONNECT example.com:443 HTTP/1.1\r\n\r\n",
        b"CONNECT openrouter.ai:80 HTTP/1.1\r\n\r\n",
        b"CONNECT openrouter.ai.evil:443 HTTP/1.1\r\n\r\n",
        b"GET https://openrouter.ai/api/v1/models HTTP/1.1\r\n\r\n",
        b"CONNECT openrouter.ai:443 HTTP/1.1\r\nX: " + b"x" * 8192,
    ],
)
def test_rejects_other_destinations_without_connecting(tunnel, monkeypatch, wire_request):
    """Reject general egress and malformed requests before opening a socket."""
    monkeypatch.setattr(network.socket, "create_connection", lambda *a, **k: pytest.fail("Unexpected egress"))
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(2)
        client.connect(tunnel)
        client.sendall(wire_request)
        assert read_header(client).startswith(b"HTTP/1.1 403")


def test_namespace_bridge_preserves_opaque_bytes(tunnel, monkeypatch):
    """Carry CONNECT and opaque bidirectional payloads through both servers."""
    upstream, peer = socket.socketpair()
    destinations = []

    def connect(destination, timeout):
        destinations.append((destination, timeout))
        return upstream

    monkeypatch.setattr(network.socket, "create_connection", connect)
    with network.TCPServer(("127.0.0.1", 0), network.NamespaceBridge) as bridge:
        bridge.upstream = tunnel
        thread = threading.Thread(target=bridge.serve_forever, daemon=True)
        thread.start()
        try:
            with socket.socket() as client, peer:
                client.settimeout(2)
                peer.settimeout(2)
                client.connect(bridge.server_address)
                client.sendall(b"CONNECT openrouter.ai:443 HTTP/1.1\r\n\r\n")
                assert read_header(client).startswith(b"HTTP/1.1 200")
                client.sendall(b"\x16\x03\x03opaque-client-bytes")
                assert peer.recv(100) == b"\x16\x03\x03opaque-client-bytes"
                peer.sendall(b"\x16\x03\x03opaque-server-bytes")
                assert client.recv(100) == b"\x16\x03\x03opaque-server-bytes"
                assert destinations == [(("openrouter.ai", 443), 30)]
        finally:
            bridge.shutdown()
            thread.join(timeout=2)


def test_refuses_compute_node_before_starting_child(monkeypatch):
    """Require an allocation and the supported proxy on compute nodes."""
    monkeypatch.setattr(network.socket, "gethostname", lambda: "della-i23g3")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(network.sys, "argv", ["api_network.py", "--", "true"])
    monkeypatch.setattr(network.subprocess, "Popen", lambda *a, **k: pytest.fail("Unexpected child"))
    with pytest.raises(SystemExit) as error:
        network.main()
    assert error.value.code == 2


def test_allocated_compute_node_uses_loaded_cluster_proxy(monkeypatch):
    """Retain the proxy selected by the cluster module without direct fallback."""
    monkeypatch.setattr(network.socket, "gethostname", lambda: "della-r3c4n2")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("LOADEDMODULES", "other/module:proxy/default")
    monkeypatch.setenv("https_proxy", "http://cluster-proxy:3128")
    assert network.cluster_proxy() == ("cluster-proxy", 3128)
    monkeypatch.setenv("LOADEDMODULES", "other/module")
    with pytest.raises(ValueError, match="proxy/default"):
        network.cluster_proxy()


@pytest.mark.parametrize("proxy_status", [200, 403])
def test_cluster_proxy_connect_has_no_direct_fallback(tunnel, monkeypatch, proxy_status):
    """Honor the cluster proxy's allowlist and preserve its opaque TLS channel."""
    upstream, peer = socket.socketpair()
    destinations = []
    received = []

    def connect(destination, timeout):
        destinations.append(destination)
        return upstream

    def proxy_response():
        with peer:
            peer.settimeout(2)
            received.append(read_header(peer))
            peer.sendall(f"HTTP/1.1 {proxy_status} Result\r\n\r\n".encode())
            if proxy_status == 200:
                peer.sendall(b"opaque-tls")

    monkeypatch.setattr(network.socket, "create_connection", connect)
    monkeypatch.setattr(network.UnixServer, "cluster_proxy", ("cluster-proxy", 3128), raising=False)
    thread = threading.Thread(target=proxy_response)
    thread.start()
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(2)
        client.connect(tunnel)
        client.sendall(b"CONNECT openrouter.ai:443 HTTP/1.1\r\n\r\n")
        expected = b"200" if proxy_status == 200 else b"502"
        assert read_header(client).split(b" ")[1] == expected
        if proxy_status == 200:
            assert client.recv(100) == b"opaque-tls"
    thread.join(timeout=2)
    assert destinations == [("cluster-proxy", 3128)]
    assert received == [b"CONNECT openrouter.ai:443 HTTP/1.1\r\nHost: openrouter.ai:443\r\n\r\n"]
