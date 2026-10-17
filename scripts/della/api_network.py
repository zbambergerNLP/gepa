"""Keep Terminal-Bench's private loopback while tunneling approved OpenRouter HTTPS.

The tunnel carries TLS unchanged. It has no model key, provider fallback, general
internet forwarding. Allocated compute nodes retain the cluster's approved proxy.
"""

import argparse
import os
import select
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlsplit


def cluster_proxy():
    """Use direct vis access or the loaded proxy/default on an allocated Della node."""
    host = socket.gethostname().split(".")[0]
    if host.startswith("della-vis"):
        return None
    if (
        not host.startswith("della-")
        or not os.environ.get("SLURM_JOB_ID")
        or "proxy/default" not in os.environ.get("LOADEDMODULES", "").split(":")
    ):
        raise ValueError("Use Della vis for preparation, or a Della Slurm allocation with module load proxy/default")
    proxy = urlsplit(os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or "")
    if proxy.scheme != "http" or not proxy.hostname or proxy.username or proxy.password or proxy.path not in ("", "/"):
        raise ValueError("proxy/default must provide an unauthenticated HTTP CONNECT proxy")
    return proxy.hostname, proxy.port or 80


def relay(left, right):
    """Forward opaque TLS bytes until either peer closes its connection."""
    while True:
        readable, _, _ = select.select([left, right], [], [], 1200)
        if not readable:
            raise TimeoutError("OpenRouter tunnel idle for 1200 seconds")
        for source in readable:
            try:
                data = source.recv(65536)
                if not data:
                    return
                (right if source is left else left).sendall(data)
            except (ConnectionResetError, BrokenPipeError):
                return


class OpenRouterConnect(socketserver.BaseRequestHandler):
    """Accept only HTTPS CONNECT to the configured benchmark provider."""

    def handle(self):
        """Validate the destination before opening a remote connection."""
        self.request.settimeout(20)
        header = b""
        while not header.endswith(b"\r\n\r\n") and len(header) < 8192:
            byte = self.request.recv(1)
            if not byte:
                return
            header += byte
        if header.split(b"\r\n", 1)[0] not in (
            b"CONNECT openrouter.ai:443 HTTP/1.1",
            b"CONNECT openrouter.ai:443 HTTP/1.0",
        ) or not header.endswith(b"\r\n\r\n"):
            self.request.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            return
        proxy = getattr(self.server, "cluster_proxy", None)
        with socket.create_connection(proxy or ("openrouter.ai", 443), timeout=30) as upstream:
            if proxy:
                upstream.sendall(b"CONNECT openrouter.ai:443 HTTP/1.1\r\nHost: openrouter.ai:443\r\n\r\n")
                response = b""
                while not response.endswith(b"\r\n\r\n") and len(response) < 8192:
                    byte = upstream.recv(1)
                    if not byte:
                        break
                    response += byte
                if not response.endswith(b"\r\n\r\n") or response.split(b" ")[1:2] != [b"200"]:
                    self.request.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
                    return
            upstream.settimeout(None)
            self.request.settimeout(None)
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            relay(self.request, upstream)


class NamespaceBridge(socketserver.BaseRequestHandler):
    """Bridge private loopback to the owner's filesystem socket."""

    def handle(self):
        """Forward one connection without reading credentials or TLS contents."""
        with socket.socket(socket.AF_UNIX) as upstream:
            upstream.connect(self.server.upstream)
            relay(self.request, upstream)


class UnixServer(socketserver.ThreadingUnixStreamServer):
    """Keep tunnel connections scoped to this wrapper's lifetime."""

    daemon_threads = True


class TCPServer(socketserver.ThreadingTCPServer):
    """Serve only within the new private network namespace."""

    daemon_threads = True


def main():
    """Start a private namespace and stop its tunnel when the command exits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace-socket", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Supply a command after --")
    try:
        proxy = cluster_proxy()
    except ValueError as error:
        parser.error(str(error))
    if args.namespace_socket:
        with TCPServer(("127.0.0.1", 0), NamespaceBridge) as server:
            server.upstream = str(args.namespace_socket)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            proxy = f"http://127.0.0.1:{server.server_address[1]}"
            env = {
                **os.environ,
                "HTTPS_PROXY": proxy,
                "https_proxy": proxy,
                "HTTP_PROXY": proxy,
                "http_proxy": proxy,
                "NO_PROXY": "localhost,127.0.0.1,::1",
                "no_proxy": "localhost,127.0.0.1,::1",
            }
            return subprocess.call(command, env=env)
    with tempfile.TemporaryDirectory(prefix="gepa-openrouter-") as directory:
        socket_path = Path(directory) / "connect.sock"
        with UnixServer(str(socket_path), OpenRouterConnect) as server:
            server.cluster_proxy = proxy
            os.chmod(socket_path, 0o600)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            child_command = [
                "uv",
                "run",
                "--no-project",
                "--python",
                sys.executable,
                "python",
                str(Path(__file__).resolve()),
                "--namespace-socket",
                str(socket_path),
                "--",
                *command,
            ]
            child = subprocess.Popen(
                [
                    "unshare",
                    "--user",
                    "--map-root-user",
                    "--net",
                    "--",
                    "bash",
                    "-euc",
                    'ip link set lo up; exec "$@"',
                    "gepa-api-network",
                    *child_command,
                ],
                start_new_session=True,
            )
            try:
                return child.wait()
            finally:
                # Reap only the process group this wrapper created, including
                # task processes left behind by an interrupted command.
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
