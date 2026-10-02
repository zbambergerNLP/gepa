"""Keep a denied external API from starting expensive local model servers."""

import json

import httpx2
import pytest

from examples.hotpotqa.typesafe_preflight import check_connectivity


@pytest.mark.parametrize("status", [200, 401, 404, 405])
def test_preflight_reaches_origin_without_credentials_or_inference(status):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx2.Response(status)

    with httpx2.Client(transport=httpx2.MockTransport(handle)) as client:
        result = check_connectivity(client)
    assert result == {"status": "PASS", "http_status": status}
    assert len(requests) == 1
    assert requests[0].method == "HEAD"
    assert requests[0].url.host == "api.typesafe.ai"
    assert "authorization" not in requests[0].headers


def test_preflight_proxy_denial_is_sanitized_and_not_retried():
    requests = []

    def handle(request):
        requests.append(request)
        raise httpx2.ProxyError("403 proxy denied; sensitive proxy details")

    with httpx2.Client(transport=httpx2.MockTransport(handle)) as client:
        result = check_connectivity(client)
    assert result == {"status": "FAIL", "error_type": "ProxyError", "http_status": None}
    assert len(requests) == 1
    assert "sensitive" not in json.dumps(result)


@pytest.mark.parametrize("status", [403, 407, 429, 500, 503])
def test_preflight_rejects_unavailable_origin(status):
    with httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(status))) as client:
        assert check_connectivity(client)["status"] == "FAIL"
