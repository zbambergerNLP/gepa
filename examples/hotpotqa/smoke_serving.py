"""Record one complete request/response exchange with a locally served HotPotQA model.

This is a standalone diagnostic run by ``scripts/della/verify_deepseek_serving.sh``
on a GPU node. It sends one simple chat prompt with the campaign's exact request
settings and writes a transcript containing what went in, the prompt text the
server actually rendered from it, and everything that came back, including the
reasoning tokens. It is not part of any campaign job and writes nothing outside
its output directory.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from examples.common.provider_retries import complete_with_retries
from examples.hotpotqa.utils import resolve_hotpotqa_lm_kwargs

SMOKE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France? Answer with one word."},
]
# LiteLLM-only settings that are not part of the JSON body it sends to the server.
_CLIENT_ONLY_FIELDS = {
    "api_base",
    "num_retries",
    "max_retries",
    "timeout",
    "extra_body",
    "model_info",
    "cache",
    "_gepa_provider_retry",
}


def _post_json(
    url: str, payload: dict[str, Any], timeout: float, *, attempt_log: Path | None = None
) -> dict[str, Any]:
    """POST one JSON payload and decode the JSON response.

    Args:
        url: Absolute endpoint URL on the local server.
        payload: Request body.
        timeout: Seconds to wait for the response.
        attempt_log: Physical-attempt log for model-generating requests.

    Returns:
        Decoded JSON response body.
    """
    def send(**kwargs: Any) -> dict[str, Any]:
        """Send the raw protocol unchanged after removing client-only retry metadata."""
        remaining = kwargs.pop("timeout", timeout)
        kwargs.pop("num_retries", None)
        kwargs.pop("max_retries", None)
        request = urllib.request.Request(
            url,
            data=json.dumps(kwargs).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=remaining) as response:
            return json.loads(response.read())

    if url.rstrip("/").endswith("/chat/completions"):
        return complete_with_retries(
            send, {**payload, "timeout": timeout},
            {"role": "serving_probe", "log_path": str(attempt_log) if attempt_log is not None else None},
        )
    return send(**payload)


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    """GET one JSON document from the local server.

    Args:
        url: Absolute endpoint URL on the local server.
        timeout: Seconds to wait for the response.

    Returns:
        Decoded JSON response body.
    """
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def build_chat_request(model: str, served_name: str, api_base: str) -> dict[str, Any]:
    """Build the chat-completions body the campaign's client sends for one prompt.

    LiteLLM merges ``extra_body`` into the top-level JSON body, so its
    ``chat_template_kwargs`` are lifted here exactly as the server receives them.

    Args:
        model: Exact LiteLLM model identifier of the campaign arm.
        served_name: Model name the local server reports on ``/v1/models``.
        api_base: Local OpenAI-compatible ``/v1`` endpoint.

    Returns:
        JSON-serializable chat-completions request body.
    """
    lm_kwargs = resolve_hotpotqa_lm_kwargs(model, api_base)
    body: dict[str, Any] = {"model": served_name, "messages": SMOKE_MESSAGES}
    body.update({key: value for key, value in lm_kwargs.items() if key not in _CLIENT_ONLY_FIELDS})
    extra_body = lm_kwargs.get("extra_body")
    if isinstance(extra_body, dict):
        body.update(extra_body)
    return body


def run_smoke_exchange(
    model: str, served_name: str, api_base: str, timeout: float, *, attempt_log: Path | None = None
) -> dict[str, Any]:
    """Render, send, and record one chat exchange.

    Args:
        model: Exact LiteLLM model identifier of the campaign arm.
        served_name: Model name the local server reports on ``/v1/models``.
        api_base: Local OpenAI-compatible ``/v1`` endpoint.
        timeout: Seconds to wait for each server call.
        attempt_log: Optional path for all model request attempts.

    Returns:
        Transcript with the request, the server-rendered prompt, and the response.
    """
    parsed = urlsplit(api_base)
    server_root = f"{parsed.scheme}://{parsed.netloc}"
    chat_request = build_chat_request(model, served_name, api_base)
    tokenize_request = {
        "model": served_name,
        "messages": SMOKE_MESSAGES,
        "add_generation_prompt": True,
    }
    if "chat_template_kwargs" in chat_request:
        tokenize_request["chat_template_kwargs"] = chat_request["chat_template_kwargs"]
    tokenized = _post_json(f"{server_root}/tokenize", tokenize_request, timeout)
    detokenized = _post_json(
        f"{server_root}/detokenize",
        {"model": served_name, "tokens": tokenized["tokens"]},
        timeout,
    )
    started = time.monotonic()
    response = _post_json(f"{api_base.rstrip('/')}/chat/completions", chat_request, timeout, attempt_log=attempt_log)
    elapsed = time.monotonic() - started
    message = response["choices"][0]["message"]
    try:
        server_version = _get_json(f"{server_root}/version", timeout).get("version")
    except OSError:
        server_version = None
    return {
        "model": model,
        "served_name": served_name,
        "server_version": server_version,
        "served_models": _get_json(f"{api_base.rstrip('/')}/models", timeout),
        "request": chat_request,
        "rendered_prompt": detokenized["prompt"],
        "rendered_prompt_token_count": len(tokenized["tokens"]),
        "response": response,
        "reasoning": message.get("reasoning") or message.get("reasoning_content"),
        "content": message.get("content"),
        "finish_reason": response["choices"][0].get("finish_reason"),
        "usage": response.get("usage"),
        "elapsed_seconds": round(elapsed, 2),
    }


def render_markdown(transcript: dict[str, Any]) -> str:
    """Render a transcript as a human-readable Markdown report.

    Args:
        transcript: Output of :func:`run_smoke_exchange`.

    Returns:
        Markdown text with each stage of the exchange in its own section.
    """
    fence = "````"
    sections = [
        f"# Serving smoke test: {transcript['served_name']}",
        f"vLLM {transcript['server_version']}; finish reason `{transcript['finish_reason']}`; "
        f"{transcript['elapsed_seconds']} s; usage `{json.dumps(transcript['usage'])}`",
        "## Request sent to /v1/chat/completions",
        f"{fence}json\n{json.dumps(transcript['request'], indent=2)}\n{fence}",
        f"## Prompt rendered by the server ({transcript['rendered_prompt_token_count']} tokens)",
        f"{fence}text\n{transcript['rendered_prompt']}\n{fence}",
        "## Reasoning returned by the server",
        f"{fence}text\n{transcript['reasoning']}\n{fence}",
        "## Final content returned by the server",
        f"{fence}text\n{transcript['content']}\n{fence}",
        "## Raw response",
        f"{fence}json\n{json.dumps(transcript['response'], indent=2)}\n{fence}",
    ]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    """Parse the CLI, run one exchange, and write the JSON and Markdown transcripts."""
    parser = argparse.ArgumentParser(description="Record one full exchange with a locally served model")
    parser.add_argument("--model", required=True, help="Exact LiteLLM model identifier of the campaign arm")
    parser.add_argument("--served-name", required=True, help="Model name reported by /v1/models")
    parser.add_argument("--api-base", required=True, help="Local OpenAI-compatible /v1 endpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for transcript.json/.md")
    parser.add_argument("--timeout", type=float, default=3600, help="Per-call timeout in seconds")
    args = parser.parse_args()
    transcript = run_smoke_exchange(
        args.model, args.served_name, args.api_base, args.timeout,
        attempt_log=args.output_dir / "provider-attempts.jsonl",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "transcript.json").write_text(json.dumps(transcript, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "transcript.md").write_text(render_markdown(transcript), encoding="utf-8")
    print(f"content: {transcript['content']!r}")
    print(f"reasoning characters: {len(transcript['reasoning'] or '')}")
    print(f"wrote {args.output_dir / 'transcript.json'} and transcript.md")
    if not transcript["content"] or not transcript["reasoning"]:
        raise SystemExit("Smoke exchange returned no content or no reasoning; see the transcript.")


if __name__ == "__main__":
    main()
