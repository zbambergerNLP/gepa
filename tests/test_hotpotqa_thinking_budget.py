"""Reject incompatible runtimes before installing the tested GPU kernel repair."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from examples.common.experiment_models import DEEPSEEK_V4_1_FLASH_MODEL, QWEN3_8_27B_MODEL
from examples.hotpotqa import verify_thinking_budget as probe
from examples.hotpotqa.serving import safe_thinking_budget as repair
from examples.hotpotqa.utils import resolve_hotpotqa_lm_kwargs


@pytest.mark.parametrize("model", [DEEPSEEK_V4_1_FLASH_MODEL, QWEN3_8_27B_MODEL])
def test_reasoning_budget_is_solver_specific(model):
    """Reserve solver answer space without interrupting optimizer tool arguments."""
    solver = resolve_hotpotqa_lm_kwargs(model, None)
    optimizer = resolve_hotpotqa_lm_kwargs(model, None, role="optimizer")
    assert solver["extra_body"]["thinking_token_budget"] == 32_768
    assert "thinking_token_budget" not in optimizer["extra_body"]
    assert solver["max_tokens"] == 65_536
    assert optimizer["max_tokens"] == (131_072 if model == DEEPSEEK_V4_1_FLASH_MODEL else 32_768)


@pytest.mark.parametrize(
    "reasoning,finish,content,status",
    [
        (0, "stop", "ready", "PASS"),
        (64, "stop", "ready", "FAIL"),
        (None, "stop", "ready", "PASS"),
        (0, "length", "ready", "FAIL"),
        (0, "stop", None, "FAIL"),
    ],
)
def test_boundary_probe_requires_observed_budget_and_final_content(
    tmp_path, monkeypatch, reasoning, finish, content, status
):
    """Fail before qualification when budget handling is broken or unobservable."""
    response = {
        "id": "probe-response",
        "choices": [{"message": {"content": content}, "finish_reason": finish, "token_ids": [128822, 5]}],
        "usage": {"completion_tokens_details": {"reasoning_tokens": reasoning}},
    }
    post = Mock(side_effect=[{"tokens": [128822]}, response])
    monkeypatch.setattr(probe, "_post_json", post)
    output = tmp_path / "probe"
    report = probe.verify_thinking_budget(DEEPSEEK_V4_1_FLASH_MODEL, "http://127.0.0.1:8000/v1", output)
    assert report["status"] == status
    request = post.call_args.args[1]
    assert request["thinking_token_budget"] == 0 and request["max_tokens"] == 256
    assert request["top_p"] == 0.95 and request["temperature"] == 1.0
    assert request["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": 75}
    assert json.loads((output / "response.json").read_text()) == response
    assert (output / "request.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        probe.verify_thinking_budget(DEEPSEEK_V4_1_FLASH_MODEL, "http://127.0.0.1:8000/v1", output)


@pytest.mark.parametrize("token_ids", [None, [], [0, 0, 0], [4, 128822]])
def test_boundary_probe_rejects_missing_or_delayed_forcing(tmp_path, monkeypatch, token_ids):
    """Catch the observed BOS failure even when a server omits reasoning usage."""
    response = {"choices": [{"message": {"content": "ready"}, "finish_reason": "stop", "token_ids": token_ids}]}
    monkeypatch.setattr(probe, "_post_json", Mock(side_effect=[{"tokens": [128822]}, response]))
    report = probe.verify_thinking_budget(DEEPSEEK_V4_1_FLASH_MODEL, "http://127.0.0.1:8000/v1", tmp_path / "probe")
    assert report["status"] == "FAIL" and report["boundary_verified"] is False


def test_repair_requires_the_reviewed_forcing_operation():
    """Fail closed if an upstream change removes or duplicates the patch location."""
    for source in ("def unrelated(): pass", repair._ORIGINAL + "\n" + repair._ORIGINAL):
        with pytest.raises(RuntimeError, match="no longer matches"):
            repair.repaired_source(source)


def test_repair_rejects_a_different_vllm_release(monkeypatch):
    """Avoid silently applying the experimental kernel repair after an upgrade."""
    monkeypatch.setattr(repair.importlib.metadata, "version", lambda _: "different-version")
    with pytest.raises(RuntimeError, match="pinned DeepSeek"):
        repair.install()


@pytest.mark.parametrize("wrong_source,already_compiled", [(True, False), (False, True)])
def test_repair_rejects_source_drift_and_late_installation(tmp_path, monkeypatch, wrong_source, already_compiled):
    """Require exact source bytes and install before any kernel compilation."""
    path = tmp_path / "thinking_budget.py"
    path.write_text("reviewed source")
    monkeypatch.setattr(repair.importlib.metadata, "version", lambda _: repair.VLLM_VERSION)
    monkeypatch.setattr(
        repair, "VLLM_SOURCE_SHA256", "wrong" if wrong_source else hashlib.sha256(path.read_bytes()).hexdigest()
    )
    kernel = SimpleNamespace(hash="compiled" if already_compiled else None, _unsafe_update_src=Mock())
    module = SimpleNamespace(__file__=str(path), _thinking_budget_kernel=kernel)
    monkeypatch.setattr(repair.importlib, "import_module", lambda _: module)
    with pytest.raises(RuntimeError, match="reviewed source|before the kernel"):
        repair.install()
    kernel._unsafe_update_src.assert_not_called()


def test_repair_is_idempotent_and_rejects_unsupported_logit_layout(tmp_path, monkeypatch):
    """Do not repatch a compiled kernel or write into padded/noncontiguous rows."""
    path = tmp_path / "thinking_budget.py"
    path.write_text("reviewed source")
    monkeypatch.setattr(repair.importlib.metadata, "version", lambda _: repair.VLLM_VERSION)
    monkeypatch.setattr(repair, "VLLM_SOURCE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    kernel = SimpleNamespace(src="def kernel():\n" + repair._ORIGINAL, hash=None, _unsafe_update_src=Mock())
    original = Mock(return_value="unchanged")
    module = SimpleNamespace(__file__=str(path), _thinking_budget_kernel=kernel, apply_thinking_budget=original)
    monkeypatch.setattr(repair.importlib, "import_module", lambda _: module)
    identity = repair.install()
    kernel.hash = "now-compiled"
    assert repair.install() == identity
    kernel._unsafe_update_src.assert_called_once()
    contiguous = SimpleNamespace(ndim=2, shape=(1, 129280), stride=lambda axis: (129280, 1)[axis])
    assert module.apply_thinking_budget(contiguous, "argument") == "unchanged"
    original.assert_called_once_with(contiguous, "argument")
    for strides in [(129300, 1), (129280, 2)]:
        logits = SimpleNamespace(ndim=2, shape=(1, 129280), stride=lambda axis, shape=strides: shape[axis])
        with pytest.raises(RuntimeError, match="contiguous"):
            module.apply_thinking_budget(logits)
    original.assert_called_once()
