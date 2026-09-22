"""Apply the pinned DeepSeek runtime's reasoning-boundary sampling repair."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
from pathlib import Path
from typing import Any

VLLM_VERSION = "0.1.1.dev5+ge77daef89"
VLLM_MODULE = "vllm.v1.worker.gpu.sample.thinking_budget"
VLLM_SOURCE_SHA256 = "020513e1d3406f367de6ad995a95055008fd0c28f2873647e8283c9870e22404"
_ORIGINAL = "    tl.store(logits_ptr + token_idx * logits_stride + force_token_id, 1.0e9)"
_REPLACEMENT = """    # A huge finite logit loses precision in top-p filtering and can mask
    # every token. Force a categorical outcome without an extreme logit.
    for block_start in range(0, logits_stride, 1024):
        token_ids = block_start + tl.arange(0, 1024)
        forced_logits = tl.where(token_ids == force_token_id, 0.0, -float("inf"))
        tl.store(
            logits_ptr + token_idx * logits_stride + token_ids,
            forced_logits,
            token_ids < logits_stride,
        )"""


def repaired_source(source: str) -> str:
    """Replace only the verified kernel's unsafe forced-logit operation."""
    if source.count(_ORIGINAL) != 1:
        raise RuntimeError("The reasoning-budget kernel no longer matches the reviewed repair.")
    return source.replace(_ORIGINAL, _REPLACEMENT)


def install() -> str:
    """Repair the exact pinned kernel before it is compiled in each worker.

    Triton's source-update method invalidates the kernel's compilation key.
    This kernel is launched directly from Python, with no compiled Triton
    callers to invalidate. Installed package files remain byte-identical;
    the launcher opts in explicitly and pins this repair with the run source.
    """
    if importlib.metadata.version("vllm") != VLLM_VERSION:
        raise RuntimeError("The reasoning-budget repair requires the pinned DeepSeek vLLM version.")
    module: Any = importlib.import_module(VLLM_MODULE)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != VLLM_SOURCE_SHA256:
        raise RuntimeError("The installed reasoning-budget implementation differs from the reviewed source.")
    if getattr(module, "_gepa_safe_thinking_budget", None) is not None:
        return module._gepa_safe_thinking_budget
    kernel = module._thinking_budget_kernel
    if kernel.hash is not None:
        raise RuntimeError("Install the reasoning-budget repair before the kernel is compiled.")
    source = repaired_source(kernel.src)
    kernel._unsafe_update_src(source)
    original_apply = module.apply_thinking_budget

    def apply(logits: Any, *args: Any, **kwargs: Any) -> Any:
        """Require the contiguous rows used by the pinned V2 sampler."""
        if logits.ndim != 2 or logits.stride(1) != 1 or logits.stride(0) != logits.shape[1]:
            raise RuntimeError("The reasoning-budget repair requires contiguous logit rows.")
        return original_apply(logits, *args, **kwargs)

    module.apply_thinking_budget = apply
    module._gepa_safe_thinking_budget = hashlib.sha256(source.encode()).hexdigest()
    return module._gepa_safe_thinking_budget
