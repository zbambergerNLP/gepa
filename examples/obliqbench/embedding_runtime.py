"""Support explicit memory-efficient embedding runtimes without reusing incompatible indexes."""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from examples.obliqbench.benchmark_settings import EMBEDDING_MAX_LENGTH, EMBEDDING_MODEL, EMBEDDING_REVISION
from examples.obliqbench.retrieval import QwenEncoder as LegacyQwenEncoder
from examples.obliqbench.retrieval import file_sha256
from examples.obliqbench.retrieval import retrieval_contract as legacy_retrieval_contract


def retrieval_contract(
    backend: str, device: str, batch_size: int, *, attention: str = "eager", dtype: str = "float32"
) -> dict[str, Any]:
    """Keep legacy cache keys intact and bind explicit new numerical settings to new indexes."""
    if attention not in {"eager", "sdpa"} or dtype not in {"float32", "bfloat16"}:
        raise ValueError("Choose embedding attention eager/sdpa and dtype float32/bfloat16")
    contract = legacy_retrieval_contract(backend, device, batch_size)
    if (attention, dtype) == ("eager", "float32"):
        return contract
    if backend != "qwen":
        raise ValueError("Embedding attention/dtype settings require the qwen retriever")
    contract["settings"].update({"attention": attention, "dtype": dtype})
    if attention == "sdpa":
        contract["settings"]["cuda_sdpa_policy"] = "fused_only_no_math_fallback"
    contract["encoder_implementation_sha256"] = file_sha256(Path(__file__))
    return contract


class QwenEncoder(LegacyQwenEncoder):
    """Retain the published encoding protocol with an explicitly selected attention implementation."""

    def __init__(self, device: str, *, attention: str = "eager", dtype: str = "float32"):
        if attention not in {"eager", "sdpa"} or dtype not in {"float32", "bfloat16"}:
            raise ValueError("Choose embedding attention eager/sdpa and dtype float32/bfloat16")
        self._fused_cuda_sdpa = attention == "sdpa" and device.startswith("cuda")
        if (attention, dtype) == ("eager", "float32"):
            super().__init__(device)
            return
        torch = importlib.import_module("torch")
        sentence_transformers = importlib.import_module("sentence_transformers")
        self.model = sentence_transformers.SentenceTransformer(
            EMBEDDING_MODEL,
            revision=EMBEDDING_REVISION,
            device=device,
            trust_remote_code=False,
            model_kwargs={"dtype": getattr(torch, dtype), "attn_implementation": attention},
            processor_kwargs={"padding_side": "left"},
        )
        self.model.max_seq_length = EMBEDDING_MAX_LENGTH
        if not self.model.prompts.get("query"):
            raise ValueError("Pinned embedding checkpoint is missing its query prompt")

    def encode(self, texts: list[str], *, query: bool) -> np.ndarray:
        """Use supported fused CUDA kernels or fail instead of silently allocating quadratic attention."""
        context = nullcontext()
        if self._fused_cuda_sdpa:
            attention = importlib.import_module("torch.nn.attention")
            context = attention.sdpa_kernel(
                [
                    attention.SDPBackend.FLASH_ATTENTION,
                    attention.SDPBackend.EFFICIENT_ATTENTION,
                    attention.SDPBackend.CUDNN_ATTENTION,
                ]
            )
        with context:
            return super().encode(texts, query=query)
