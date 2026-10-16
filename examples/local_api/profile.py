"""Freeze OpenRouter routes and prices without claiming local checkpoint identity."""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from examples.common.provider_retries import provider_retry_kwargs
from gepa.strategies.forest_constants import OPTIMIZER_ROLE, SOLVER_ROLE

API_BASE = "https://openrouter.ai/api/v1"
BENCHMARKS = ("hotpotqa", "decisionbench", "appworld", "obliqbench", "taubench", "terminalbench")
ROLES = ("solver", "proposer", "tau")
TAU_MODEL = "openai/gpt-4.1"
PARAMETERS = {"max_tokens", "temperature", "top_p", "top_k", "seed", "reasoning_effort"}
JEV_TRANSPORT = {
    "model": "typesafe/jev-1.13",
    "response_model": "typesafe/jev-1.13-20260917",
    "api_base": "https://openrouter.ai/api",
}


def get_json(url: str, *, key: str | None = None) -> dict:
    """Read public metadata or authenticate without putting credentials in URLs."""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        return json.load(response)


def endpoints(model: str) -> list[dict]:
    """Fetch current provider-level prices, capacities and supported parameters."""
    return get_json(f"{API_BASE}/models/{model}/endpoints")["data"]["endpoints"]


def validate_parameters(parameters: dict) -> None:
    """Require an output ceiling and reject unrecorded provider-specific overrides."""
    if not isinstance(parameters, dict) or set(parameters) - PARAMETERS:
        raise ValueError("Unsupported decoding fields")
    if type(parameters.get("max_tokens")) is not int or parameters["max_tokens"] < 1:
        raise ValueError("Decoding parameters require a positive integer max_tokens")
    for field in ("temperature", "top_p"):
        if field in parameters and (
            type(parameters[field]) not in (int, float) or not math.isfinite(parameters[field]) or parameters[field] < 0
        ):
            raise ValueError(f"Invalid {field}")
    if "top_p" in parameters and parameters["top_p"] > 1:
        raise ValueError("top_p must not exceed one")
    for field in ("seed", "top_k"):
        if field in parameters and (type(parameters[field]) is not int or parameters[field] < 0):
            raise ValueError(f"Invalid {field}")
    if "reasoning_effort" in parameters and parameters["reasoning_effort"] not in {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    }:
        raise ValueError("Unsupported reasoning effort")


def eligible(endpoint: dict, model: str, parameters: dict, *, tau: bool = False) -> bool:
    """Require the exact chosen model, capacity, native tools and requested controls."""
    supported = set(endpoint.get("supported_parameters", []))
    if "max_completion_tokens" in supported:
        supported.add("max_tokens")
    required = (set(parameters) - {"reasoning_effort"}) | {"tools", "tool_choice"}
    if "reasoning_effort" in parameters:
        required.add("reasoning")
    return (
        endpoint.get("status") == 0
        and endpoint.get("model_id") == model
        and (endpoint.get("max_completion_tokens") or 0) >= parameters["max_tokens"]
        and endpoint.get("context_length", 0) > parameters["max_tokens"]
        and required <= supported
        and (not tau or (model == TAU_MODEL and "2025-04-14" in endpoint.get("name", "")))
    )


def route_record(endpoint: dict) -> dict:
    """Keep material route facts while excluding volatile uptime and latency data."""
    return {
        key: deepcopy(endpoint[key])
        for key in (
            "model_id",
            "tag",
            "provider_name",
            "name",
            "quantization",
            "context_length",
            "max_completion_tokens",
            "pricing",
            "supported_parameters",
        )
    }


def make_profile(
    solver_model: str,
    proposer_model: str,
    *,
    solver_parameters: dict | None = None,
    proposer_parameters: dict | None = None,
    input_output_ratio: float = 5.0,
) -> dict:
    """Choose prices only after the user chooses models and decoding requirements."""
    if not math.isfinite(input_output_ratio) or input_output_ratio < 0:
        raise ValueError("Input/output ratio must be finite and nonnegative")
    choices = {
        "solver": (solver_model, solver_parameters or {"max_tokens": 32_768}),
        "proposer": (proposer_model, proposer_parameters or {"max_tokens": 32_768}),
        "tau": (TAU_MODEL, {"max_tokens": 32_768, "temperature": 0.0}),
    }
    roles = {}
    for role, (model, parameters) in choices.items():
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+/[A-Za-z0-9_./:-]+", model):
            raise ValueError("Use an explicit OpenRouter provider/model ID")
        validate_parameters(parameters)
        candidates = [row for row in endpoints(model) if eligible(row, model, parameters, tau=role == "tau")]
        if not candidates:
            raise ValueError(f"No compatible, available endpoint for {role}: {model}")
        chosen = min(
            candidates,
            key=lambda row: (
                Decimal(row["pricing"]["prompt"]) * Decimal(str(input_output_ratio))
                + Decimal(row["pricing"]["completion"]),
                row["tag"],
            ),
        )
        roles[role] = {**route_record(chosen), "parameters": deepcopy(parameters)}
    return {
        "schema_version": 2,
        "provider": "openrouter",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "selection": {"uncached_input_tokens_per_output_token": input_output_ratio, "fallbacks": False},
        "roles": roles,
        "jev": dict(JEV_TRANSPORT),
        "compatibility": "New hosted campaign. Model choices and request parameters are explicit; omitted "
        "sampling parameters use provider defaults. Quantization is catalog-reported; checkpoint revisions "
        "are unavailable. No Della checkpoint, baseline or timing identity is reused.",
    }


def load_profile(path: Path) -> dict:
    """Reject secrets, arbitrary endpoints and incomplete route identities."""
    profile = json.loads(path.read_text())
    if set(profile) != {"schema_version", "provider", "observed_at", "selection", "roles", "jev", "compatibility"}:
        raise ValueError("Unexpected API profile fields; never put credentials in a profile")
    if profile["schema_version"] != 2 or profile["provider"] != "openrouter" or set(profile["roles"]) != set(ROLES):
        raise ValueError("Unsupported API profile")
    if profile["jev"] != JEV_TRANSPORT:
        raise ValueError("Unpinned Jev transport")
    for role, route in profile["roles"].items():
        if set(route) != {
            "model_id",
            "tag",
            "provider_name",
            "name",
            "quantization",
            "context_length",
            "max_completion_tokens",
            "pricing",
            "supported_parameters",
            "parameters",
        }:
            raise ValueError("Incomplete or unexpected route identity")
        validate_parameters(route["parameters"])
        if not eligible({**route, "status": 0}, route["model_id"], route["parameters"], tau=role == "tau"):
            raise ValueError(f"Incompatible {role} route")
        for field in ("prompt", "completion"):
            price = Decimal(route["pricing"][field])
            if not price.is_finite() or price < 0:
                raise ValueError("Invalid token price")
    return profile


def verify_routes(profile: dict, roles: tuple[str, ...] = ("solver", "proposer", "tau")) -> None:
    """Fail closed on route drift, unavailability or a price increase; never reroute."""
    for role in roles:
        saved = profile["roles"][role]
        current = next((row for row in endpoints(saved["model_id"]) if row["tag"] == saved["tag"]), None)
        if current is None or not eligible(current, saved["model_id"], saved["parameters"], tau=role == "tau"):
            raise ValueError(f"Pinned {role} route unavailable or no longer compatible")
        if not set(saved["supported_parameters"]) <= set(current["supported_parameters"]):
            raise ValueError(f"Pinned {role} route lost parameter support")
        for field in ("model_id", "provider_name", "name", "quantization", "context_length", "max_completion_tokens"):
            if current[field] != saved[field]:
                raise ValueError(f"Pinned {role} route changed {field}; use a new campaign after review")
        if any(
            Decimal(current["pricing"][field]) > Decimal(saved["pricing"][field]) for field in ("prompt", "completion")
        ):
            raise ValueError(f"Pinned {role} route exceeds its saved price cap")


def provider_options(route: dict) -> dict:
    """Pin one endpoint and cap both input and output prices in dollars per million."""
    return {
        "only": [route["tag"]],
        "allow_fallbacks": False,
        "require_parameters": True,
        "max_price": {
            name: float(Decimal(route["pricing"][field]) * 1_000_000)
            for name, field in (("prompt", "prompt"), ("completion", "completion"))
        },
    }


def role_kwargs(profile: dict, role: str) -> dict:
    """Translate hosted controls explicitly while retaining output ceilings and retries."""
    route = profile["roles"][role]
    kwargs = {
        "timeout": 3600,
        "num_retries": 0,
        "drop_params": False,
        **provider_retry_kwargs(role=SOLVER_ROLE if role == "solver" else OPTIMIZER_ROLE),
        "api_base": API_BASE,
        **deepcopy(route["parameters"]),
        "extra_body": {"provider": provider_options(route)},
    }
    if "reasoning_effort" in kwargs:
        kwargs["extra_body"]["reasoning"] = {"effort": kwargs.pop("reasoning_effort")}
    if "top_k" in kwargs:
        kwargs["extra_body"]["top_k"] = kwargs.pop("top_k")

    return kwargs


def tau_settings(profile: dict | None) -> dict:
    """Keep Tau's official model while routing both simulator and judge explicitly."""
    from examples.taubench.model_settings import USER_KWARGS, USER_MODEL

    if profile is None:
        return {"model": USER_MODEL, "kwargs": deepcopy(USER_KWARGS)}
    return {
        "model": "openrouter/" + TAU_MODEL,
        "official_model": USER_MODEL,
        "kwargs": {
            **deepcopy(profile["roles"]["tau"]["parameters"]),
            "timeout": USER_KWARGS["timeout"],
            "api_base": API_BASE,
            "extra_body": {"provider": provider_options(profile["roles"]["tau"])},
        },
    }


def model_info(profile: dict, role: str) -> dict:
    """Use the chosen hosted context ceiling and catalog rates, never local zero costs."""
    route = profile["roles"][role]
    return {
        "max_input_tokens": route["context_length"],
        "max_output_tokens": route["parameters"]["max_tokens"],
        "input_cost_per_token": float(route["pricing"]["prompt"]),
        "output_cost_per_token": float(route["pricing"]["completion"]),
    }


def load_credential(env_file: Path | None = None) -> str:
    """Load only the authorized key; do not import unrelated dotenv credentials."""
    if env_file is not None:
        from dotenv import dotenv_values

        key = dotenv_values(env_file, interpolate=False).get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("The supplied dotenv has no OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = key
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY or supply --env-file; never put a key in a profile")
    return key
