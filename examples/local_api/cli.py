"""Prepare local API profiles and bounded connection checks; never launch benchmarks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from examples.common.artifacts import atomic_json
from examples.local_api.profile import (
    API_BASE,
    BENCHMARKS,
    get_json,
    load_credential,
    load_profile,
    make_profile,
    role_kwargs,
    tau_settings,
    verify_routes,
)


def doctor() -> dict:
    """Inspect local prerequisites without downloading data or executing tasks."""
    from examples.appworld.prepare import LOCAL_RUNTIME
    from examples.common.wiki17_bm25 import DEFAULT_WIKI17_ROOT
    from examples.swebench_live.settings import ROOT as SWE_LIVE_ROOT
    from examples.taubench.benchmark_settings import DEFAULT_SOURCE

    packages = {}
    for name in ("litellm", "typesafe-sdk", "dspy", "decision-bench", "sentence-transformers", "harbor"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    paths = {
        "hotpotqa": DEFAULT_WIKI17_ROOT,
        "appworld": LOCAL_RUNTIME / "world",
        "obliqbench": Path(".cache/obliqbench/index"),
        "taubench": DEFAULT_SOURCE,
        "swebench_live": SWE_LIVE_ROOT,
    }
    docker = {"installed": shutil.which("docker") is not None, "running": False}
    if docker["installed"]:
        result = subprocess.run(
            ["docker", "info", "--format", "{{json .}}"], capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            info = json.loads(result.stdout)
            docker.update(running=True, architecture=info["Architecture"], memory_bytes=info["MemTotal"])
    return {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "free_disk_bytes": shutil.disk_usage(Path.cwd()).free,
        "packages": packages,
        "docker": docker,
        "data_paths": {name: {"path": str(path.resolve()), "exists": path.exists()} for name, path in paths.items()},
        "benchmarks": list(BENCHMARKS),
        "scope": "Presence check only. Each harness still verifies its pinned source, data and native runtime before model work.",
        "benchmark_runs_started": 0,
    }


def connection_check(
    profile: dict,
    key: str,
    output: Path,
    *,
    allow_calls: bool = False,
    roles: tuple[str, ...] = ("solver", "proposer", "tau", "jev"),
) -> dict:
    """Make at most one small request per service, retaining unknown outcomes without retry."""
    import litellm
    import typesafe_sdk

    from gepa.lm_constants import PROVIDER_RETRY_KEY
    from gepa.strategies.jev_constants import JEV_QUESTION_NAME
    from gepa.strategies.jev_controller import JevController

    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "state": "started",
        "benchmark_runs_started": 0,
        "checks": [],
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)

    def save() -> None:
        atomic_json(output, report)

    try:
        verify_routes(profile, tuple(role for role in roles if role != "jev"))
        get_json(f"{API_BASE}/key", key=key)
    except Exception as error:
        report.update(
            state="preflight_failed",
            error_type=type(error).__name__,
            error=str(error).replace(key, "[REDACTED]")[:1200],
        )
        save()
        return report
    report["authentication"] = "passed"
    if allow_calls:
        litellm.suppress_debug_info = True
        for role in roles:
            row = {"role": role, "state": "started", "api_requests": 1}
            report["checks"].append(row)
            save()
            try:
                if role == "jev":
                    controller = JevController(provider="openrouter")
                    with typesafe_sdk.TypeSafeClient(
                        api_key=key,
                        model=controller.run_contract()["model"],
                        base_url=profile["jev"]["api_base"],
                        timeout=30,
                        retry=typesafe_sdk.RetryPolicy(max_retries=0),
                    ) as client:
                        response = client.system_one(
                            model=profile["jev"]["model"],
                            state={"value": 2},
                            questions={
                                JEV_QUESTION_NAME: {
                                    "type": "choice",
                                    "instructions": "Is the value even?",
                                    "criteria": {"even": "Divisible by two", "odd": "Not divisible by two"},
                                }
                            },
                        ).model_dump(mode="json")
                    row.update(
                        model=response["model"],
                        provider=response.get("provider"),
                        usage=response.get("usage"),
                        response=response,
                    )
                    save()
                    controller._validate(
                        response, {"even", "odd"}, expected_model=controller.run_contract()["response_model"]
                    )
                else:
                    kwargs = tau_settings(profile)["kwargs"] if role == "tau" else role_kwargs(profile, role)
                    kwargs.pop(PROVIDER_RETRY_KEY, None)
                    kwargs.update(timeout=30, num_retries=0, max_retries=0, max_tokens=256)
                    response = litellm.completion(
                        model="openrouter/" + profile["roles"][role]["model_id"],
                        messages=[{"role": "user", "content": "Reply with only OK."}],
                        **kwargs,
                    )
                    row.update(
                        model=response.model,
                        provider=response.get("provider"),
                        usage=response.usage.model_dump(),
                        finish_reason=response.choices[0].finish_reason,
                        output_nonempty=bool(response.choices[0].message.content),
                    )
                    if not row["output_nonempty"] or row["finish_reason"] != "stop":
                        raise ValueError("Small connection probe did not produce a complete text response")
                    if row["provider"] != profile["roles"][role]["provider_name"]:
                        raise ValueError("Response provider did not match the frozen route")
                row["state"] = "passed"
            except Exception as error:
                row.update(
                    state="failed_or_unknown",
                    error_type=type(error).__name__,
                    error=str(error).replace(key, "[REDACTED]")[:1200],
                )
            save()
    report["state"] = "checked" if all(row["state"] == "passed" for row in report["checks"]) else "needs_review"
    report["scope"] = (
        "Authentication/short text/typed API compatibility only; no benchmark, optimizer, tool execution or quality validation."
    )
    save()
    return report


def main(argv: list[str] | None = None) -> int:
    """Separate metadata preparation, machine inspection and explicitly permitted checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Freeze current routes and prices; no inference calls")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--input-output-ratio", type=float, default=5.0)
    plan.add_argument("--solver-model", required=True, help="OpenRouter model ID; no model is assumed")
    plan.add_argument("--proposer-model", required=True, help="OpenRouter model ID; may equal solver")
    plan.add_argument("--solver-parameters", default='{"max_tokens":32768}', help="Explicit decoding JSON")
    plan.add_argument("--proposer-parameters", default='{"max_tokens":32768}', help="Explicit decoding JSON")
    plan.add_argument("--tau-model", help="Opt into a separate Tau variant with this simulator and judge model")
    plan.add_argument("--tau-parameters", help="Explicit auxiliary decoding JSON; requires --tau-model")
    commands.add_parser("doctor", help="Inspect local prerequisites only")
    check = commands.add_parser("check", help="Check auth/routes; optional four tiny API requests")
    check.add_argument("--profile", type=Path, required=True)
    check.add_argument("--env-file", type=Path)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--allow-check-calls", action="store_true")
    check.add_argument(
        "--roles", nargs="+", choices=("solver", "proposer", "tau", "jev"), default=["solver", "proposer", "tau", "jev"]
    )
    args = parser.parse_args(argv)
    if args.command == "plan":
        profile = make_profile(
            args.solver_model,
            args.proposer_model,
            solver_parameters=json.loads(args.solver_parameters),
            proposer_parameters=json.loads(args.proposer_parameters),
            tau_model=args.tau_model,
            tau_parameters=json.loads(args.tau_parameters) if args.tau_parameters is not None else None,
            input_output_ratio=args.input_output_ratio,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(profile, stream, indent=2)
        result = {
            "profile": str(args.output),
            "routes": {k: v["tag"] for k, v in profile["roles"].items()},
            "benchmark_runs_started": 0,
        }
    elif args.command == "doctor":
        result = doctor()
    else:
        result = connection_check(
            load_profile(args.profile),
            load_credential(args.env_file),
            args.output,
            allow_calls=args.allow_check_calls,
            roles=tuple(dict.fromkeys(args.roles)),
        )
    print(json.dumps(result, indent=2))
    return 1 if result.get("state") == "needs_review" else 0


if __name__ == "__main__":
    raise SystemExit(main())
