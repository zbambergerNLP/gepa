"""Keep minibatch ablations consistent across sampling, contracts and launchers."""

import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_wikipedia_react_v2_config import _hotpot_args

from examples.common.experiment_models import DEEPSEEK_V4_1_FLASH_MODEL
from examples.hotpotqa.main import _run_key, build_config, build_parser, build_run_contract
from gepa.core.data_loader import ListDataLoader
from gepa.strategies.batch_sampler import IndependentEpochShuffledBatchSampler

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(("condition", "controller"), [("vanilla", "verbalized"), ("react_v2", "jev")])
@pytest.mark.parametrize("size", [3, 8, 16, 32])
def test_minibatch_reaches_sampler_and_resume_identity(condition, controller, size, tmp_path):
    """Preserve matching batches across methods and distinguish resume identities."""
    args = _hotpot_args(
        condition=condition,
        controller_selection=controller,
        reflection_model=DEEPSEEK_V4_1_FLASH_MODEL,
        reflection_minibatch_size=size,
    )
    config, _ = build_config(condition, args, {}, str(tmp_path))
    contract = build_run_contract(condition, args)
    sampler = config.reflection.batch_sampler
    assert config.reflection.reflection_minibatch_size == size
    assert sampler.contract() == contract["optimizer"]["training_batch_order"]
    assert contract["optimizer"]["reflection_minibatch_size"] == size
    assert config.reflection.module_selector == contract["optimizer"]["component_selector"] == "round_robin"
    reference = IndependentEpochShuffledBatchSampler(size, args.seed)
    loader = ListDataLoader(list(range(150)))
    for iteration in range(25):
        batch = sampler.next_minibatch_ids(loader, SimpleNamespace(i=iteration))
        assert len(batch) == size
        assert batch == reference.next_minibatch_ids(loader, SimpleNamespace(i=iteration))
    restored = IndependentEpochShuffledBatchSampler(size, args.seed)
    restored.set_state(sampler.get_state())
    assert restored.next_minibatch_ids(loader, SimpleNamespace(i=25)) == sampler.next_minibatch_ids(
        loader, SimpleNamespace(i=25)
    )
    other = _hotpot_args(condition=condition, reflection_minibatch_size=size + 1)
    assert _run_key(condition, args) != _run_key(condition, other)


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_minibatch_fails_before_configuration(size, tmp_path):
    """Reject invalid sizes before sampling can fail or loop on an empty batch."""
    args = _hotpot_args(reflection_minibatch_size=size)
    for build in (
        lambda: build_config("vanilla", args, {}, str(tmp_path)),
        lambda: build_run_contract("vanilla", args),
    ):
        with pytest.raises(ValueError, match="positive integer"):
            build()


def test_default_three_keeps_existing_contract_and_cli():
    """Leave existing three-example runs and their checkpoint keys unchanged."""
    assert build_parser().parse_args([]).reflection_minibatch_size == 3
    assert build_parser().parse_args(["--reflection-minibatch-size", "16"]).reflection_minibatch_size == 16
    assert build_run_contract("vanilla", _hotpot_args()) == build_run_contract(
        "vanilla", _hotpot_args(reflection_minibatch_size=3)
    )


@pytest.mark.parametrize("size", ["8", "16", "32"])
def test_workload_forwards_minibatch_to_experiment(size):
    """Execute the launch command with a recording client to check the actual CLI."""
    source = (ROOT / "scripts/della/remote/hotpotqa_workload.sh").read_text()
    block = source[source.index('"${PY}" -m examples.hotpotqa.main') :]
    env = {
        **os.environ,
        **dict.fromkeys(re.findall(r"\$\{([A-Z][A-Z_0-9]*)", block), "fixture"),
        "PY": "record_args",
        "HOTPOTQA_REFLECTION_MINIBATCH_SIZE": size,
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            'record_args() { printf "%s\\n" "$@"; }\n'
            "SOLVER_API_ARG=()\nREFLECTION_API_ARG=()\nTRACKING_ARGS=()\n" + block,
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    assert arguments[arguments.index("--reflection-minibatch-size") + 1] == size


def test_minibatch_locks_allow_matched_campaign_concurrency():
    """Keep the legacy lock and isolate new sizes while sharing a baseline campaign."""
    source = (ROOT / "examples/hotpotqa/run_hotpotqa.sbatch").read_text()
    start = source.index('MINIBATCH_LOCK_SUFFIX=""')
    block = source[start : source.index("exec {RUN_LOCK_FD}", start)]
    paths = []
    for size in (3, 8, 16, 32):
        result = subprocess.run(
            ["bash", "-c", "set -eu\n" + block + '\nprintf "%s" "$RUN_LOCK_PATH"'],
            env={
                **os.environ,
                "RUN_LOCK_DIR": "/fixture",
                "MODEL_PROFILE": "qwen",
                "BUDGET_PROFILE": "standard",
                "CONDITION": "vanilla",
                "HOTPOTQA_PILOT_ONLY": "0",
                "HOTPOTQA_REFLECTION_MINIBATCH_SIZE": str(size),
            },
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
        paths.append(result.stdout)
    assert paths[0] == "/fixture/qwen-standard-vanilla-pilot0.lock"
    assert len(set(paths)) == 4
