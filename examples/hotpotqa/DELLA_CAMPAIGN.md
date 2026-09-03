<!-- Source: Gilad Morad, https://gist.github.com/gilad12-coder/b0142ad28de98c47487ea9847686206a (fetched 2026-09-02).
     Adapted in this checkout: the Qwen arm no longer depends on an external POSIT checkout; this repo builds its own
     hash-locked vLLM serving venv (section 4). Everything else follows the gist. -->

# Running the HotPotQA campaign on Della

Zach,

The code is in [PR #59](https://github.com/zbambergerNLP/gepa/pull/59), and [Graphite](https://app.graphite.dev/github/pr/zbambergerNLP/gepa/59) has the review view. Use this commit for the first campaign:

```text
169ddda125b1abe305c7714bbb5b3fc38b21b587
```

We are running six configurations with Qwen3.8-27B and six with GLM-5.3-Flash. Each run uses one model for the student, proposer, and Controller.

## 1. Run the launcher locally

Run the launcher from your laptop or another machine that can SSH non-interactively to both Della hosts. Do not invoke `examples/hotpotqa/run_hotpotqa.sbatch` directly.

The launcher uses each machine as follows:

- Your local machine freezes and syncs the source commit, submits jobs, and fetches results.
- `della-vis1` downloads and verifies the internet-dependent artifacts.
- The Della login node calls `sbatch`.
- The allocated eight-H200 Slurm nodes serve the local models and run GEPA.

Model inference and GEPA both run on Della. The local script handles submission and file transfer.

Local prerequisites:

```bash
command -v git
command -v ssh
command -v rsync
command -v sha256sum
command -v uv
```

Your SSH agent or `~/.ssh/config` must already provide access. Visit both hosts once so their verified host keys are present in `~/.ssh/known_hosts`:

```bash
ssh YOUR_NETID@della.princeton.edu exit
ssh YOUR_NETID@della-vis1.princeton.edu exit
```

The scripts use `BatchMode=yes` and `StrictHostKeyChecking=yes`. A launch will fail if either host needs a password prompt or has an unknown host key.

## 2. Check out the experiment source

For a fresh checkout:

```bash
git clone https://github.com/zbambergerNLP/gepa.git gepa-hotpotqa
cd gepa-hotpotqa

git fetch origin codex/review-current-07-hotpotqa-slurm
git switch --detach 169ddda125b1abe305c7714bbb5b3fc38b21b587
```

For an existing checkout, check for local work before switching commits:

```bash
cd /path/to/gepa
git status --short
git fetch origin codex/review-current-07-hotpotqa-slurm
git switch --detach 169ddda125b1abe305c7714bbb5b3fc38b21b587
```

Check the commit and worktree:

```bash
test "$(git rev-parse HEAD)" = "169ddda125b1abe305c7714bbb5b3fc38b21b587"
test -z "$(git status --porcelain --untracked-files=normal)"
echo "Source is exact and clean."
```

The launcher rejects a dirty checkout.

## 3. Configure the Della paths

Create the ignored local configuration file:

```bash
cp scripts/della/.env.example scripts/della/.env
chmod 600 scripts/della/.env
```

Edit `scripts/della/.env` and replace every placeholder:

```bash
REMOTE_USER="YOUR_NETID"
REMOTE_HOST="della.princeton.edu"
REMOTE_VIS_HOST="della-vis1.princeton.edu"
REMOTE_DIR="/scratch/gpfs/YOUR_ALLOCATION/YOUR_NETID/gepa"
SCRATCH_BASE="/scratch/gpfs/YOUR_ALLOCATION/YOUR_NETID/gepa"
MODEL_STORAGE="/projects/YOUR_ALLOCATION/model_storage"
GPU_PARTITION="ailab"
```

For the existing BSTEWART shared model location, use:

```bash
MODEL_STORAGE="/projects/BSTEWART/model_storage"
```

`MODEL_STORAGE` must be writable while the artifacts are first prepared and readable from the visualization, login, and compute nodes.

Recheck the file mode:

```bash
test "$(stat -f '%Lp' scripts/della/.env 2>/dev/null || stat -c '%a' scripts/della/.env)" = "600"
```

## 4. Check the serving prerequisites

The Qwen arm is served by a vLLM environment that this repository builds itself from
`examples/hotpotqa/serving/requirements.in` and its hash-locked resolution
`examples/hotpotqa/serving/requirements-x86_64-linux-py312.txt` (regenerate with
`scripts/della/lock_serving_env.sh` after changing the `.in` file, and commit the lock).
No other project's checkout or virtual environment is used.

Run the read-only preflight from your laptop; it covers steps 1-4:

```bash
scripts/della/preflight_hotpotqa.sh
```

It checks the local tools, the exact commit and clean tree, `scripts/della/.env`,
non-interactive SSH to both hosts, and on `della-vis1`: Apptainer, the `cudatoolkit/13.0`
module, a writable `MODEL_STORAGE`, the home quota, and (once built) that the serving venv
matches the committed lock. The pinned vLLM is 0.25.1 on torch 2.11 / CUDA 13.0, which
satisfies the 0.17.0 floor with the data-parallel, multi-API-server, and native tool-call
options.

Della requires an SSH key **and** a password step; the launcher scripts use `BatchMode=yes`,
so open the persistent master connections once per laptop session:

```bash
scripts/della/della_session.sh open
```

## 5. Build the shared artifacts

Run this once from the repository on your laptop:

```bash
scripts/della/build_env.sh
```

`build_env.sh` runs the downloads on `della-vis1`. It also:

- builds the frozen Python 3.11.13 and uv 0.9.13 GEPA environment;
- builds the hash-locked vLLM serving venv at `$REMOTE_DIR/.serving-venv` and freezes its manifest;
- builds and verifies the frozen Wiki-2017 BM25 index;
- caches the exact HotPotQA 150/300/300 split;
- downloads and byte-verifies both pinned model checkpoints; and
- builds and verifies the pinned GLM SGLang Apptainer image.

The shared files will be at:

```text
$MODEL_STORAGE/Qwen3.8-27B
$MODEL_STORAGE/GLM-5.3-Flash
$MODEL_STORAGE/runtimes/sglang-glm-5.3-flash-x86_64.sif
```

Model and runtime revisions:

```text
Qwen/Qwen3.8-27B
revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

zai-org/GLM-5.3-Flash
revision 04c4e9e95c5da8862dced7e5056455116f83a7e0

docker://lmsysorg/sglang@sha256:0836f0160fa785e424e68d13ef88ddd548f87e6e11ad9f0e4de982e4f9188aaf
```

Use `build_env.sh` rather than `huggingface-cli download`. The launcher requires the `.gepa-model-integrity.json` manifests generated during this build. Existing valid files are reused and verified.

After the build succeeds, check that the three files are present:

```bash
source scripts/della/.env

ssh "${REMOTE_USER}@${REMOTE_VIS_HOST}" \
  "test -s '${MODEL_STORAGE}/Qwen3.8-27B/.gepa-model-integrity.json' &&
   test -s '${MODEL_STORAGE}/GLM-5.3-Flash/.gepa-model-integrity.json' &&
   test -s '${MODEL_STORAGE}/runtimes/sglang-glm-5.3-flash-x86_64.sif' &&
   echo 'All pinned model/runtime artifacts are present.'"
```

The byte-level verification runs inside `build_env.sh`.

## 6. Experiment matrix

The launcher enforces this order separately for Qwen and GLM:

| Order | Tree | Code condition | Metric-call budget |
|---:|---|---|---:|
| 1 | Standard | `vanilla` | 6,871 |
| 2 | Standard | `react_v2` | 6,871 |
| 3 | Standard | `react_v2_random` | 6,871 |
| 4 | Standard | `action` | 6,871 |
| 5 | Expanded | `vanilla` | 13,742 |
| 6 | Expanded | `react_v2` | 13,742 |

Each job uses `afterok` on the previous job. Within a model arm, the four standard-tree runs finish before either expanded-tree run starts. The Qwen and GLM arms are independent and can occupy two nodes at once.

The GLM submission starts with a four-hour, 20-attempt native multi-tool canary. Its experiment jobs depend on that canary. Each Qwen job runs a native tool-call check before optimization.

## 7. Submit the jobs

Launch the Qwen chain:

```bash
MODEL_PROFILE=qwen3.8-27b \
BUDGET_PROFILE=campaign \
CONDITION=all \
HOTPOTQA_CAMPAIGN_ID=hotpotqa-final-v1 \
scripts/della/submit_hotpotqa.sh
```

Launch the GLM chain:

```bash
MODEL_PROFILE=glm-5.3-flash \
BUDGET_PROFILE=campaign \
CONDITION=all \
HOTPOTQA_CAMPAIGN_ID=hotpotqa-final-v1 \
scripts/della/submit_hotpotqa.sh
```

The Qwen command submits six jobs. The GLM command submits a canary followed by six jobs. The campaign produces 12 result cells.

The launcher stages the source at:

```text
$REMOTE_DIR/sources/169ddda125b1abe305c7714bbb5b3fc38b21b587
```

The jobs load the checkpoints from Della storage and set Hugging Face and Transformers to offline mode. They do not use OpenRouter or another hosted model API.

The Slurm time limits are safety caps, not runtime estimates: Qwen standard jobs receive 72 hours, Qwen expanded jobs 144 hours, GLM campaign jobs 144 hours, and the GLM canary 4 hours.

## 8. Monitor the jobs

From a Della shell:

```bash
squeue -u "$USER" -o "%.18i %.60j %.2t %.12M %.30R"
```

Qwen job order:

```text
gepa-hp-qwen3.8-27b-standard-vanilla
gepa-hp-qwen3.8-27b-standard-react_v2
gepa-hp-qwen3.8-27b-standard-react_v2_random
gepa-hp-qwen3.8-27b-standard-action
gepa-hp-qwen3.8-27b-expanded-vanilla
gepa-hp-qwen3.8-27b-expanded-react_v2
```

GLM starts with:

```text
gepa-hp-glm-canary
```

The six GLM experiment jobs use the same standard-first order with the `glm-5.3-flash` profile.

Inspect the dependency and state of one job:

```bash
scontrol show job JOB_ID | tr ' ' '\n' | grep -E '^(JobName|JobState|Reason|Dependency)='
```

After jobs leave the queue, inspect accounting records. Replace the date with the actual submission date:

```bash
sacct -u "$USER" --starttime YYYY-MM-DD \
  --format=JobIDRaw,JobName%64,State,ExitCode,Elapsed,Timelimit
```

Logs are written under:

```text
$SCRATCH_BASE/logs/hotpotqa/hotpotqa-final-v1/169ddda125b1abe305c7714bbb5b3fc38b21b587/
```

List them:

```bash
source scripts/della/.env

find "${SCRATCH_BASE}/logs/hotpotqa/hotpotqa-final-v1/169ddda125b1abe305c7714bbb5b3fc38b21b587" \
  -maxdepth 1 -type f -print | sort
```

Each Slurm job has a `hotpotqa-<job-name>-<job-id>.log`; the corresponding local model server has `gen-<job-id>.log`.

## 9. Resume a failed or preempted run

Keep the existing output. Resubmit the same source commit, campaign ID, model, budget profile, and condition. GEPA will reuse its saved state, response journal, evaluation cache, and held-out checkpoints.

Example: resume Qwen standard ReAct V2:

```bash
MODEL_PROFILE=qwen3.8-27b \
BUDGET_PROFILE=standard \
CONDITION=react_v2 \
HOTPOTQA_CAMPAIGN_ID=hotpotqa-final-v1 \
scripts/della/submit_hotpotqa.sh
```

Example: resume GLM expanded ReAct V2:

```bash
MODEL_PROFILE=glm-5.3-flash \
BUDGET_PROFILE=expanded \
CONDITION=react_v2 \
HOTPOTQA_CAMPAIGN_ID=hotpotqa-final-v1 \
scripts/della/submit_hotpotqa.sh
```

A targeted GLM resubmission adds a new canary before the run. When an `afterok` parent fails, its old dependent jobs cannot run. Cancel those job IDs and submit the missing runs explicitly. Do not submit the `.sbatch` file directly.

## 10. Fetch the results

Once all result cells have completed, run this from the same local checkout:

```bash
HOTPOTQA_SOURCE_COMMIT=169ddda125b1abe305c7714bbb5b3fc38b21b587 \
HOTPOTQA_CAMPAIGN_ID=hotpotqa-final-v1 \
scripts/della/fetch_hotpotqa_results.sh
```

The local result directory is:

```text
outputs/hotpotqa-campaigns/hotpotqa-final-v1/169ddda125b1abe305c7714bbb5b3fc38b21b587/
```

It contains:

```text
runs/
logs/
campaign-locks/
hotpotqa_analysis.json
```

The command prints validation EM, test EM, test F1, call and candidate counts, candidate and proposal Jaccard diversity, action usage and acceptance, action entropy, Controller entropy, fallback rate, and tail-sampling rate.

The analyzer accepts a completed subset. Check that all campaign cells are present before comparing results:

```bash
python3 - <<'PY'
import json
from pathlib import Path

commit = "169ddda125b1abe305c7714bbb5b3fc38b21b587"
analysis = (
    Path("outputs/hotpotqa-campaigns/hotpotqa-final-v1")
    / commit
    / "hotpotqa_analysis.json"
)
runs = json.loads(analysis.read_text())["runs"]

models = {
    "hosted_vllm/Qwen/Qwen3.8-27B",
    "hosted_vllm/zai-org/GLM-5.3-Flash",
}
cells = {
    (6871, "vanilla"),
    (6871, "react_v2"),
    (6871, "react_v2_random"),
    (6871, "action"),
    (13742, "vanilla"),
    (13742, "react_v2"),
}
expected = {
    (model, budget, condition)
    for model in models
    for budget, condition in cells
}
actual = {
    (run["model"], run["max_metric_calls"], run["condition"])
    for run in runs
}

missing = sorted(expected - actual)
extra = sorted(actual - expected)
if missing or extra:
    raise SystemExit(
        f"Campaign is incomplete. Missing={missing}; extra={extra}"
    )
print("All 12 HotPotQA campaign cells completed.")
PY
```

Print a compact table:

```bash
jq -r '
  .runs[]
  | [.model_label, .budget_profile, .condition,
     .best_validation_exact_match, .test_exact_match, .test_f1]
  | @tsv
' \
outputs/hotpotqa-campaigns/hotpotqa-final-v1/169ddda125b1abe305c7714bbb5b3fc38b21b587/hotpotqa_analysis.json
```

Paired confidence intervals are not implemented. Ignore the absolute `path` fields in the JSON because they point to temporary fetch directories. Use the result directory shown above. The reported scores and mechanism statistics are valid.

## 11. Final check

Confirm these items before using the results:

- The source commit is exactly `169ddda125b1abe305c7714bbb5b3fc38b21b587`.
- `build_env.sh` completed without an integrity error.
- The Qwen chain produced six successful Slurm jobs.
- The GLM canary passed and its chain produced six successful Slurm jobs.
- No relevant job ended in `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, or `CANCELLED`.
- The completeness script prints `All 12 HotPotQA campaign cells completed.`
- `hotpotqa_analysis.json`, run artifacts, campaign locks, and logs were fetched locally.

If a preflight or integrity check fails, keep the error and logs and stop. The failed run should not be included in the comparison.
