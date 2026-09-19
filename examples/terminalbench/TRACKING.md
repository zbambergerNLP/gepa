# Offline W&B reporting

Terminal-Bench optimization and held-out evaluation accept `--wandb-project`,
`--wandb-entity`, and `--wandb-group`. Pass the same flags to
`examples.terminalbench.run_ablations` to apply them to every method, text scope,
and budget. The corresponding `TERMINALBENCH_WANDB_PROJECT`,
`TERMINALBENCH_WANDB_ENTITY`, and `TERMINALBENCH_WANDB_GROUP` environment variables
are also supported. With no project, tracking is disabled.

For example, add these flags to an otherwise qualified campaign command:

```bash
--wandb-project forest-terminalbench --wandb-entity gilad-mo12 --wandb-group CAMPAIGN_ID
```

The logger always calls `wandb.init(mode="offline")`, including when an inherited
`WANDB_MODE` says `online`. It does not call `wandb.login` or synchronize from a
GPU node. Install the repository's `full` extra before creating the offline
environment; no compute-node API key is needed.

Each optimization allocation gets its own run ID. Its config contains the exact
run contract, logical cell digest, and allocation ID. It records validation
Pass@1 against actual metric calls, prompt sizes, candidate text, proposal
acceptance, and reported physical token usage by model and role. Missing token
counts remain explicitly unreported. The usage summary is cumulative over the
saved optimization and Harbor roots, including previous allocations; do not sum
these cumulative summaries across resumed segments. Optimization completion alone never sets
`heldout_complete` or `completed_ablation`. Reporting failures are saved to
`RUN_DIR/tracking/errors.jsonl` without changing evaluation or selection.

After held-out evaluation succeeds, the evaluator creates a separate result run.
It validates the completed optimization checkpoint, frozen winner and baseline,
three distinct complete test repetitions for each, and the summary against the
per-task scores. This report includes mean Pass@1, sample standard deviation,
baseline gain, exact runtime/data contracts, and checksum-listed JSON evidence.
It describes one completed ablation, not completion of the entire campaign.
Re-reporting identical evidence to the same destination reuses its local marker;
subsequent additions to the shared comparison do not duplicate earlier reports.

## Upload after an ablation

1. Fetch the completed cell directory and its shared held-out directory to the
   authenticated laptop. Preserve the entire `tracking/` directory, including
   `.wandb` files, `files/media/`, and `files/evidence/`. Keep the full Harbor logs
   in the experiment archive; W&B is a report, not a replacement for that archive.
2. Compare remote/local SHA-256 manifests for the downloaded artifacts. Review
   `tracking/errors.jsonl`; a successful benchmark does not prove reporting succeeded.
3. Synchronize each completed offline directory identified by `tracking/*.json`:

   ```bash
   uv run wandb sync --entity gilad-mo12 --project forest-terminalbench \
     /absolute/local/cell/tracking/wandb/offline-run-TIMESTAMP-RUN_ID
   ```

4. Inspect the cloud run and confirm its contract, metrics, completion flags and
   evidence files. Retain local markers and archives. Do not use `--sync-all`
   across unrelated benchmarks, and do not mark an upload complete merely because
   an offline log exists.

An older completed archive without tracking can be reported without new model or
Harbor calls:

```bash
uv run python -m examples.terminalbench.tracking \
  --run-dir /absolute/local/cell --heldout-dir /absolute/local/shared-test \
  --cell system_prompt__react_v2 --project forest-terminalbench \
  --entity gilad-mo12 --group CAMPAIGN_ID
```

Use the actual cell label, source checkout, and verified runtime contracts.
Incomplete optimizations or test records are rejected. These reporting options
do not select new budgets, concurrency, models, or benchmark tasks, and setting
them does not submit a job.
