# Della benchmark preparation

Benchmark evaluation and optimization use the six entrypoints documented in the
[shared suite guide](../../examples/BENCHMARKS.md). The scripts here prepare
remote environments, pinned model checkpoints, and HotPotQA's frozen retrieval
index, and verify DeepSeek serving.

Copy `.env.example` to `.env`, fill in the host and storage settings, and restrict
it to mode 600. SSH uses verified host keys and the user's existing configuration.
Keep secrets and the local `.env` out of Git.

| Script | Purpose |
|---|---|
| `della_session.sh` | Open, inspect, or close shared SSH sessions |
| `preflight_hotpotqa.sh` | Check source, SSH, CUDA, storage, and serving locks |
| `sync_to_della.sh` | Sync the checkout while preserving remote caches and outputs |
| `build_env.sh` | Prepare environments, Wiki-2017 retrieval, and pinned checkpoints |
| `lock_serving_env.sh` | Regenerate serving dependency locks from their source requirements |
| `submit_deepseek_smoke.sh` | Submit and retrieve the independent serving diagnostic |
| `verify_deepseek_serving.sh` | Exercise the pinned DeepSeek server and tool protocol on allocated GPUs |
| `remote/setup_terminalbench.sh` | Install isolated Harbor and expose the cluster Apptainer runtime on the visualization host |
| `remote/with_private_network.sh` | Run an allocated single-node job with shared private loopback for Harbor, task containers, and model servers |

Terminal-Bench also needs [offline task preparation](../../examples/terminalbench/README.md)
on the visualization host. The shared runner verifies the sealed task packages
and prepared SIF images before running on compute nodes. CLI installation and an
empty container cache alone are insufficient.

`build_env.sh` starts detached downloads; its exit is not proof that they finished.
Wait for `MODELS_DONE` in the printed log and verify the checkpoint manifests.
`remote/setup_env.sh` records installed environment manifests, and
`remote/download_model.sh` verifies existing shared checkpoints without silently
replacing their bytes. Large downloads run on the configured visualization host.

The Qwen and DeepSeek serving environments use separate locks under
`examples/hotpotqa/serving/`. Request defaults, bounded retries, and solver/proposer
roles come from `examples/common/experiment_models.py` and `model_settings.py`.
`runtime_constants.sh` is generated from `examples/common/launcher_constants.py`.

After preparing the model servers, use the shared runner's `--mode pilot` on
training examples, then `--condition both` for vanilla GEPA and FOREST. The
runner handles validation selection, frozen winners, held-out repetitions,
baseline reuse, resume identity, and timing. Job allocation is separate from
benchmark execution; these tools do not automatically submit optimization
campaigns or continue expired allocations.

## Private loopback for Terminal-Bench

Inside an allocated single-node job, wrap the command that starts both model
servers and the benchmark together:

```bash
bash scripts/della/remote/with_private_network.sh bash /path/to/allocated-job-body.sh
```

The launcher uses Linux's [root-mapped user and network namespaces](https://man7.org/linux/man-pages/man1/unshare.1.html)
and brings up only loopback. Harbor's parent and Apptainer children retain their
shared `127.0.0.1` control channel, while task services can use their original
low ports inside the private network. Apptainer 1.5 explicitly supports an
[existing root-mapped namespace](https://github.com/apptainer/apptainer/blob/v1.5.0/internal/pkg/runtime/launch/launcher_linux.go#L101-L111).
No host sysctl, task port, vendor source, external route, or forwarding rule is
changed. The launcher fails if namespace creation or loopback setup is denied.

All images, task packages, model snapshots, dependencies, and verifier resources
must already be staged. The job cannot reach external services, model servers
started outside the wrapper, or other nodes. It sets `VLLM_HOST_IP=127.0.0.1`,
`NCCL_SOCKET_IFNAME="=lo"`, and `GLOO_SOCKET_IFNAME=lo`; keep API endpoints on
loopback too. Run trials sequentially because their fixed task ports share this
job namespace. GPU access and multi-GPU collectives still need verification on
the actual allocation; the launcher does not change Slurm device permissions.

## Jev evidence pooling

Jev controller policy `jev_joint_action_section_v5_evidence_pool` addresses the
provider's 32k state-plus-question limit for large single-component reflection
minibatches. It sends contiguous groups of at most three complete training
records with the full component and choice constraints in every request. Each
group's validated, normalized probability distribution is weighted by its record
count; their arithmetic mean receives one exploration mixture and one sampled
choice. This preserves every supplied record, including repeated text and error
records, but changes the controller's statistical policy. It requires a fresh
source-bound campaign and cannot continue a v4 checkpoint. Solver minibatch
size, candidate admission, Manifestor evidence, and evaluation budgets do not
change. Inputs of three or fewer records and joint-component selection retain
their single-request behavior. An individual oversized group still fails closed;
there is no truncation, model fallback, or adaptive retry with less evidence.

Every group has its own response journal entry, provider request identity,
attempt ledger, and native retry allowance. A completed group is reused after
interruption. A direct request started without a committed response requires
review before another call; offline requests retain the sealed mailbox protocol.
Pooled metadata contains the individual raw responses and total usage, and labels
the resulting probabilities as an aggregate rather than a single Jev response.
It does not invent a joint provider confidence or count aggregate usage twice.
