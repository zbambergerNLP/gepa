# Attribution and provenance

This integration is original GEPA adapter code. It invokes, but does not vendor, the following projects:

- [Harbor](https://github.com/harbor-framework/harbor), version `0.22.0`, Apache-2.0. The fixed JSON command contract in `terminal_bench_adapter.py` is derived from `src/harbor/agents/terminus_2/templates/terminus-json-plain.txt` at Harbor tag `v0.22.0`.
- [Terminal-Bench / Frontier-Bench](https://github.com/harbor-framework/terminal-bench), registry dataset `terminal-bench/terminal-bench@3.0.0`, source tag `v3.0.0` (`2b0442c3c583b710ca8da14c8e601b99f2f1f244`). Individual task authors and licenses are recorded in the upstream task metadata.
- [Agent Trajectory Interchange Format](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md), emitted by Terminus 2 and retained verbatim by this adapter.

The manifest registry content hash is `sha256:a32a61879ea94eb9dc16fa1fbeb398759f0c07ca633d9d1f6aec760207036da3`. Task refs were resolved from the official Harbor registry with Harbor 0.22.0 on 2026-08-22. No benchmark task contents, oracle solutions, proxy metrics, or synthetic substitutes are committed here.
