# Handover: ReAct V2 Reflection and Benchmark Harnesses

> **Superseded status note (August 22, 2026).** This file replaces the August 1
> handover for `rev1_action-conditioned_reflection`. That branch description,
> its selector/action tables, and its proposed Rev 3 work are historical and
> must not be used as the status of the current implementation.

## Current implementation

The primary non-vanilla workflow is now **Controller -> Manifestor -> ReAct
V2**, implemented in
`src/gepa/proposer/reflective_mutation/three_role.py`. The Controller uses
verbalized sampling to select an independently addressable document region and
edit action. At reflection level 2, the Manifestor converts the selected
semantic action into grounded, provider-routed steering; ReAct V2 alone applies
the edit.

The current contract is:

- system prompts and skills use explicit section/unit templates;
- ReAct V2 exposes literal `INSERT_TEXT`, `DELETE_TEXT`, `REPLACE_TEXT`, and
  `MOVE_TEXT` operations in the broad tool set, with insert/delete as the
  minimal atomic basis;
- semantic `rephrase`, `summarize`, and `expand` actions are directly coupled
  to an edit operation and may be decomposed into insert/delete calls for the
  atomic-basis ablation;
- accepted, rejected, and dropped attempts are retained as ordered
  user/assistant history on the candidate branch only; no global history is
  constructed;
- malformed documents, invalid tool calls, and context-budget overflow are
  explicit failures rather than silent rewrites or history compression; and
- the older free-form/action-suffix reflection path remains historical or an
  ablation, not the primary proposer implementation.

The public configuration and construction paths are documented in
`src/gepa/api.py` and `src/gepa/gepa_launcher.py`. The underlying implementation
is in:

| Concern | Source |
|---|---|
| Controller, reflection levels, and role orchestration | `src/gepa/proposer/reflective_mutation/three_role.py` |
| Manifestor and provider-role routing | `src/gepa/proposer/reflective_mutation/manifestor.py` |
| ReAct V2 protocol, edit tools, and local history | `src/gepa/proposer/reflective_mutation/react_v2_proposer.py` |
| Prompt/skill section templates | `src/gepa/strategies/document_template.py` |
| Edit operations and semantic coupling | `src/gepa/strategies/edit_tools.py`, `src/gepa/strategies/intervention.py` |

## Current benchmark harnesses

- [HotpotQA](../examples/hotpotqa/README.md) discards all dataset-supplied
  passages and performs the two-hop program through the English Wikipedia
  MediaWiki API. The committed sample is explicit smoke input only; production
  loading uses `hotpot_qa/fullwiki` and fails instead of silently falling back.
- [HoVer](../examples/hover/README.md) loads the official v1.1 data, performs
  the three-hop program through the same Wikipedia-only retrieval layer, and
  fails instead of silently substituting synthetic data.
- [Terminal-Bench](../examples/terminalbench/README.md) runs through an isolated
  Harbor `0.22.0` CLI boundary against the immutable
  `terminal-bench/terminal-bench@3.0.0` registry content hash. It optimizes only
  the Terminus instruction prompt initially, keeps tmux fixed, disables skills,
  permits long-horizon trajectories within Harbor timeouts, and retains
  verifier rewards, errors, and complete validated ATIF trajectories by task
  ID. Student and proposer model configuration are separate.

The benchmark entry points support matched vanilla and ReAct V2 configurations.
Run contracts pin the material model, data, budget, and ablation settings so a
resume cannot silently mix configurations.

## Experiment status

Experiment execution was deliberately excluded from this implementation pass.
No HotpotQA, HoVer, or Terminal-Bench comparison run has been launched, and
this handover makes no new performance, scaling, or H200-utilization claim.
Before launching, record the exact DeepSeek V4 Flash proposer identifier, Qwen
3.8 student checkpoint, H200 allocation, matched depths, and matched metric-call
budgets. Historical IFBench result files remain historical artifacts; they do
not establish results for the current ReAct V2 design or these harnesses.
