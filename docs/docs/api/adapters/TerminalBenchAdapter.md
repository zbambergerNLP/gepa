# TerminalBenchAdapter

This fork's `TerminusAdapter` is a Harbor port of GEPA's published adapter for
Terminal-Bench 2.1. It supports two editable scopes: one unified initial
instruction prompt (the default), or all 14 prompts and two skills. Both start with identical
model input and skill files; the smaller scope fixes later prompts and skills.
The upstream legacy
`tb run` implementation and constructor are not used unchanged. The
[published upstream source](https://github.com/gepa-ai/gepa/blob/4f1613773d0c13c8f1551543a801b299bd8acf73/src/gepa/adapters/terminal_bench_adapter/terminal_bench_adapter.py)
is pinned in `TERMINUS_ADAPTER_CONTRACT` alongside the local port identity.

::: gepa.adapters.terminal_bench_adapter.terminal_bench_adapter.TerminusAdapter
    handler: python
    options:
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
        members_order: source
        show_signature_annotations: true
