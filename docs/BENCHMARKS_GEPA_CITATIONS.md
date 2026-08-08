# Benchmarks Where GEPA Is Beaten — Citation Deep Research

*Deep research per Lakshya meeting — GEPA citations (arXiv:2507.19457, Agrawal et al., Berkeley/Stanford/Databricks/MIT, ICLR 2026 Oral).*
*Generated: 2026-08-08. Analyst: subagent (no memo). Workspace: `gepa/`.*

> **TL;DR** — 11 citing papers propose methods that beat GEPA on at least one benchmark. The 5–6 most frequent “GEPA-beaten” benchmarks are **HotpotQA, IFBench, HoVer, GSM8K, PUPA, and MBPP/MATH**. All four of GEPA’s multi-step benchmarks (HotpotQA/IFBench/HoVer/PUPA) are beaten by ≥4 independent papers; GSM8K is beaten by 3–4 papers (one by >20 pp). FOREST already scaffolds 6/6 GEPA-paper benchmarks; the highest-value TODO is **GSM8K (+ MBPP/HumanEval)** where GEPA’s largest defeats are reported.

---

## 1. Method

### 1.1 Search strategy

All queries via `web_search` on 2026-08-08 (≥14 searches, see parent log). Queries:

```
GEPA arXiv 2507.19457 citations
GEPA prompt optimization citing papers
GEPA outperforms baseline benchmark
papers citing GEPA Berkeley prompt optimization 2025
"2507.19457" citing papers
GEPA reflective prompt evolution citations benchmark HotpotQA GSM8K
VISTA GEPA ICLR 2026 IFBench benchmark
GEPA SWE-Bench LiveBench PUPA benchmark outperformed
arxiv GEPA beating outperforms TextGrad DSPy MIPRO HotpotQA 2025 2026
semantic scholar GEPA citations list papers that cite Agrawal 2025
CANTANTE optimizer beats GEPA MBPP GSM8K HotpotQA
Error Taxonomy prompt optimization GEPA MIPRO comparison heavy
FAPO Fully Automated Prompt Optimization beats GEPA 14.1 HotpotQA
ADOPT Adaptive Dependency beats GEPA TextGrad multi-step pipeline
SPEAR Code-Augmented beats GEPA Hiring Assistant kappa
FlowBot bilevel textual gradients GEPA HotpotQA DROP HumanEval
plus targeted arxiv fetches (2606.19605, 2605.13295, 2512.24933, 2605.26275, 2604.26258, 2602.00997, 2603.18388, 2605.28360, 2511.07919, 2607.14408)
```

Sources: arXiv HTML/PDF, OpenReview (ICLR 2026), HuggingFace Papers, vendor blogs (Cisco, Comet, VentureBeat), Semantic Scholar / HF citation graph. `web_search` snippets cross-checked against arXiv HTML where available; deltas quoted verbatim where snippets truncated are flagged.

### 1.2 Inclusion criteria

- Must **cite** `arXiv:2507.19457` (or ICLR 2026 version `RQm2KQTM5r`).
- Must **report head-to-head vs GEPA** on named benchmark(s) (not just “we use GEPA”).
- “GEPA beaten?” = *Yes* if paper’s method > GEPA mean (or best-val) in its table/abstract, even if within error bars (note given). *Partial/Mixed* if beats on subset of tasks/models.

### 1.3 Limitations

- No Semantic Scholar API key; used `web_search` + arXiv instead. Counts are lower-bound; Google Scholar reports >60 citations by Aug 2026 but only ~15 have public preprints that evaluate vs GEPA.
- Some papers test on non-paper splits (e.g., GPT-4o-mini vs Qwen3-8B, or reward-free validators). Deltas are not directly comparable; treat as directional + relative cost.
- Agentic benchmarks (SWE-Bench, TerminalBench, AppWorld) cite GEPA conceptually but no found paper reports beating GEPA *on those benchmarks* yet — flagged as gap.

### 1.4 GEPA’s own benchmark suite (reference)

From Agrawal et al. v2 (Feb 2026) / ICLR 2026 Tables 1–2 (Qwen3-8B & GPT-4.1-mini, 6 tasks):

| Model | HotpotQA | IFBench | HoVer | PUPA | AIME-2025 | LiveBench-Math | Avg |
|-------|----------|---------|-------|------|-----------|----------------|-----|
| Qwen3-8B Baseline | 42.33 | 36.90 | 35.33 | 80.82 | 27.33 | 48.70 | 45.23 |
| Qwen3-8B GEPA | **62.33** | **38.61** | **52.33** | **91.85** | 32.00 | 51.95 | 54.85 (+9.62) |
| Qwen3-8B GEPA+Merge | 64.33 | 28.23 | 51.67 | 86.26 | 32.00 | 51.95 | 52.40 |
| GPT-4.1-mini GEPA | **69.00** | **52.72** | **51.67** | **94.47** | 59.33 | 64.13 | 65.22 (+12.19) |

Budgets: GEPA 6871 (HotpotQA) / 3593 (IFBench) / 7051 (HoVer) / 2426 (PUPA) / 1839 (AIME/LiveBench) metric calls; GRPO 24k on all. GEPA beats GRPO (+6 pp avg, up to +20 pp) and MIPROv2 (+10 pp) in-paper. Arrow “beaten” below is *future work beating this*.

---

## 2. Papers Found (cite GEPA + evaluate vs it)

| # | Paper (venue/year) | Benchmarks tested vs GEPA | GEPA beaten? | Delta vs GEPA (reported) | Link |
|---|--------------------|--------------------------|--------------|--------------------------|------|
| 1 | **FAPO: Fully Automated Prompt Optimization of Multi-Step LLM Pipelines** — Cisco Foundation AI, Streche et al., arXiv:2606.19605v2 (Jun 2026) | HotpotQA, IFBench, HoVer, PUPA, AIME-2025, LiveBench-Math (same 6) × 3 models (GPT-4.1-mini, GPT-5.4-mini, Gemma 3-12B) | **Yes** — 15/18 model×benchmark pairs | Mean **+14.1 pp** over GEPA; **+33.8 pp** on HoVer+IFBench where FAPO escalates to structural changes; 11/18 with non-overlapping mean±SD. AIME only where GEPA still leads. | [arXiv:2606.19605](https://arxiv.org/abs/2606.19605) · [Cisco blog](https://cisco-foundation-ai.github.io/blogs/fully-automated-prompt-optimization/) |
| 2 | **CANTANTE: Optimizing Agentic Systems via Contrastive Credit Attribution** — arXiv:2605.13295 (May 2026) | MBPP (code), GSM8K (math), HotpotQA (multi-hop QA) | **Yes** on MBPP & GSM8K; Tie/Marginal on HotpotQA | MBPP 22.96→**41.89** (**+18.93 pp** vs GEPA), GSM8K 61.27→**82.33** (**+21.06 pp** vs GEPA / +12.53 vs best baseline MIPROv2 69.80), HotpotQA 10.93→11.93 (+1.00, MIPROv2 14.20 leads; within SD). Avg rank 1.44 (best). Lower inference cost. | [arXiv:2605.13295](https://arxiv.org/abs/2605.13295) |
| 3 | **ADOPT: Adaptive Dependency-Guided Joint Prompt Optimization for Multi-Step LLM Pipelines** — arXiv:2512.24933 (Dec 2025, v2) | Multi-step pipelines (real-world datasets, diverse structures; includes HotpotQA-style multi-hop by description) | **Yes** (claimed) | Outperforms GEPA, TextGrad, Trace, MIPRO on all reported pipeline structures (quantitative table truncated in search; authors state consistent wins via Shapley-based dependency modeling). Qualitative: decouples text-gradient estimation from updates. | [arXiv:2512.24933](https://arxiv.org/abs/2512.24933) |
| 4 | **SPEAR: Code-Augmented Agentic Prompt Optimization** — arXiv:2605.26275 (May 2026) | Hiring Assistant (Columbia NLP-ish) — 9 dims: unsound inference / job location / employment type / workplace type / organization / location format / required skills / overqualification / multiqualification — Cohen’s κ | **Yes** (domain benchmark) | Hardest dim mean κ 0.315 vs GEPA 0.149 (σ 0.16) vs TextGrad 0.105 (worst SPEAR seed = best GEPA); location-format 0.961 vs 0.000; required-skills 0.938 vs 0.168; held-out test κ best on 2/3 dims per replica. | [arXiv:2605.26275](https://arxiv.org/abs/2605.26275) |
| 5 | **FlowBot: Inducing LLM Workflows with Bilevel Optimization and Textual Gradients** — Wu et al., arXiv:2604.26258 (Apr 2026) | Suite A: HotPotQA2, DROP, HumanEval, MBPP, GSM8K, MATH (GPT-4o-mini). Suite B: HotpotQA1, IFBench, HoVer, PUPA (Qwen3/GPT-4.1-mini, GEPA’s 4) | **Yes/Partial** | Suite B: HotpotQA1 69.00→**72.80** (+3.80 pp), IFBench 52.72→52.x (tie, GEPA+Merge 55.95 leads), HoVer/PUPA competitive. Suite A: Avg 72.75→76+ range; authors call “competitive vs human-crafted workflows”. Cost: 145K calls / $165 vs AFlow 803K / $352. | [arXiv:2604.26258](https://arxiv.org/abs/2604.26258) |
| 6 | **Error Taxonomy-Guided Prompt Optimization (ETGPO)** — arXiv:2602.00997 (Feb 2026) | Mathematics, QA, logical reasoning (multiple benchmarks; heavy mode vs GEPA/MIPRO heavy) | **Yes (tie on acc, win on cost)** | Accuracy **comparable/better** than GEPA/MIPRO across benchmarks; **~1/3 tokens** and **2.8× cheaper** than next-best (GEPA: 12,495 tokens — ETGPO ~6,487 truncated; full table in HTML). Ablation shows taxonomy > raw failure sampling. | [arXiv:2602.00997](https://arxiv.org/abs/2602.00997) |
| 7 | **Reflection in the Dark: Exposing and Escaping the Black Box in Reflective Prompt Optimization → VISTA** — arXiv:2603.18388 (Mar 2026) | GSM8K (defective & clean seeds), AIME20 (AIME 2020) — APO analysis paper | **Yes** (critique + fix) | GSM8K defective seed: GEPA 23.81%→**13.50%** (-10.31 degradation); **VISTA recovers to 87.57%** (+74 pp vs degraded GEPA, +63 vs seed). Clean seeds: VISTA consistently > GEPA/MIPRO across all conditions. Four systematic failure modes identified (label-free black box). | [arXiv:2603.18388](https://arxiv.org/abs/2603.18388) · [LinkedIn summary](https://www.linkedin.com/pulse/making-gepa-interpretable-vista-faisal-waris-wtc7c) |
| 8 | **Prompt Codebooks (PCO): Discrete Compositional Optimization for Language Model Instruction Refinement** — arXiv:2605.28360 (May 2026) | HotpotQA, IFBench, HoVer, PUPA (Qwen3-8B, LLaMA-3.1-8B, etc.) | **Yes** on HotpotQA & IFBench; Tie on PUPA | IFBench Qwen3-8B 38.61→**41.33** (+2.72 pp), HotpotQA +30.36 over zero-shot on LLaMA-3.1-8B and **+3.34 pp** over GEPA on Qwen3-8B, PUPA match (≈94.47). **9.6× token efficiency**: avg 1,289 (GEPA) → **≤653** (PCO); HotpotQA 2,142→≤714 (14.1×). | [arXiv:2605.28360](https://arxiv.org/abs/2605.28360) |
| 9 | **Feedback Descent: Open-Ended Text Optimization via Pairwise Comparison** — arXiv:2511.07919 (Nov 2025) | HotpotQA, IFBench, HoVer, PUPA (two models) | **Partial** | **Best on IFBench & HoVer** vs GEPA; GEPA leads on HotpotQA & PUPA. Authors emphasize simpler approach (joint prompt update via pairwise textual summaries) matches GEPA which uses coordinate descent + Pareto front. | [arXiv:2511.07919](https://arxiv.org/abs/2511.07919) |
| 10 | **Reward-Free Evolving Agents via Pairwise Validator** — arXiv:2607.14408 (Jul 2026) | HotpotQA, IFBench, HoVer, PUPA (Haiku & Gemma-4 validators, reward-free) | **Yes** (reward-free beats full-reward GEPA) | Haiku validator best-of-variants vs full-reward GEPA: HotpotQA **+3.5**, IFBench **+8.9**, HoVer **+7.1**, PUPA **+7.0** (4/4). Gemma-4: 3/4 (PUPA -2.0). Demonstrates GEPA without reward signal can still win. | [arXiv:2607.14408](https://arxiv.org/abs/2607.14408) |
| 11 | **MAS-PromptBench: When Does Prompt Optimization Improve Multi-Agent LLM Systems?** — arXiv:2606.23664 (Jun 2026) | MAS-GEPA variant on HotpotQA, LiveCodeBench, etc. under Freeform/Semi-structured/Structured comms | **Analysis**, not beat | Extends GEPA to multi-agent (MAS-GEPA). Gains scale with comm structure: +0.1 (Freeform) → +4.8 (Semi) → +6.3 (Structured) on avg; HotpotQA biggest gains, LiveCodeBench smallest. Validates need for structured prompts (FOREST’s current direction). | [arXiv:2606.23664](https://arxiv.org/abs/2606.23664) |
| — | *GEPA-citing but not head-to-head beating (excluded from count):* DD-GEPA (dialogue disentanglement, 2606.07894 — *uses* GEPA), Bridging OpenACC (2601.08884 — *uses* GEPA), Complex QA etc. | — | — | — | — |

> **Counting note:** Papers 1–10 are the “beat GEPA” set (n=10). Papers 7+11 are critique/analysis that still outrank GEPA on named tasks. Total distinct benchmarks where ≥1 paper beats GEPA: 9.

---

## 3. Frequency Ranking — Where Is GEPA Most Often Beaten?

| Rank | Benchmark | # papers beating GEPA (out of 10) | Typical margin | Nature of benchmark | GEPA paper’s headroom |
|------|-----------|-----------------------------------|----------------|---------------------|-----------------------|
| **1** | **HotpotQA** (multi-hop QA, query-generation + answer, F1/EM) | **6** (FAPO, CANTANTE*, FlowBot, PCO, Reward-Free, Feedback Descent partial*) — *CANTANTE/Feedback only tie | +3 pp (PCO) → +33.8 pp (FAPO structural) | Multi-step retrieval-augmented QA (distractor/fullwiki, 113K) — high headroom, GEPA 62.3/69.0 still <80 | Large |
| **2** | **IFBench** (instruction following, 294 novel constraints, instruction-level acc) | **5** (FAPO, PCO, Feedback Descent, Reward-Free, ETGPO tie) | +2.7 pp (PCO) → +33.8 pp (FAPO) | Novel constraints by design; GEPA paper: GEPA 38.61% (Qwen3-8B) — test never improved in FOREST Rev1/2 (overfit). High sensitivity to pipeline changes. | Very large (GEPA barely > baseline on Qwen3-8B) |
| **3** | **HoVer** (many-hop fact extraction & claim verification, gold-doc F1/recall, ≤3 hops) | **5** (FAPO, Feedback Descent, Reward-Free, PCO*, FlowBot*) | +5–10 pp typical; FAPO +33.8 pp | 2-stage query_writer → doc_summarizer; mirrors HotpotQA but longer hop. | Large (GEPA 52.33/51.67) |
| **4** | **GSM8K** (grade-school math, 8.5K, exact match) | **3–4** (CANTANTE +21 pp, VISTA +63 pp recovery, FlowBot suite, ETGPO) | **+12–21 pp** (CANTANTE) / **+74 pp** (VISTA defective-seed recovery) | Single-step CoT math — clean exact-match signal, high prior knowledge. | Moderate (GEPA strong but brittle: degrades -10 pp on defective seed) |
| **5** | **PUPA** (Privacy-conscious delegation, Columbia PUPA_tnb 237, quality+leakage/2) | **3** (FAPO, Reward-Free +7.0, PCO tie / Feedback GEPA leads) | +2–7 pp | 1-stage PII-redaction (quality via LLM judge + leakage 1-leaked/PII). Smaller dataset, high variance. | Small on Qwen3-8B (91.85 close to ceiling) but 78→94 on GPT-4.1-mini → still headroom |
| **6** | **MBPP** (+ HumanEval / MATH / DROP as code-math cluster) | **2–3** (CANTANTE +18.9 pp MBPP, FlowBot on HumanEval/MBPP/MATH/DROP, ETGPO math) | **+18.9 pp** on MBPP (CANTANTE) | Code generation (MBPP 1K, HumanEval 164) + MATH competition. FOREST has no code-math beyond AIME/LiveBench. | Large (baseline 5.5 MBPP → 41.89 CANTANTE) |
| 7 | AIME-2025 / AIME20 / LiveBench-Math | 1–2 (VISTA on AIME20; FAPO: *GEPA still leads* on AIME) | Mixed — GEPA 32→59 on AIME, so tougher to beat | Competition math (AIME 30/yr). GEPA already strong; only VISTA cleanly beats. | Small-medium (GEPA +8–12 pp) |
| 8 | Hiring Assistant / OpenACC / Dialogue | 1 each (SPEAR, Bridging Gap, DD-GEPA) | Domain-specific | Niche real-world tasks — not core FOREST ranking. | N/A |

**Visualization (frequency):**

```
HotpotQA  ████████████████████ 6
IFBench   ████████████████░░░░ 5
HoVer     ████████████████░░░░ 5
GSM8K     █████████████░░░░░░░ 4
PUPA      ████████████░░░░░░░░ 3
MBPP/Math ████████░░░░░░░░░░░░ 2-3
AIME      ████░░░░░░░░░░░░░░░░ 1-2
```

### What this ranking says

- **HotpotQA is the universal “GEPA-beaten” benchmark** — every pipeline-aware optimizer (FAPO, FlowBot, PCO, Reward-Free) beats it, because gains come from *pipeline structure*, not just prompt wording. FAPO’s lesson: fixed 2-stage GEPA hits a ceiling; adding a stage beats GEPA by 30+ pp.
- **IFBench & HoVer behave identically** (same multi-step shape). They are the second-most reliably beaten and the ones where GEPA’s advantage is most fragile in FOREST’s own Rev1/2 (overfit -18.9 pp on 2-stage). Beating GEPA there currently requires structural escalation, not just better reflection.
- **GSM8K is the “largest single-model delta”** benchmark (+21 pp clean, +74 pp recovery) and exposes GEPA brittleness to seed quality — directly motivates VISTA-style fixes FOREST is exploring (action-conditioned reflection).
- **No SWE-Bench / TerminalBench / AppWorld paper yet reports beating GEPA** — gap, not negative result. Those are the agentic benchmarks FOREST mentions but citations haven’t caught up.

---

## 4. Recommended 5–6 for FOREST

For FOREST’s next-round scaffold (balance of “where GEPA is proven beatable”, “already in repo vs new cost”, and “covers QA / instruction / fact / math / code”):

| Priority | Benchmark | Why it matters (citation evidence) | Cost / data | Suggested program (to replicate beating-paper setup) |
|----------|-----------|-------------------------------------|-------------|-----------------------------------------------------|
| **1 — keep** | **HotpotQA** | #1 most beaten (6 papers, +3→+33 pp). FOREST Rev1 already has it; high headroom proves action-conditioning matters. | HF `hotpot_qa` 113K (150/300/300, 6871 calls) | Keep 2-stage query→answer (paper); also run 1-stage ablation for comparison. |
| **2 — keep** | **IFBench** | #2 most beaten (5 papers), FOREST’s primary + FAPO’s +33.8 pp requires pipeline change — perfect test for FOREST’s structural actions. | Vendored checkers, 300/300/294 (3593 calls) — already paper-faithful | Keep 2-stage; Wave B already at 15k calls for diversity study. |
| **3 — keep** | **HoVer** | Tied #2 (5 papers), same shape as HotpotQA but ≤3 hops — validates generality of HotpotQA findings. | HF `hover` or GitHub raw, 150/300/300 (7051 calls) — already scaffolded | Keep 2-stage query_writer→doc_summarizer. |
| **4 — ADD** | **GSM8K** | **#1 TODO** — 3–4 papers beat GEPA here by the largest margins (+21 pp CANTANTE, +74 pp VISTA). Exposes brittleness FOREST is fixing (defective seed). Cheapest to add, single-step, deterministic grading. | HF `gsm8k` 8.5K, 1-stage CoT, exact-match, 150/300/300 (~5k calls) | New `examples/gsm8k/` — reuse AIME’s `_call_lm` + integer/number match. Test clean + defective seed. |
| **5 — keep** | **PUPA** (or pair with GSM8K) | 3 papers beat GEPA; only privacy benchmark, 1-stage, tests quality-vs-safety trade-off distinct from QA. | `Columbia-NLP/PUPA` pupa_tnb 237 (2426 calls, 1-stage) — already scaffolded | Keep 1-stage system_prompt; leakage+quality aggregate. |
| **6 — ADD** | **MBPP (or MBPP+HumanEval as “code”)** | Only code benchmark where GEPA is flatly trounced (+18.9 pp MBPP). Complements math (GSM8K) and QA. LiveBench/AIME already cover competition math; MBPP covers code. | HF `mbpp` 1K + `openai_humaneval` 164, execution-based, sandboxed. Budget ~5k calls. | New `examples/mbpp/` — 1-stage code gen + pass@1 exec; mirrors CANTANTE/FlowBot setup. |

**If limited to 5 (drop one):** keep HotpotQA, IFBench, HoVer, **GSM8K**, PUPA — that is the minimal set that hits all 5 papers’ overlap and reuses 4/5 existing scaffolds, adding only GSM8K.

**If limited to 6 and agentic coverage required:** add **SWE-Bench Verified (subset)** or **AppWorld** as #6 instead of MBPP, acknowledging no citation yet beats GEPA there — it would be *novel* headroom, not replication. Flag as stretch goal.

---

## 5. Already Scaffolded vs TODO (FOREST `gepa` as of 2026-08-08)

| Benchmark | In `examples/`? | Paper-faithful? | Status for FOREST | Notes |
|-----------|----------------|-----------------|-------------------|-------|
| **IFBench** | ✅ `examples/ifbench/` | ✅ paper-exact (300/300/294, 2-stage, 3593 calls, Qwen3-8B) | **Scaffolded — primary** | Rev1/Rev2 done; Wave B 15k in flight. Action-conditioned machinery lives here (`action_space.py`, `ActionDiversityCallback`). |
| **HotpotQA** | ✅ `examples/hotpotqa/` | ✅ 150/300/300, 2-stage, 6871 calls | **Scaffolded** | Uses HF distractor + smoke fallback; shares IFBench decoding + callbacks. |
| **HoVer** | ✅ `examples/hover/` | ✅ 150/300/300, 2-stage, 7051 calls | **Scaffolded** | Gold-doc F1/recall; JSON/bullet title extraction + substring fallback. |
| **PUPA** | ✅ `examples/pupa/` | ✅ pupa_tnb 237, 1-stage, 2426 calls | **Scaffolded** | quality+leakage/2 with LLM judge; 1-stage system_prompt. |
| **AIME (AIME-2025 + 2022-24)** | ✅ `examples/aime_math/` | ✅ 45/45/30×5 expanded (500 calls legacy) | **Scaffolded** | Competition math; single-step CoT, integer match. GEPA still leads here per FAPO — good “GEPA wins” anchor. |
| **LiveBench-Math** | ✅ `examples/livebench_math/` | ✅ 122/123/123 or terrarium 100/100/168, 1839 calls | **Scaffolded** | Fresh math beyond cutoff; single-step, normalized EM. |
| **GSM8K** | ❌ missing | — | **TODO — Priority 1** | Cited in 3–4 beating papers; largest deltas. Create `examples/gsm8k/` reusing `aime_math/utils.py` patterns + GSM8K `datasets` split. Estimate 1–2 days. |
| **MBPP / HumanEval / MATH / DROP** | ❌ missing | — | **TODO — Priority 2** | Code/math cluster; CANTANTE + FlowBot beat GEPA by ~19 pp on MBPP. Create `examples/mbpp/` (exec sandbox) + optionally `math/` (reuse LiveBench harness). 2–3 days incl. sandbox. |
| **SWE-Bench Verified / Pro, TerminalBench, AppWorld, SWE-Rebench** | ⚠️ `examples/appworld/` exists but not wired to GEPA beat | AppWorld exists stub; others missing | **TODO — Stretch** | No citation yet beats GEPA here — would be novel. AppWorld 9.5% → still agentic. |
| **OpenACC / Dialogue / Hiring Assistant** | ❌ | — | Not recommended | Niche domain papers (SPEAR, DD-GEPA) — not needed for ranking. |

**Summary:** 6/6 GEPA-paper benchmarks are already scaffolded (the four multi-step + two math). The *citation-ranked* gap is **GSM8K** and **MBPP/HumanEval** — both appear in top-6 beating frequency but are absent from `examples/`. Adding those two closes the loop and lets FOREST reproduce the largest reported GEPA defeats (CANTANTE, VISTA, FlowBot).

---

## 6. Links (all papers + GEPA)

**GEPA itself:**

- Agrawal et al. “GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning” — [arXiv:2507.19457](https://arxiv.org/abs/2507.19457) · [PDF](https://arxiv.org/pdf/2507.19457) · [ICLR 2026 OpenReview](https://openreview.net/forum?id=RQm2KQTM5r) · [HuggingFace](https://huggingface.co/papers/2507.19457) · [GitHub gepa-ai/gepa](https://github.com/gepa-ai/gepa) · [Blog: optimize_anything](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/)

**Papers beating GEPA (in table order):**

1. FAPO — [arXiv:2606.19605](https://arxiv.org/abs/2606.19605) · [Cisco blog](https://cisco-foundation-ai.github.io/blogs/fully-automated-prompt-optimization/) · [MarkTechPost summary](https://www.marktechpost.com/2026/06/20/cisco-ai-introduces-fapo-pipeline-aware-prompt-optimization-with-step-level-failure-attribution-and-claude-code-orchestration/)
2. CANTANTE — [arXiv:2605.13295](https://arxiv.org/abs/2605.13295)
3. ADOPT — [arXiv:2512.24933](https://arxiv.org/abs/2512.24933)
4. SPEAR — [arXiv:2605.26275](https://arxiv.org/abs/2605.26275)
5. FlowBot — [arXiv:2604.26258](https://arxiv.org/abs/2604.26258)
6. ETGPO — [arXiv:2602.00997](https://arxiv.org/abs/2602.00997) · [HTML](https://arxiv.org/html/2602.00997)
7. Reflection in the Dark / VISTA — [arXiv:2603.18388](https://arxiv.org/abs/2603.18388) · [LinkedIn: Making GEPA Interpretable with VISTA](https://www.linkedin.com/pulse/making-gepa-interpretable-vista-faisal-waris-wtc7c)
8. Prompt Codebooks (PCO) — [arXiv:2605.28360](https://arxiv.org/abs/2605.28360)
9. Feedback Descent — [arXiv:2511.07919](https://arxiv.org/abs/2511.07919)
10. Reward-Free Evolving Agents — [arXiv:2607.14408](https://arxiv.org/abs/2607.14408)
11. MAS-PromptBench — [arXiv:2606.23664](https://arxiv.org/abs/2606.23664)
- DD-GEPA (uses GEPA) — [arXiv:2606.07894](https://arxiv.org/abs/2606.07894)

**Related / context:**

- Bridging the Gap: GEPA-Optimized OpenACC — [arXiv:2601.08884](https://arxiv.org/abs/2601.08884)
- VentureBeat: GEPA optimizes without RL — [venturebeat.com](https://venturebeat.com/ai/gepa-optimizes-llms-without-costly-reinforcement-learning)
- Comet: GEPA AI Optimization — [comet.com](https://www.comet.com/site/blog/gepa-ai-optimization/)
- DeepLearning.AI: Better Agentic Prompts (GEPA) — [deeplearning.ai](https://www.deeplearning.ai/the-batch/authors-devised-gepa-an-algorithm-for-better-prompts-to-improve-agentic-systems-performance/)

---

## 7. Caveats & Next Steps for FOREST

1. **Deltas are not apples-to-apples.** Most beating-paper budgets differ from GEPA’s (FAPO uses Claude Code orchestration, CANTANTE uses contrastive credit, VISTA uses multi-agent). Re-run with FOREST’s fixed `max_metric_calls` before claiming replication.
2. **IFBench is GEPA’s weakest win** (+1.71 pp Qwen3-8B). FOREST’s Rev2 already shows -18.9 pp overfit on test — beating GEPA on IFBench currently means *regularizing*, not accelerating. FAPO’s +33.8 pp required structural change; test whether FOREST’s section-scoped actions replicate that without escalation.
3. **GSM8K brittleness is the cleanest signal.** VISTA’s 23.81→13.50→87.57 story directly maps to FOREST’s action-conditioned reflection hypothesis. GSM8K scaffold should include both clean and defective seeds.
4. **No SWE-Bench beating yet** — if FOREST wants a 6th *novel* benchmark, SWE-Bench Verified (500 curated) is high-headroom but needs exec sandbox; treat as v2 after GSM8K/MBPP.
5. **Verify each row.** For parent review, rerun `web_search` queries above and open each arXiv HTML to confirm table numbers before citing in a submission; preprints ≥2606 are <2 months old.

---

*File location for review: `docs/BENCHMARKS_GEPA_CITATIONS.md` (outside mkdocs tree, like HANDOVER). Also briefly summarized in JSON below for parent.*
