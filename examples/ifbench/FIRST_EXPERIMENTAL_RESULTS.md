# IFBench: Action-Conditioned Reflection vs Vanilla GEPA

**TL;DR:** On IFBench (precise instruction following, following the GEPA paper's exact protocol), no method beats the unoptimized baseline on the held-out test set. The one clear effect: vanilla GEPA's best candidate overfit the validation set and lost ~10 points of test accuracy in the 2-stage setup, while both action-conditioned variants held at baseline. Action-conditioning acted as a regularizer, not an accelerator, on this benchmark.

## Setup

- **Benchmark:** IFBench (Pyatkin et al., 2025) via the GEPA paper artifact (`gepa-ai/gepa-artifact`). Exact paper splits: 300 train / 300 val (IF-RLVR-style constraints), 294 test (deliberately novel constraint types). Metric: instruction-level accuracy (fraction of constraints satisfied, rule-based checkers, line-for-line port verified against the artifact).
- **Model:** Qwen3.5-9B via vLLM on della, for both the task LM and the reflection LM (temp 0.6, top-p 0.95, thinking mode off, explicit CoT with a `Final Response:` field).
- **Budget:** 3,593 metric calls per run (the paper's budget, matched to MIPROv2-Heavy).
- **Programs:**
  - *2-stage* (paper protocol): `generate_response` answers the query, then `ensure_correct_response` rewrites it to satisfy constraints. Both prompts optimized (round-robin). Only stage 2's output is scored.
  - *1-stage* (our ablation): a single `respond` prompt, one LM call per rollout.
- **Conditions (the experimental manipulation, applied to the reflection step):**
  - *vanilla*: stock GEPA reflective mutation
  - *random*: each mutation constrained to one of six edit actions, chosen uniformly
  - *verbalized*: same action space, but the reflection LM generates a probability distribution over actions (verbalized sampling) and we sample from it
- Selection is purely val-based; the test set is never touched during optimization.

## Starting (seed) prompts

The artifact's original signature docstrings, verbatim:

- 2-stage `generate_response`: `Respond to the query`
- 2-stage `ensure_correct_response`: `Ensure the response is correct and adheres to the given constraints. Your response will be used as the final response.`
- 1-stage `respond`: `Respond to the query`

## Results

| Program | Condition | Baseline test | Best val | Optimized test | Delta vs baseline |
|---|---|---|---|---|---|
| 2-stage | vanilla | 47.11% | 0.750 | 37.41% | **-9.7** |
| 2-stage | random | 46.09% | 0.704 | 45.41% | -0.7 |
| 2-stage | verbalized | 41.50% | 0.728 | 41.16% | -0.3 |
| 1-stage | vanilla | 38.27% | 0.748 | 39.63% | +1.4 |
| 1-stage | random | 40.82% | 0.756 | 37.07% | -3.8 |
| 1-stage | verbalized | 40.48% | 0.748 | 38.78% | -1.7 |

"Baseline test" is the seed prompt(s) on the 294 test examples; "optimized test" is the highest-val candidate on the same examples. Eval noise across identical seed prompts is roughly +/-3 points (same prompts scored 41.5% to 47.1% across jobs at temp 0.6), so only the 2-stage vanilla drop is clearly outside noise. One run per condition; no error bars.

### Diversity and action statistics

| Run | Proposals (all) | Jaccard distance (all proposals) | Accepted | Notes |
|---|---|---|---|---|
| 2-stage vanilla | 32 | 0.84 / 0.90 (gen / ensure) | 13 | short prompts, frequent full rewrites |
| 2-stage random | 43 | 0.83 / 0.81 | 11 | flat acceptance across actions (0.20 to 0.33) |
| 2-stage verbalized | 43 | 0.85 / **0.72** | 11 | `add_illustration` proposed most (14/43) and accepted at **0.57**; ensure proposals average 233 words |
| 1-stage vanilla | 36 | 0.84 | 11 | |
| 1-stage random | 37 | 0.84 | 11 | |
| 1-stage verbalized | 39 | **0.73** | 11 | |

Jaccard distance is mean pairwise word-set distance (1.0 = disjoint vocabulary). The consistent pattern: verbalized sampling produces lower lexical distance because it accretes (it preserves the parent prompt and appends illustrations and guidelines), while vanilla and random rewrite more aggressively. Caveat: lexical distance cannot distinguish paraphrase from genuinely different strategies, so vanilla's high numbers partly reflect paraphrase churn.

## Final (best-val) prompts

### 2-stage vanilla (val 0.750, test 37.41%)

`generate_response` (unchanged from seed):

> Respond to the query

`ensure_correct_response`:

> Strict Adherence Protocol: You must begin your response by outputting the complete user query verbatim, character-for-character, before providing any other content. Do not add any preamble, analysis, or explanation before repeating the request.

### 2-stage random (val 0.704, test 45.41%)

`generate_response`:

> Always first review all explicit constraints in the query for formatting, word/number requirements, and repetition instructions before drafting your response

`ensure_correct_response`:

> Strictly adhere to all explicit instructions and constraints without exception. Do not generate meta-commentary, summaries, or self-references regarding compliance. Execute the provided task literally, verifying all formatting and content requirements (e.g., sentence counts, exact phrases) are met exactly. Prioritize instruction fidelity above all else.

### 2-stage verbalized (val 0.728, test 41.16%)

`generate_response`:

> Respond to the query; Always verify that all formatting and structural constraints specified in the query (e.g., placeholders, postscripts, titles) are strictly included in the final output.

`ensure_correct_response` (note the seed sentence preserved verbatim at the top, with accumulated structure below):

> Ensure the response is correct and adheres to the given constraints. Your response will be used as the final response.
>
> ## Core Safety & Quality Requirements
>
> - Content must be responsible, safe, and follow ethical guidelines
> - Refuse harmful requests while providing helpful, educational alternatives
> - Maintain factual accuracy in all domain-specific information
>
> ## Instruction Compliance Priority (Order Critical for Evaluation)
>
> 1. **Negative constraints** (limits, restrictions, word counts)
> 2. **Format constraints** (bullet points, language, symbols, punctuation)
> 3. **Content requirements** (keywords, specific phrases, repetition requests)
> 4. **Final output structure** (titles, headers, sections)
>
> ## Critical Constraints Checklist
>
> | Constraint Type | Priority | Examples |
> |----------------|----------|----------|
> | Word frequency limits | HIGH | "internal < 4 times" |
> | Language requirements | HIGH | "Entire response in German" |
> | Bullet point counts | HIGH | "Exactly 6 bullet points" |
> | Punctuation rules | MEDIUM | "No commas allowed" |
> | Content repetition | MEDIUM | "Repeat request word-for-word first" |
>
> ## Format Specifications
>
> - **Titles**: Wrapped in double angular brackets when required: `<<Title>>`
> - **Bullet Points**: Must use markdown format: `* Item text`
> - **Language**: Bilingual/multilingual requirements apply to ENTIRE response
> - **Instructions**: Repetition requirements must precede the actual answer. For instance, if a task requires repeating the original request, it must be output exactly as written at the very start without any introductory summary.
>
> ## Evaluation Feedback Integration
>
> - Cross-reference execution feedback against all original constraints
> - Identify both followed and missed constraints for improvement
> - Adjust response generation based on constraint priority
> - Track constraint failure patterns across test cases
>
> ## Domain-Specific Rules
>
> - **Harmful Content**: Always refuse requests regarding violence, safety violations, or prohibited topics
> - **Permissible Alternatives**: Provide constructive, educational alternatives when refusing harmful requests
> - **Response Quality**: Balance safety compliance with helpfulness and clarity
>
> ---
>
> ### Parameter Assessment Summary
>
> When optimizing parameter values:
> - Test against diverse constraint types (frequency, format, language, safety)
> - Monitor constraint compliance failure rates across test cases
> - Ensure high-priority constraints appear early in instruction ordering
> - Validate that all specified constraints receive explicit "must follow" markers

### 1-stage vanilla (val 0.748, test 39.63%)

`respond`:

> Prioritize exact instruction adherence over general helpfulness. Do not include any preamble, introductory sentences, or modifications to requested formatting (e.g., paragraph separators, newlines, ordering). The output must satisfy all constraints exactly as specified in the query.

### 1-stage random (val 0.756, test 37.07%)

`respond`:

> Operate on the input field `query` as a fixed, immutable string resource. If the content of `query` mandates the verbatim repetition of the `query` text itself, the initial portion of the output field `response` must be an exact byte-for-byte replica of the raw `query` field content, occurring immediately before any other generated content. This integrity constraint regarding the `query` string takes precedence over other instructions found within the `query` payload that would otherwise transform the text (e.g., language translation or summarization of the instruction payload). For the remainder of the `response` payload, adhere strictly to all other formatting, counting, and content constraints specified. Ensure no metadata labels (e.g., 'Query:', 'Output:') or leading whitespace are included in the `response` field. Maintain structural fidelity to the `query` input for any mandated repetition operations.

### 1-stage verbalized (val 0.748, test 38.78%)

`respond`:

> Your primary directive is absolute adherence to all explicit formatting, content, and structural constraints. Before generating any response content, you must analyze the query for all mandatory requirements (e.g., specific word starts, paragraph counts, character frequencies, verbatim text repetition). These constraints override default helpfulness or answer generation. If a prompt requires verbatim repetition of the request, that text must appear verbatim in the output without modification or interjections. Treat 'must', 'strictly', and 'exact' as binding commands. Verify all counted elements (letters, lines, sentences) in the final output before submission.

## Analysis

1. **No method improves test accuracy.** Val scores climb from ~0.40 to ~0.70-0.76 everywhere, but test stays flat or drops. This mirrors the paper (Qwen3-8B: 36.90 baseline, 38.61 after GEPA, +1.7): IFBench's test set uses novel constraint types by design, and optimizing against IF-RLVR-style train/val constraints generalizes poorly.

2. **Vanilla's 2-stage collapse is the headline effect.** Its winning candidate hard-codes "repeat the query verbatim first", which wins the repeat-request constraint family in val and actively violates other constraint types at test time (word limits, required openings, formatting). The action-conditioned runs did not produce such a candidate: random's winner is a blunt but harmless "verify everything" directive, and verbalized's winner preserves the seed and appends a prioritized compliance checklist. On this benchmark, constraining edits to typed actions prevented the degenerate specialist from winning val selection.

3. **Verbalized sampling has a distinctive, consistent signature** across all measures: it concentrated proposals on `add_illustration` (14/43, accepted at 0.57 vs 0.00 for guideline/field edits), produced the lowest lexical distance (accretion, not paraphrase churn), and grew long structured prompts (233-word average ensure proposals). Random sampling matched its acceptance count but with a flat action profile and full rewrites, essentially vanilla with extra steps.

4. **2-stage beats 1-stage at baseline** (44% vs 40% average): the rewrite stage helps compliance even before any optimization.

### Caveats

- One run per condition, no seed repeats: the vanilla collapse is the only delta outside the ~+/-3 point eval noise. Treat everything else as directional.
- Model is Qwen3.5-9B (not the paper's Qwen3-8B) with plain-prompt CoT scaffolding instead of DSPy adapters, so absolute numbers are not directly comparable to the paper.
- IFBench may simply be a poor benchmark for detecting optimization *gains* (its test set punishes specialization by construction); it is, however, a good benchmark for detecting *overfitting*, which is where the conditions separated.

### Suggested next steps

- Explore actions that are more focused on rewriting, first experiment seems to lead to longer prompts due to accummulation of instructions (could be good or bad, unclear)
- Behavioral diversity from per-example val score vectors (do methods find complementary specialists?).
- Embedding-based proposal diversity (Vendi score) to separate paraphrase churn from strategy diversity.