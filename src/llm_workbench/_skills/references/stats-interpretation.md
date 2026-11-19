# Reading the statistics

llm-workbench uses a **paired (repeated-measures)** design: every prompt is run on every
model, so the unit of analysis is the per-prompt *difference* between models. That is what
makes the tests below legitimate, and why sample sizes are smaller than an unpaired design.

## Power analysis (`power`) — how many prompts

- **Effect size `dz`** = (mean of the per-prompt differences) / (SD of those differences).
  Bigger `dz` = models are further apart = fewer prompts needed. You can't know `dz` in
  advance; a pilot estimates it.
- `power --from-run runs/pilot --reference <m>` reports, per model vs. the reference: the
  observed effect, the power the pilot already achieved, and the N needed to reach your target
  power. **Use the recommended N for the smallest (hardest-to-detect) effect** — that gap sets
  the sample size for the whole study.
- Without a pilot, `--effect-size dz` (or `--mean-diff`/`--sd-diff`) sizes it from an assumption
  — but a guessed effect can be off by an order of magnitude; a real pilot is worth the cost.
- Defaults: `--alpha 0.05`, `--power 0.8`. Sample sizes use the paired t-test; `analyze` uses
  Wilcoxon (~95% as efficient), so **add a small margin** to the N.

## Reliability (read this first in `analyze`)

Each model's **task-completion rate** = fraction of prompts that produced a usable answer,
over every recorded row including failures. The quality stats below are **complete-case** —
they cover only prompts a model actually answered, so a model that fails often can still post
a strong quality score. **Always read reliability and quality together**: a high-quality model
that only answers 70% of prompts may be worse in practice than a slightly lower-quality one
that answers 99%.

## Quality statistics (`analyze`, on the quality composite)

The full battery runs on a single **quality composite** (aggregate across quality metrics),
not per-metric, to keep the multiple-comparisons story clean. To drill into one metric, filter
`--eval`/`--scorer` and re-run. Reported where applicable:

- **Friedman test + Kendall's W** — omnibus across ≥3 models: *is anyone different at all?*
  Friedman gives the p-value; W (0–1) is the effect size / agreement of the ranking.
- **Pairwise Wilcoxon signed-rank, Holm-corrected** — the post-hocs: *which specific pairs
  differ?* Holm controls the family-wise error across the pairs.
- **Mixed-effects model** (`score ~ model` + random intercept per prompt) — each model's
  difference from the reference, with significance. Graded data only; skipped for binary
  pass/fail.
- **Paired Cohen's d** — practical effect-size magnitude per pair (how big, not just whether).
- **TOST equivalence** (`--margin`, default 0.2 score points) — whether two models are
  *statistically the same* within a margin. Significance tests can only fail to find a
  difference; TOST is what actually says "these are equivalent."

How to decide: if Friedman is significant, read the Holm-corrected pairwise results + Cohen's d
for the candidate vs. the reference. A significant difference with a meaningful `d` and
acceptable reliability → prefer that model. If differences are non-significant **and** TOST
says equivalent within your margin → the models are interchangeable on quality, so decide on
cost/latency/reliability instead.

## Judge calibration (`calibrate`)

Before trusting judge scores, check they match a human's on the same rubric:

- **Bias** = mean(judge) − mean(human). Positive = the judge scores higher than you.
- **Exact-agreement** and **within-1-agreement** rates on the rubric.

A large bias or low agreement means the rubric or judge prompt is ambiguous — revise it and
re-run before drawing conclusions. Calibrate on the pilot (before the full run) and/or on the
full run.

Deeper background on each test lives in the README's "Result Analysis" and "Power Analysis"
sections (https://github.com/Nebulock-Inc/llm-workbench#readme).
