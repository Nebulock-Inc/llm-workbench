---
name: analyze-results
description: Use to analyze a finished llm-workbench run and turn it into a model recommendation — run the statistics (reliability, significance, effect size, equivalence) and interpret them for a decision. Triggers on "analyze my run", "which model won", "which model is best", "are these models significantly different", "are they equivalent / the same", "read my results", "interpret analysis.md".
---

# Analyze results and pick the winner

`analyze` runs the statistics appropriate to the paired design and writes `analysis.json`,
`analysis.md`, and comparison plots. Needs the `analysis` extra. Read
`../../references/stats-interpretation.md` for what each test means; this skill is the procedure
and how to decide.

```bash
llm-workbench analyze runs/experiment-1 --reference anthropic/claude-sonnet-4-5
```

Options: `--eval`/`--scorer` (restrict to specific metrics), `--exclude-metric` (keep a metric
out of the composite, e.g. one on a different scale), `--alpha` (0.05), `--margin` (TOST, 0.2),
`--out-dir`, `--plots`/`--no-plots`, `--verbose`/`--simple`. Full list in
`../../references/cli-cheatsheet.md`.

## Read it in this order

1. **Reliability first.** Each model's task-completion rate. The quality stats are
   complete-case, so a model that fails often can still post a high quality score — a high
   scorer that only answers 70% of prompts may lose to a 99%-reliable one. Read reliability and
   quality **together**; never report a winner on quality alone.

2. **Overall + per-metric means** for orientation.

3. **The statistical battery on the quality composite** (one family of tests on the aggregate,
   to keep multiple-comparisons clean):
   - **Friedman + Kendall's W** — is *anyone* different (≥3 models)? W is the effect size.
   - **Holm-corrected pairwise Wilcoxon** — *which* pairs differ.
   - **Mixed-effects** — each model's difference from the reference, with significance (graded
     data only).
   - **Paired Cohen's d** — how big each pairwise difference is.
   - **TOST equivalence** (`--margin`) — whether two models are *statistically the same* within
     a margin. The only test that can affirm "equivalent"; significance tests can't.

   To drill into one metric instead of the composite, filter `--eval`/`--scorer` and re-run.

## Turn it into a recommendation

- Significant difference (Friedman + Holm pairwise) with a meaningful Cohen's d **and**
  acceptable reliability → prefer that model.
- Non-significant **and** TOST says equivalent within the margin → the models are
  interchangeable on quality; decide on cost, latency, or reliability instead.
- Non-significant but **not** TOST-equivalent → underpowered or genuinely uncertain; the
  `power-analysis` skill can say whether more prompts would resolve it.

State the recommendation in plain language, grounded in the numbers, and name the tradeoff when
one model wins quality but another wins reliability/cost.

**Gate for the guide:** a written recommendation that reads reliability, significance, effect
size, and equivalence together — not a single headline number.
