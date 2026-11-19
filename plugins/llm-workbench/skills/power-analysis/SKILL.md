---
name: power-analysis
description: Use to decide how many prompts an llm-workbench comparison needs — run power analysis on a pilot run (or from an assumed effect size) and interpret the recommended sample size. Triggers on "how many prompts do I need", "power analysis", "sample size", "is my pilot big enough", "what N should I use", "did I have enough power".
---

# Power analysis — how many prompts

Turns a per-prompt effect size into the number of prompts needed to detect it reliably. Needs
the `analysis` extra (`uv add "llm-workbench[analysis]"`). Read
`../../references/stats-interpretation.md` for the concepts; this skill is the procedure.

Why it's needed: the number of prompts depends on how *different* the candidate models actually
are on your task — a gap you can't know in advance. Close models need many more prompts than
far-apart ones. A pilot measures that gap; power analysis turns it into a target N.

## From a pilot run (recommended)

```bash
llm-workbench power --from-run runs/pilot --reference ollama/llama3.1
```

Reports, per model vs. the reference: the observed effect, the power the pilot already achieved,
and the N needed to reach your target power — plus a recommended N.

**Interpret:**
- Effect size `dz` = mean of per-prompt differences / SD of those differences. Bigger = easier
  to detect = smaller N.
- **Use the recommended N for the smallest (hardest-to-detect) effect.** That pair sets the
  sample size for the whole study; sizing to an easier pair under-powers the hard one.
- Tune `--alpha` (default 0.05), `--power` (default 0.8), `--alternative`.

## Without a pilot

If you can only assume an effect size, size it directly:
```bash
llm-workbench power --effect-size 0.5               # N at alpha=0.05, power=0.8
llm-workbench power --mean-diff 0.4 --sd-diff 0.8   # dz = 0.4/0.8 = 0.5 → same N
```
Warn the user: a guessed effect can be off by an order of magnitude. Prefer a real pilot.

## One caveat

Sample sizes use the paired t-test; `analyze` uses the Wilcoxon signed-rank test (~95% as
efficient). **Add a small margin** to the recommended N so the nonparametric test you'll
actually run stays adequately powered.

**Gate for the guide:** a target N chosen (the recommended N for the smallest effect, plus a
margin). Hand it to the `build-dataset` skill to grow the test set to that size.
