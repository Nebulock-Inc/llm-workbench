"""Sample-size / power analysis for the paired (repeated-measures) comparison.

Models are compared on the same prompts, so the unit of analysis is the per-prompt
paired difference. Sample size uses the paired t-test power (a one-sample t on the
differences) with effect size dz = mean(diff) / SD(diff). The nonparametric Wilcoxon
that `analyze` uses is ~95% as efficient, so treat these N as a mild lower bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def n_for_effect(effect_size: float, alpha: float = 0.05, power: float = 0.8,
                 alternative: str = "two-sided") -> int:
    """Prompts needed to detect a paired effect size dz at the given alpha and power."""
    from statsmodels.stats.power import TTestPower

    if not effect_size:
        raise ValueError("effect_size (dz) must be non-zero.")
    n = TTestPower().solve_power(effect_size=abs(effect_size), alpha=alpha, power=power,
                                 alternative=alternative)
    return int(math.ceil(n))


def achieved_power(effect_size: float, nobs: int, alpha: float = 0.05,
                   alternative: str = "two-sided") -> float:
    """Power to detect a paired effect size dz given nobs prompts."""
    from statsmodels.stats.power import TTestPower

    if not effect_size or nobs < 2:
        return 0.0
    return float(TTestPower().power(effect_size=abs(effect_size), nobs=nobs, alpha=alpha,
                                    alternative=alternative))


@dataclass
class PairPower:
    model: str
    reference: str
    dz: float
    mean_diff: float
    n_pilot: int
    n_needed: int | None       # None when dz == 0 (no detectable effect)
    power_at_pilot: float


def from_run(results_path, reference: str | None = None, alpha: float = 0.05,
             power: float = 0.8, alternative: str = "two-sided",
             standardize: bool = True, evals: set[str] | None = None,
             scorers: set[str] | None = None) -> dict:
    """Estimate required N from a pilot run's observed effect sizes on the quality composite.

    For each model vs the reference: the observed paired dz, prompts needed to detect
    that effect at the target power, and the power actually achieved at the pilot's N.

    By default (`standardize=True`) all metrics are pooled — graded quality AND binary
    accuracy metrics like the `ground_truth` checks — each z-scored within metric to a
    common SD scale, so the mean difference is expressed in SD units while the resulting dz
    is a dimensionless effect size and N is a prompt count. Pass `standardize=False` for
    the raw graded-quality-only composite. To size the study on a specific track instead
    (e.g. the AI / Maliciousness determination accuracy), filter to it with `evals` /
    `scorers`.
    """
    from llm_workbench.analysis import frame

    long_df = frame.load_long(results_path)
    if evals:
        long_df = long_df[long_df["eval"].isin(evals)]
    if scorers:
        long_df = long_df[long_df["scorer"].isin(scorers)]
    if long_df.empty:
        raise ValueError("No metrics match the given --eval / --scorer filter.")

    composite = frame.composite_quality(long_df, standardize=standardize)
    if composite is None or composite.shape[1] < 2:
        # Common footgun: selecting only binary metrics (e.g. ground_truth) without
        # --standardize — the default graded composite drops them, leaving nothing.
        if (not standardize and (long_df["kind"] == "binary").any()
                and not (long_df["kind"] == "quality").any()):
            raise ValueError(
                "The selected metric(s) are binary accuracy metrics (e.g. ground_truth), which "
                "the raw graded composite excludes. Drop --raw so they are standardized and "
                "pooled (z-scored to SD units)."
            )
        raise ValueError("Pilot needs >=2 models and at least one quality metric.")

    models = list(composite.columns)
    reference = reference or sorted(models)[0]
    if reference not in models:
        raise ValueError(f"--reference {reference!r} not among models: {', '.join(models)}")

    pairs = []
    for model in models:
        if model == reference:
            continue
        diff = (composite[model] - composite[reference]).dropna()
        sd = diff.std()
        mean_diff = float(diff.mean())
        if sd == 0:
            # Zero-variance paired differences: dz = mean/sd is 0/0 (no effect) only when the
            # mean is also zero. A constant *nonzero* difference is a perfectly consistent
            # effect (unbounded dz), which must not be reported as zero effect.
            if mean_diff == 0:
                dz = 0.0
            else:
                raise ValueError(
                    f"Cannot estimate dz for {model!r} vs {reference!r} from zero-variance paired "
                    f"differences with a nonzero mean ({mean_diff:g}): a perfectly consistent "
                    "difference implies an unbounded effect size. Add variation or exclude this metric."
                )
        else:
            dz = float(mean_diff / sd)
        pairs.append(PairPower(
            model=model, reference=reference, dz=dz, mean_diff=mean_diff,
            n_pilot=len(diff),
            n_needed=(n_for_effect(dz, alpha, power, alternative) if dz else None),
            power_at_pilot=achieved_power(dz, len(diff), alpha, alternative),
        ))
    # The metrics that actually entered the composite (graded-only when not standardized).
    metrics = sorted(frame.pooled_rows(long_df, standardize=standardize)["metric"].unique().tolist())
    return {"reference": reference, "alpha": alpha, "target_power": power,
            "alternative": alternative, "standardize": standardize,
            "metrics": metrics, "pairs": pairs}
