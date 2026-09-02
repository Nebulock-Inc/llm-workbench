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
             power: float = 0.8, alternative: str = "two-sided") -> dict:
    """Estimate required N from a pilot run's observed effect sizes on the quality composite.

    For each model vs the reference: the observed paired dz, prompts needed to detect
    that effect at the target power, and the power actually achieved at the pilot's N.
    """
    from llm_lab.analysis import frame

    long_df = frame.load_long(results_path)
    composite = frame.composite_quality(long_df)
    if composite is None or composite.shape[1] < 2:
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
        dz = float(diff.mean() / sd) if sd else 0.0
        pairs.append(PairPower(
            model=model, reference=reference, dz=dz, mean_diff=float(diff.mean()),
            n_pilot=len(diff),
            n_needed=(n_for_effect(dz, alpha, power, alternative) if dz else None),
            power_at_pilot=achieved_power(dz, len(diff), alpha, alternative),
        ))
    return {"reference": reference, "alpha": alpha, "target_power": power,
            "alternative": alternative, "pairs": pairs}
