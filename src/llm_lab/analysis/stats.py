"""Statistical tests for the model comparison, ported from the source analysis.

All tests treat the comparison as repeated-measures / paired on the record (prompt):
- Friedman + Kendall's W  : nonparametric omnibus across >=3 models.
- pairwise Wilcoxon (Holm): nonparametric post-hoc / the 2-model case.
- mixed-effects model     : score ~ C(model) [+ C(metric)], random intercept per prompt.
- paired Cohen's d        : effect size, mean(diff) / SD(diff).
- TOST                    : affirmative equivalence within a margin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def holm_adjust(pvals) -> np.ndarray:
    """Holm step-down adjustment; adjusted p-values in the input order."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        p = pvals[idx]
        if np.isnan(p):
            adjusted[idx] = np.nan
            continue
        val = min(1.0, p * (m - rank))
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted


def interpret_w(w: float) -> str:
    return "negligible" if w < 0.1 else "small" if w < 0.3 else "moderate+"


def interpret_d(d: float) -> str:
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"


@dataclass
class FriedmanResult:
    chi2: float
    p: float
    w: float           # Kendall's W (effect size)
    n: int             # prompts
    k: int             # models
    significant: bool


def friedman(matrix: pd.DataFrame, alpha: float = 0.05) -> FriedmanResult:
    """Friedman chi-square + Kendall's W on a complete-case (prompts x models) matrix."""
    cols = [matrix[c].values for c in matrix.columns]
    chi2, p = stats.friedmanchisquare(*cols)
    n, k = matrix.shape
    w = chi2 / (n * (k - 1))
    return FriedmanResult(float(chi2), float(p), float(w), int(n), int(k), bool(p < alpha))


def pairwise_wilcoxon(matrix: pd.DataFrame, alpha: float = 0.05) -> list[dict]:
    """Pairwise Wilcoxon signed-rank, Holm-corrected. Higher-mean model reported first."""
    cols = list(matrix.columns)
    pairs = list(combinations(cols, 2))
    raw = []
    for a, b in pairs:
        try:
            _, p = stats.wilcoxon(matrix[a], matrix[b])
        except ValueError:  # all differences zero
            p = 1.0
        raw.append(p)
    adj = holm_adjust(raw)
    rows = []
    for (a, b), rp, ap in zip(pairs, raw, adj):
        hi, lo = (a, b) if matrix[a].mean() >= matrix[b].mean() else (b, a)
        rows.append({"a": a, "b": b, "raw_p": float(rp), "holm_p": float(ap),
                     "significant": bool(ap < alpha), "higher": hi, "lower": lo})
    return sorted(rows, key=lambda r: r["holm_p"])


def cohens_d_pairwise(matrix: pd.DataFrame) -> list[dict]:
    """Paired Cohen's d for every model pair: mean(diff) / SD(diff), with a 95% CI."""
    cols = list(matrix.columns)
    rows = []
    for a, b in combinations(cols, 2):
        diff = (matrix[a] - matrix[b]).dropna()
        sd = diff.std()
        d = float(diff.mean() / sd) if sd else 0.0
        n = len(diff)
        se = sd / np.sqrt(n) if n else float("nan")
        rows.append({"a": a, "b": b, "mean_diff": float(diff.mean()),
                     "cohens_d": d, "ci_low": float(diff.mean() - 1.96 * se),
                     "ci_high": float(diff.mean() + 1.96 * se), "magnitude": interpret_d(d)})
    return rows


def tost_vs_reference(matrix: pd.DataFrame, reference: str, margin: float, alpha: float = 0.05) -> list[dict]:
    """TOST equivalence of each model vs the reference, paired on prompt."""
    from statsmodels.stats.weightstats import DescrStatsW

    rows = []
    for model in matrix.columns:
        if model == reference:
            continue
        diff = (matrix[model] - matrix[reference]).dropna()
        w = DescrStatsW(diff)
        p, _, _ = w.ttost_mean(-margin, margin)
        lo, hi = w.tconfint_mean(alpha=2 * alpha)  # 90% CI matches the alpha=0.05 TOST decision
        rows.append({"model": model, "mean_diff": float(diff.mean()), "ci_low": float(lo),
                     "ci_high": float(hi), "tost_p": float(p), "equivalent": bool(p < alpha)})
    return rows


def skewness(matrix: pd.DataFrame) -> list[dict]:
    """Skewness of each model's per-prompt score (negative = left/ceiling skew)."""
    return [{"model": c, "skew": float(stats.skew(matrix[c].dropna()))} for c in matrix.columns]


def bootstrap_ci_vs_reference(matrix: pd.DataFrame, reference: str, *, n_boot: int = 10000,
                              seed: int = 0) -> list[dict]:
    """Bootstrap 95% CI of the mean paired difference vs the reference (distribution-free)."""
    rng = np.random.default_rng(seed)
    rows = []
    for model in matrix.columns:
        if model == reference:
            continue
        diff = (matrix[model] - matrix[reference]).dropna().to_numpy()
        if len(diff) == 0:
            continue
        boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"model": model, "mean_diff": float(diff.mean()),
                     "ci_low": float(lo), "ci_high": float(hi)})
    return rows


@dataclass
class MixedModelResult:
    effects: list[dict] = field(default_factory=list)   # per model vs reference
    reference: str = ""
    formula: str = ""
    error: str | None = None
    summary_text: str = ""


def mixed_model(long_df: pd.DataFrame, reference: str, alpha: float = 0.05,
                with_metric: bool = False) -> MixedModelResult:
    """Fit score ~ C(model) [+ C(metric)] with a random intercept per prompt.

    Returns each model's coefficient (score difference vs the reference), CI, p-value,
    and whether the difference is statistically real. Fit failures are captured, not raised.
    """
    import warnings

    import statsmodels.formula.api as smf

    df = long_df.dropna(subset=["score"]).reset_index(drop=True)
    models = sorted(df["model"].unique())
    cats = [reference] + [m for m in models if m != reference]
    df = df.assign(model=pd.Categorical(df["model"], categories=cats))
    # Only add the metric term when there really are >=2 metrics; a single-level
    # C(metric) is collinear with the intercept and makes the fit singular.
    with_metric = with_metric and df["metric"].nunique() >= 2
    formula = "score ~ C(model) + C(metric)" if with_metric else "score ~ C(model)"

    try:
        with warnings.catch_warnings():  # RE-covariance-singular / convergence noise
            warnings.simplefilter("ignore")
            result = smf.mixedlm(formula, data=df, groups=df["prompt_id"], re_formula="1").fit(method="lbfgs")
    except Exception as exc:  # convergence/singularity/etc — report, don't crash the run
        return MixedModelResult(reference=reference, formula=formula, error=f"{type(exc).__name__}: {exc}")

    ci = result.conf_int()
    effects = []
    for model in models:
        if model == reference:
            continue
        term = f"C(model)[T.{model}]"
        if term not in result.params:
            continue
        lo, hi = ci.loc[term].tolist()
        p = float(result.pvalues[term])
        effects.append({"model": model, "coef": float(result.params[term]), "p": p,
                        "ci_low": float(lo), "ci_high": float(hi), "real": bool(p < alpha)})
    effects.sort(key=lambda e: e["coef"], reverse=True)
    return MixedModelResult(effects=effects, reference=reference, formula=formula,
                            summary_text=str(result.summary()))
