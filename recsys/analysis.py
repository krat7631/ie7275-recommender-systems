"""Statistics and trade-off figures built from the results files (runs tagged final).

Confidence intervals and paired tests come from the per-user ranks saved with every run (results/per_user/*.npz):
users are resampled with replacement, and ALL models see the same resampled users, so a difference between two models
is a paired difference. The p-value is two-sided: 2 * min(share of resamples with difference <= 0, share with >= 0).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import RESULTS_DIR, SEED


# Scoring variants of an already fitted final model, shown next to the final runs on the full data:
# als+history is the same trained ALS with every user's vector computed from the history the harness gives it.
VARIANT_LABELS = {"foldall": "history"}


def label(run: dict) -> str:
    return run["model"] if run["tag"] == "final" else f"{run['model']}+{VARIANT_LABELS[run['tag']]}"


def final_runs(results_dir=None, variants: bool = True) -> list[dict]:
    d = Path(results_dir) if results_dir else RESULTS_DIR
    runs = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    return [r for r in runs if r["tag"] == "final" or (variants and r["tag"] in VARIANT_LABELS)]


def per_user_values(ranks: np.ndarray, metric: str) -> np.ndarray:
    """The metric for every user, from the rank of their held-out item (1 = best)."""
    if metric == "mrr":
        return 1.0 / ranks
    name, k = re.fullmatch(r"(recall|ndcg)@(\d+)", metric).groups()
    k = int(k)
    if name == "recall":
        return (ranks <= k).astype(np.float64)
    return np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)


def bootstrap_means(arrays: dict[str, np.ndarray], n_boot: int = 2000, seed: int = SEED, chunk: int = 50) -> dict[str, np.ndarray]:
    """Mean of each model's per-user values on n_boot resamples of the users, the same resamples for every model."""
    n = len(next(iter(arrays.values())))
    rng = np.random.default_rng(seed)
    out = {k: np.empty(n_boot) for k in arrays}
    for s in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - s), n))
        for k, v in arrays.items():
            out[k][s : s + len(idx)] = v[idx].mean(axis=1)
    return out


def _load_ranks(results_dir, run_name: str, key: str = "rank_test") -> np.ndarray:
    return np.load(Path(results_dir) / "per_user" / f"{run_name}.npz")[key]


def comparison_table(results_dir=None, metrics=("recall@10", "ndcg@10", "mrr"), n_boot: int = 2000) -> pd.DataFrame:
    """One row per final run: each metric with its 95% bootstrap interval, ordered by NDCG@10."""
    d = Path(results_dir) if results_dir else RESULTS_DIR
    runs = final_runs(d)
    ranks = {label(r): _load_ranks(d, r["run_name"]) for r in runs}
    rows = {label(r): {"model": label(r)} for r in runs}
    for m in metrics:
        means = bootstrap_means({k: per_user_values(v, m) for k, v in ranks.items()}, n_boot)
        for k, boots in means.items():
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows[k].update({m: float(per_user_values(ranks[k], m).mean()), f"{m}_lo": float(lo), f"{m}_hi": float(hi)})
    return pd.DataFrame(rows.values()).sort_values(f"{metrics[1] if len(metrics) > 1 else metrics[0]}", ascending=False).reset_index(drop=True)


def paired_p_values(results_dir=None, metric: str = "ndcg@10", n_boot: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(differences, p-values): matrices over final runs, row model minus column model, two-sided paired bootstrap."""
    d = Path(results_dir) if results_dir else RESULTS_DIR
    runs = final_runs(d)
    arrays = {label(r): per_user_values(_load_ranks(d, r["run_name"]), metric) for r in runs}
    means = bootstrap_means(arrays, n_boot)
    names = sorted(arrays, key=lambda k: -arrays[k].mean())
    diff = pd.DataFrame(index=names, columns=names, dtype=float)
    pval = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            delta = means[a] - means[b]
            diff.loc[a, b] = arrays[a].mean() - arrays[b].mean()
            pval.loc[a, b] = 1.0 if a == b else min(1.0, 2 * min((delta <= 0).mean(), (delta >= 0).mean()))
    return diff, pval


def cost_table(results_dir=None) -> pd.DataFrame:
    rows = [{"model": label(r), "ndcg@10": r["test"]["ndcg@10"], "fit_seconds": r["cost"]["fit_seconds"],
             "score_ms_per_user": r["cost"]["score_ms_per_user"], "n_params": r["cost"]["n_params"],
             "coverage@10": r["test"]["coverage@10"], "novelty@10": r["test"]["novelty@10"],
             "avg_pop@10": r["test"]["avg_pop@10"], "gini@10": r["test"]["gini@10"], "recall@10": r["test"]["recall@10"]}
            for r in final_runs(results_dir)]
    return pd.DataFrame(rows).sort_values("ndcg@10", ascending=False).reset_index(drop=True)
