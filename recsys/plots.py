"""Figures built from results files."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from recsys.config import COLD_REVEAL, RESULTS_DIR


SLICE_ORDER = {"item_bin": ["head", "mid", "tail", "cold"], "user_bin": ["cold", "warm", "heavy"]}


def slice_heatmap(results_dir=None, kind: str = "item_bin", metric: str = "recall@10", ax=None):
    """One row per model (runs tagged final), one column per bin, coloured by the metric on the test split.

    kind="item_bin": how popular the target item is (head / mid / tail / cold = never seen in training).
    kind="user_bin": how active the user is (cold / warm / heavy).
    """
    d = Path(results_dir) if results_dir else RESULTS_DIR
    runs = [r for r in (json.loads(p.read_text()) for p in sorted(d.glob("*.json"))) if r["tag"] in ("final", "foldall")]
    if not runs:
        raise ValueError(f"no runs tagged final in {d}")
    for r in runs:
        r["model"] = r["model"] if r["tag"] == "final" else f"{r['model']}+history"
    bins = SLICE_ORDER[kind]
    cell = lambda r, b: r["test_slices"][kind].get(b, {}).get(metric, float("nan"))  # noqa: E731
    data = np.array([[cell(r, b) for b in bins] for r in runs], dtype=float)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 0.6 * len(runs) + 2.2))
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f"{b}\n(n={runs[0]['test_slices'][kind].get(b, {}).get('n', 0):,})" for b in bins])
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([r["model"] for r in runs])
    top = np.nanmax(data) if not np.isnan(data).all() else 1.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if not np.isnan(v) and v > 0.6 * top else "black")
    ax.set_title(f"{metric} by {'target item popularity' if kind == 'item_bin' else 'user activity'}")
    ax.grid(False)
    plt.colorbar(im, ax=ax, shrink=0.8)
    if fig is not None:
        fig.tight_layout()
    return ax


def cold_start_curve(results_dir=None, metric: str = "recall@10", ax=None):
    """How a new user's recommendations improve as their first purchases are revealed.

    One line per run that has a cold_start section (run the models with
    RECSYS_DATA_DIR=data/processed_coldstart first). The diamond at the right is the same model's
    score for ordinary users with their full history, from the same dataset. Popularity is the floor.
    """
    d = Path(results_dir) if results_dir else RESULTS_DIR
    runs = [r for r in (json.loads(p.read_text()) for p in sorted(d.glob("*.json"))) if r.get("cold_start")]
    if not runs:
        raise ValueError(f"no cold-start results in {d}")
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.4))

    xs = list(range(len(COLD_REVEAL)))
    warm_x = len(COLD_REVEAL) + 0.6
    for r in runs:
        ys = [r["cold_start"]["by_revealed"][str(n)][metric] for n in COLD_REVEAL]
        style = {"color": _color(r["model"]), "ls": "--" if r["model"] == "popularity" else (":" if r["tag"] != "final" else "-")}
        (line,) = ax.plot(xs, ys, marker="o", label=r["run_name"], **style)
        ax.scatter([warm_x], [r["test"][metric]], marker="D", color=line.get_color(), zorder=3)

    ax.set_xticks(xs + [warm_x])
    ax.set_xticklabels([str(n) for n in COLD_REVEAL] + ["ordinary users,\nfull history"])
    ax.set(xlabel="purchases revealed for the new user", ylabel=metric,
           title=f"Cold-start users: {metric} vs purchases revealed (n = {runs[0]['cold_start']['n_new_users']:,} new users)")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    return ax


# ----------------------------------------------------------------------------- evaluation analysis figures
MODEL_COLORS = {
    "popularity": "#8a8f98", "knn": "#2a6fb0", "als": "#d9822b", "bpr": "#2e9e5b",
    "svd": "#7a5195", "ncf": "#c0392b", "sasrec": "#0e7c86", "content": "#a67c00",
}


def _color(model: str) -> str:
    return MODEL_COLORS.get(model.split("+")[0], "#444444")  # a scoring variant (als+history) keeps its model's colour


def comparison_chart(table, metric: str = "ndcg@10", ax=None):
    """Horizontal bars of a metric for every final run with its 95% bootstrap interval (table from analysis.comparison_table)."""
    t = table.sort_values(metric)
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 0.5 * len(t) + 1.6))
    err = np.array([t[metric] - t[f"{metric}_lo"], t[f"{metric}_hi"] - t[metric]])
    ax.barh(t["model"], t[metric], xerr=err, color=[_color(m) for m in t["model"]], ecolor="#222222", capsize=3)
    for y, (v, m) in enumerate(zip(t[metric], t["model"])):
        ax.text(t[f"{metric}_hi"].iloc[y] + t[f"{metric}_hi"].max() * 0.012, y, f"{v:.4f}", va="center", fontsize=9)
    ax.set(xlabel=f"{metric} on the test split (bars: 95% bootstrap interval over users)", title=f"Model comparison, {metric}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    return ax


def cost_scatter(cost, metric: str = "ndcg@10"):
    """Accuracy against training time and against scoring latency (log axes): the accuracy-versus-scalability trade-off."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, label in ((axes[0], "fit_seconds", "training time (seconds, log scale)"),
                           (axes[1], "score_ms_per_user", "scoring time (ms per user for all items, log scale)")):
        for _, r in cost.iterrows():
            x = max(r[col], 1e-3)
            ax.scatter(x, r[metric], s=70, color=_color(r["model"]), zorder=3)
            ax.annotate(r["model"], (x, r[metric]), textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set(xscale="log", xlabel=label, ylabel=f"{metric} on the test split")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Accuracy versus cost")
    fig.tight_layout()
    return fig


def tradeoff_scatter(cost):
    """Personalization versus popularity: accuracy against catalog coverage and against the popularity of what is recommended."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, label, log in ((axes[0], "coverage@10", "catalog coverage@10 (share of items ever recommended)", False),
                                (axes[1], "avg_pop@10", "average training popularity of recommended items (log scale)", True)):
        for _, r in cost.iterrows():
            ax.scatter(r[col], r["recall@10"], s=70, color=_color(r["model"]), zorder=3)
            ax.annotate(r["model"], (r[col], r["recall@10"]), textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set(xlabel=label, ylabel="recall@10 on the test split", xscale="log" if log else "linear")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Personalization versus popularity")
    fig.tight_layout()
    return fig
