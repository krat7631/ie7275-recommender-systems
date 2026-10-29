"""Helpers for the notebooks: read tables and code straight from the results and model files.

    python3 -m recsys.report knn                  # the sweep table of one model, best on validation first
    python3 -m recsys.report svd --by val_rmse    # rating-prediction model: lowest validation RMSE first
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from recsys.config import RESULTS_DIR, ROOT

VARIANT_TAGS = {"foldin", "foldall"}


def sweep_table(model: str, results_dir=None, by: str = "val_ndcg@10") -> pd.DataFrame:
    """Every run of one model (the sweep), best first. `selected` marks the winner.

    Rank by validation NDCG@10 (default). For the rating-prediction model use by="val_rmse" (lower is better).
    """
    d = Path(results_dir) if results_dir else RESULTS_DIR
    rows = []
    for p in sorted(d.glob(f"{model}*.json")):
        r = json.loads(p.read_text())
        if r["model"] != model or r["tag"] in VARIANT_TAGS:
            continue  # scoring variants of an already fitted model (docs/variant.py) are not hyperparameter runs
        row = {"run": r["run_name"], **{f"cfg_{k}": v for k, v in r["config"].items()}}
        row.update({"val_ndcg@10": r["val"]["ndcg@10"], "val_rmse": r["val"]["rmse"], "test_recall@10": r["test"]["recall@10"],
                    "test_ndcg@10": r["test"]["ndcg@10"], "test_mrr": r["test"]["mrr"], "test_rmse": r["test"]["rmse"],
                    "fit_seconds": r["cost"]["fit_seconds"], "score_ms_per_user": r["cost"]["score_ms_per_user"]})
        rows.append(row)
    if not rows:
        raise ValueError(f"no runs of {model!r} in {d}")
    lower_is_better = "rmse" in by
    df = pd.DataFrame(rows).sort_values(by, ascending=lower_is_better).reset_index(drop=True)
    best = df[by].min() if lower_is_better else df[by].max()
    df["selected"] = df[by] == best  # ties (e.g. the final re-run of the winner) are all marked
    return df


def show_source(model: str) -> None:
    """Print a model file so the notebook itself contains the code."""
    print((ROOT / "recsys" / "models" / f"{model}.py").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--by", default="val_ndcg@10", help="validation column to rank by, e.g. val_rmse")
    args = ap.parse_args()
    df = sweep_table(args.model, by=args.by)
    cols = ["run"] + [c for c in df.columns if c.startswith("cfg_")] + ["val_ndcg@10"]
    if df["val_rmse"].notna().any():
        cols.append("val_rmse")
    cols += ["test_recall@10", "fit_seconds", "selected"]
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
