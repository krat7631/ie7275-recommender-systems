"""Collect every results/*.json into one comparison table.

    python -m recsys.compare            # every run, including the hyperparameter sweeps
    python -m recsys.compare --final    # only runs tagged final: one row per model, for the report

Prints the table and writes summary.csv. Warns if the runs were made on different data.
"""
import argparse
import json

import pandas as pd

from recsys.config import RESULTS_DIR

COLUMNS = [
    "recall@10", "ndcg@10", "mrr", "recall@20", "ndcg@20", "rmse",
    "coverage@10", "novelty@10", "gini@10",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="only runs tagged final")
    args = ap.parse_args()
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        r = json.loads(path.read_text())
        if args.final and r["tag"] != "final":
            continue
        row = {"run": r["run_name"], "data": r["data_version"]}
        row.update({m: r["test"][m] for m in COLUMNS})
        row.update(r["cost"])
        if r.get("cold_start"):  # cold-start dataset: recall@10 for new users by purchases revealed
            row.update({f"new_recall@10_n{n}": m["recall@10"] for n, m in r["cold_start"]["by_revealed"].items()})
        rows.append(row)
    if not rows:
        raise SystemExit(f"no {'final ' if args.final else ''}results in {RESULTS_DIR}")
    df = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False)
    if df["data"].nunique() > 1:
        print(f"WARNING: results use {df['data'].nunique()} different data versions: {sorted(df['data'].unique())}\n")
    df.to_csv(RESULTS_DIR / ("summary_final.csv" if args.final else "summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
