"""Write docs/experiment_log.md: every run of every model, so the hyperparameter selection can be audited and cited.

    python3 docs/make_experiment_log.py

Each table lists all runs of one model on the full data, best on the validation split first (lowest validation RMSE for
svd). Selection never used the test columns; they are shown only to report the chosen run. Training times of sweep runs
were measured while other jobs shared the machine; the final runs (tag final) were run alone and are the ones in the cost table.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from recsys.report import sweep_table  # noqa: E402

MODELS = ["popularity", "knn", "als", "svd", "bpr", "ncf", "sasrec", "content"]


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and pd.isna(v)) else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def main() -> None:
    out = ["# Experiment log", "", __doc__.split("\n\n", 1)[1].strip(), ""]
    for m in MODELS:
        by = "val_rmse" if m == "svd" else "val_ndcg@10"
        t = sweep_table(m, by=by)
        cfg_cols = [c for c in t.columns if c.startswith("cfg_")]
        # show only the hyperparameters that differ between runs
        varying = [c for c in cfg_cols if t[c].astype(str).nunique() > 1]
        t["settings"] = t[varying].apply(lambda r: ", ".join(f"{c[4:]}={v}" for c, v in r.items()), axis=1) if varying else "(defaults)"
        keep = ["run", "settings", "val_ndcg@10"] + (["val_rmse"] if m in ("svd", "popularity") else []) + ["test_recall@10", "test_ndcg@10", "fit_seconds", "selected"]
        out += [f"## {m}", "", f"{len(t)} runs, ranked by {'validation RMSE (lower is better)' if m == 'svd' else 'validation NDCG@10'}.", "", md_table(t[keep]), ""]
    (ROOT / "docs" / "experiment_log.md").write_text("\n".join(out))
    print("wrote docs/experiment_log.md")


if __name__ == "__main__":
    main()
