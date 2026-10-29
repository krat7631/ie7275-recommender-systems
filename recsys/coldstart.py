"""Derive the cold-start experiment dataset from the processed dataset.

    python -m recsys.coldstart --src data/processed --out data/processed_coldstart

A fixed random 10% of users become "new users": all of their training interactions are removed
from train.parquet, so no model can learn anything about them. val.parquet and test.parquet are
unchanged. The removed rows go to new_user_interactions.parquet, which the harness uses to reveal
the first n purchases of a new user (n = 0, 1, 2, 3) when it scores them. Item popularity counts
and bins are recomputed from the reduced training set.

Run once to build the folder; the models and the notebook only read it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import COLD_FRAC, COLD_REVEAL, SEED
from recsys.data import fingerprint, load
from recsys.preprocess import item_bins


def build_coldstart(src_dir, out_dir, frac: float = COLD_FRAC) -> dict:
    src = load(src_dir)
    rng = np.random.default_rng(SEED)
    new_users = np.sort(rng.choice(src.n_users, size=int(round(frac * src.n_users)), replace=False))
    is_new = np.zeros(src.n_users, dtype=bool)
    is_new[new_users] = True
    held_out = is_new[src.train["user_idx"].to_numpy()]
    train = src.train[~held_out].reset_index(drop=True)
    removed = src.train[held_out].reset_index(drop=True)

    items = src.items.copy()
    items["n_train"] = np.bincount(train["item_idx"], minlength=src.n_items).astype(np.int32)
    items["pop_bin"] = item_bins(items["n_train"].to_numpy())

    users = src.users.copy()
    users["n_train"] = np.bincount(train["user_idx"], minlength=src.n_users).astype(np.int32)
    users.loc[new_users, "activity_bin"] = "new"

    meta = dict(src.meta)
    meta.update(
        data_version=fingerprint(train, src.val, src.test),
        base_data_version=src.data_version,
        n_train=len(train),
        cold_start={
            "fraction": frac,
            "n_new_users": len(new_users),
            "reveal": list(COLD_REVEAL),
            "seed": SEED,
            "target": "the user's last interaction (test.parquet row)",
            "candidates": "all items except the user's other interactions, identical for every n",
            "removed_train_rows": len(removed),
            "items_with_no_train_interactions": int((items["n_train"] == 0).sum()),
            "items_with_no_train_interactions_in_base": int((src.items["n_train"] == 0).sum()),
        },
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    src.val.to_parquet(out / "val.parquet", index=False)
    src.test.to_parquet(out / "test.parquet", index=False)
    users.to_parquet(out / "users.parquet", index=False)
    items.to_parquet(out / "items.parquet", index=False)
    pd.DataFrame({"user_idx": new_users.astype(np.int32)}).to_parquet(out / "new_users.parquet", index=False)
    removed.to_parquet(out / "new_user_interactions.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(new_users)} new users, {len(removed)} training rows removed, "
          f"{len(train)} train rows left, version {meta['data_version']}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=COLD_FRAC)
    args = ap.parse_args()
    build_coldstart(args.src, args.out, args.frac)


if __name__ == "__main__":
    main()
