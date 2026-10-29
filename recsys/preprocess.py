"""Raw Amazon Reviews 2023 (McAuley Lab) -> the processed splits every model trains on.

    python -m recsys.preprocess --reviews data/raw/Video_Games.jsonl \
        --meta data/raw/meta_Video_Games.jsonl --out data/processed
    python -m recsys.preprocess ... --out data/processed_dev --max-users 10000   # small dev set

Run once to build data/processed/; the models and the notebook only read it and never re-derive it.

Steps, each counted in meta.json["preprocess_log"]:
  1. parse JSONL, drop malformed lines
  2. drop rows with a missing/unparseable user, item, rating or timestamp
  3. drop ratings outside 1-5 and timestamps outside 1990-2023
  4. collapse repeated (user, item) pairs, keeping the earliest interaction
  5. iterative 5-core filter (every user and item keeps >= 5 interactions)
  6. leave-one-out temporal split per user: last = test, second-to-last = val, rest = train
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import K_CORE, SEED
from recsys.data import INTERACTION_DTYPES, fingerprint

TS_MIN = 631_152_000_000  # 1990-01-01 UTC, in ms
TS_MAX = 1_704_067_200_000  # 2024-01-01 UTC, in ms (dataset ends in 2023)


def read_reviews(path, log: dict) -> pd.DataFrame:
    users, items, ratings, stamps = [], [], [], []
    n_lines = bad = 0
    with open(path, "rb") as f:
        for line in f:
            n_lines += 1
            try:
                d = json.loads(line)
            except ValueError:  # includes JSONDecodeError and UnicodeDecodeError
                bad += 1
                continue
            if not isinstance(d, dict):
                bad += 1
                continue
            users.append(d.get("user_id"))
            items.append(d.get("parent_asin"))
            ratings.append(d.get("rating"))
            stamps.append(d.get("timestamp"))
            if n_lines % 1_000_000 == 0:
                print(f"  read {n_lines:,} lines")
    log["raw_lines"] = n_lines
    log["malformed_json"] = bad
    return pd.DataFrame({"user_id": users, "parent_asin": items, "rating": ratings, "timestamp": stamps})


def clean(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    missing = (
        df["user_id"].isna() | df["parent_asin"].isna() | df["rating"].isna() | df["timestamp"].isna()
        | (df["user_id"] == "") | (df["parent_asin"] == "")
    )
    log["missing_or_unparseable"] = int(missing.sum())
    df = df[~missing]

    bad_rating = (df["rating"] < 1) | (df["rating"] > 5)
    log["rating_out_of_range"] = int(bad_rating.sum())
    df = df[~bad_rating]

    bad_ts = (df["timestamp"] < TS_MIN) | (df["timestamp"] > TS_MAX)
    log["timestamp_implausible"] = int(bad_ts.sum())
    df = df[~bad_ts]
    log["after_cleaning_rows"] = len(df)
    return df.reset_index(drop=True)


def dedupe(u, i, r, ts, row, log: dict):
    """One row per (user, item): the earliest interaction, ties broken by file order."""
    o = np.lexsort((row, ts, i, u))
    u, i, r, ts, row = u[o], i[o], r[o], ts[o], row[o]
    first = np.r_[True, (u[1:] != u[:-1]) | (i[1:] != i[:-1])]
    log["duplicate_pairs_removed"] = int((~first).sum())
    return u[first], i[first], r[first], ts[first], row[first]


def k_core(u, i, r, ts, row, k: int, log: dict, prefix: str = "kcore"):
    rounds = 0
    while True:
        uc = np.bincount(u)
        ic = np.bincount(i)
        keep = (uc[u] >= k) & (ic[i] >= k)
        if keep.all():
            break
        u, i, r, ts, row = u[keep], i[keep], r[keep], ts[keep], row[keep]
        rounds += 1
    log[f"{prefix}_rounds"] = rounds
    return u, i, r, ts, row


def compact(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Re-index to 0..n-1 keeping sort order. Returns (new codes, original codes per new index)."""
    originals, new = np.unique(codes, return_inverse=True)
    return new.astype(np.int32), originals


def leave_one_out(u, i, r, ts, row, n_users: int):
    order = np.lexsort((row, ts, u))  # by user, then time, then file order for equal timestamps
    u, i, r, ts = u[order], i[order], r[order], ts[order]
    counts = np.bincount(u, minlength=n_users)
    if counts.min() < 3:
        raise ValueError("every user needs at least 3 interactions for a train/val/test split")
    pos = np.arange(len(u)) - (np.cumsum(counts) - counts)[u]
    n = counts[u]
    is_test, is_val, is_train = pos == n - 1, pos == n - 2, pos < n - 2

    def frame(mask):
        return pd.DataFrame(
            {"user_idx": u[mask], "item_idx": i[mask], "rating": r[mask], "timestamp": ts[mask]}
        ).astype(INTERACTION_DTYPES)

    tied = int((ts[is_test] == ts[is_val]).sum())
    return frame(is_train), frame(is_val), frame(is_test), tied


def user_bins(n_train: np.ndarray) -> tuple[np.ndarray, dict]:
    """Activity buckets named as in the proposal. 'cold' is the least active third of users (a few
    interactions), not a user with no history; users with no history are the cold-start experiment."""
    q1, q2 = np.quantile(n_train, [1 / 3, 2 / 3])
    bins = np.where(n_train <= q1, "cold", np.where(n_train > q2, "heavy", "warm"))
    return bins, {"cold": f"n_train <= {q1:g}", "warm": f"{q1:g} < n_train <= {q2:g}", "heavy": f"n_train > {q2:g}"}


def item_bins(n_train: np.ndarray) -> np.ndarray:
    """cold = no train interactions; of the rest by popularity: head = top 20%, mid = next 30%, tail = rest."""
    bins = np.full(len(n_train), "cold", dtype=object)
    seen = np.flatnonzero(n_train > 0)
    ranked = seen[np.lexsort((seen, -n_train[seen]))]  # most popular first, ties by item_idx
    pct = np.arange(len(ranked)) / max(len(ranked), 1)
    bins[ranked] = np.where(pct < 0.2, "head", np.where(pct < 0.5, "mid", "tail"))
    return bins


def _text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(str(x).strip() for x in v if x)
    return str(v).strip()


def _price(v) -> float:
    try:
        return float(str(v).replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return float("nan")


def _codes(values: pd.Series, min_count: int) -> tuple[pd.Series, int]:
    """Integer-encode a categorical. Code 0 means unknown or rare (fewer than min_count items)."""
    counts = values.value_counts()
    kept = counts[(counts >= min_count) & (counts.index != "")].index
    mapping = {v: n + 1 for n, v in enumerate(kept)}
    return values.map(mapping).fillna(0).astype("int32"), len(mapping) + 1


def add_item_features(items: pd.DataFrame, log: dict) -> dict:
    """Missing-value handling, categorical encoding and scaling for the item attributes.

    These are item-level attributes with no interaction information, so nothing here leaks.
      price      : missing -> price_missing flag = 1 and price_log_z = 0 (the mean after scaling);
                   otherwise log1p(price) standardised to mean 0, std 1 over the priced items
      store, main_category, categories level 2 and 3 : integer codes, 0 = unknown or rare
      text       : title + features + description, the single text field for embedding models
    """
    parts = items["categories"].str.split(" > ")
    items["text"] = items[["title", "features", "description"]].apply(lambda r: " ".join(x for x in r if x), axis=1)

    items["price_missing"] = items["price"].isna().astype("int8")
    lp = np.log1p(items["price"].astype("float64"))
    items["price_log_z"] = ((lp - lp.mean()) / lp.std()).fillna(0.0).astype("float32")

    card = {}
    for col, src, min_count in (
        ("store_code", items["store"], 10),
        ("main_category_code", items["main_category"], 5),
        ("cat2_code", parts.str[1].fillna(""), 5),
        ("cat3_code", parts.str[2].fillna(""), 5),
    ):
        items[col], card[col] = _codes(src, min_count)

    log["items_missing_price"] = int(items["price_missing"].sum())
    log["items_empty_description"] = int((items["description"] == "").sum())
    log["items_empty_features"] = int((items["features"] == "").sum())
    log["items_missing_store"] = int((items["store"] == "").sum())
    log["items_missing_category"] = int((items["categories"] == "").sum())
    return card


def read_item_meta(path, wanted: set[str]) -> pd.DataFrame:
    """Text and attribute fields for the retained items.

    average_rating and rating_number are deliberately left out: they are computed over all reviews,
    including the validation and test interactions, so using them as features would leak the answer.
    """
    rows = []
    with open(path, "rb") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            asin = d.get("parent_asin")
            if asin not in wanted:
                continue
            cats = d.get("categories")
            rows.append(
                {
                    "parent_asin": asin,
                    "title": _text(d.get("title")),
                    "store": _text(d.get("store")),
                    "main_category": _text(d.get("main_category")),
                    "categories": " > ".join(str(c) for c in cats) if isinstance(cats, list) else "",
                    "features": _text(d.get("features")),
                    "description": _text(d.get("description")),
                    "price": _price(d.get("price")),
                }
            )
    return pd.DataFrame(rows).drop_duplicates("parent_asin")


def build(reviews_path, meta_path, out_dir, max_users: int | None = None, k: int = K_CORE) -> dict:
    t0 = time.time()
    log: dict = {}
    print("reading reviews")
    df = clean(read_reviews(reviews_path, log), log)

    u, user_ids = pd.factorize(df["user_id"], sort=True)
    i, item_ids = pd.factorize(df["parent_asin"], sort=True)
    r = df["rating"].to_numpy(np.float32)
    ts = df["timestamp"].to_numpy(np.int64)
    row = np.arange(len(df))
    del df

    u, i, r, ts, row = dedupe(u, i, r, ts, row, log)
    log["after_dedupe_rows"] = len(u)
    log["after_dedupe_users"] = int(len(np.unique(u)))
    log["after_dedupe_items"] = int(len(np.unique(i)))

    u, i, r, ts, row = k_core(u, i, r, ts, row, k, log)
    log["after_kcore_rows"] = len(u)
    log["after_kcore_users"] = int(len(np.unique(u)))
    log["after_kcore_items"] = int(len(np.unique(i)))

    if max_users is not None:
        rng = np.random.default_rng(SEED)
        chosen = rng.choice(np.unique(u), size=min(max_users, len(np.unique(u))), replace=False)
        keep = np.isin(u, chosen)
        u, i, r, ts, row = u[keep], i[keep], r[keep], ts[keep], row[keep]
        u, i, r, ts, row = k_core(u, i, r, ts, row, k, log, prefix="dev_kcore")
        log["dev_sample_users_requested"] = max_users

    u, kept_users = compact(u)
    i, kept_items = compact(i)
    n_users, n_items = len(kept_users), len(kept_items)

    train, val, test, tied = leave_one_out(u, i, r, ts, row, n_users)
    log["users_with_tied_last_two_timestamps"] = tied

    n_train_user = np.bincount(train["user_idx"], minlength=n_users)
    n_train_item = np.bincount(train["item_idx"], minlength=n_items)
    ubins, ubin_rules = user_bins(n_train_user)
    ibins = item_bins(n_train_item)
    log["test_targets_on_items_with_no_train_interactions"] = int((n_train_item[test["item_idx"]] == 0).sum())
    log["val_targets_on_items_with_no_train_interactions"] = int((n_train_item[val["item_idx"]] == 0).sum())

    print("reading item metadata")
    asins = item_ids[kept_items]
    meta = read_item_meta(meta_path, set(asins))
    items = pd.DataFrame({"item_idx": np.arange(n_items, dtype=np.int32), "parent_asin": asins})
    items = items.merge(meta, on="parent_asin", how="left")
    items["has_metadata"] = items["title"].notna()
    for col in ("title", "store", "main_category", "categories", "features", "description"):
        items[col] = items[col].fillna("")
    items["price"] = items["price"].astype("float32")
    items["n_train"] = n_train_item.astype(np.int32)
    items["pop_bin"] = ibins
    log["items_without_metadata"] = int((~items["has_metadata"]).sum())
    feature_cardinalities = add_item_features(items, log)

    users = pd.DataFrame(
        {
            "user_idx": np.arange(n_users, dtype=np.int32),
            "user_id": user_ids[kept_users],
            "n_train": n_train_user.astype(np.int32),
            "activity_bin": ubins,
        }
    )

    data_version = fingerprint(train, val, test)

    meta_json = {
        "dataset": "Amazon Reviews 2023, Video Games (McAuley Lab, UCSD)",
        "data_version": data_version,
        "k_core": k,
        "seed": SEED,
        "max_users": max_users,
        "n_users": n_users,
        "n_items": n_items,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "user_activity_bins": ubin_rules,
        "item_pop_bins": "cold: 0 train interactions; head: top 20% of the rest by train count; mid: next 30%; tail: bottom 50%",
        "item_feature_cardinalities": feature_cardinalities,
        "dtypes": INTERACTION_DTYPES,
        "created": datetime.now().isoformat(timespec="seconds"),
        "preprocess_log": log,
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    val.to_parquet(out / "val.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    users.to_parquet(out / "users.parquet", index=False)
    items.to_parquet(out / "items.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(meta_json, indent=2))
    print(f"done in {time.time() - t0:.0f}s: {n_users} users, {n_items} items, "
          f"{len(train)}/{len(val)}/{len(test)} train/val/test, version {data_version}")
    return meta_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-users", type=int, default=None, help="random user sample, for a small dev dataset")
    args = ap.parse_args()
    build(args.reviews, args.meta, args.out, args.max_users)


if __name__ == "__main__":
    main()
