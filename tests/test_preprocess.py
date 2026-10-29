import json

import numpy as np
import pandas as pd
import pytest

from conftest import PLANTED, make_raw
from recsys.config import K_CORE
from recsys.data import build_histories, load, to_csr
from recsys.preprocess import build


def test_each_cleaning_rule_catches_exactly_the_planted_defects(processed):
    log = processed["meta"]["preprocess_log"]
    assert log["malformed_json"] == PLANTED["malformed_json"]
    assert log["missing_or_unparseable"] == PLANTED["missing"]
    assert log["rating_out_of_range"] == PLANTED["bad_rating"]
    assert log["timestamp_implausible"] == PLANTED["bad_timestamp"]
    assert log["duplicate_pairs_removed"] == PLANTED["duplicates"]


def test_duplicates_keep_the_earliest_interaction(processed):
    d = processed["data"]
    allrows = pd.concat([d.train, d.val, d.test])
    assert not allrows.duplicated(["user_idx", "item_idx"]).any()

    raw = []
    for line in processed["raw"][0].read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("user_id") and r.get("parent_asin") and r.get("timestamp") and r["timestamp"] > 10**9:
            raw.append((r["user_id"], r["parent_asin"], r["timestamp"]))
    earliest = pd.DataFrame(raw, columns=["user_id", "parent_asin", "ts"]).groupby(["user_id", "parent_asin"])["ts"].min()

    out = allrows.assign(user_id=d.users["user_id"].to_numpy()[allrows["user_idx"]],
                         parent_asin=d.items["parent_asin"].to_numpy()[allrows["item_idx"]])
    expected = earliest.loc[list(zip(out["user_id"], out["parent_asin"]))].to_numpy()
    assert (out["timestamp"].to_numpy() == expected).all()


def test_sparse_users_and_items_are_dropped(processed):
    d = processed["data"]
    assert not d.users["user_id"].str.startswith("SPARSE").any()
    assert not d.items["parent_asin"].str.startswith("SPARSE").any()


def test_five_core_holds_on_the_full_data(processed):
    d = processed["data"]
    full = pd.concat([d.train, d.val, d.test])
    assert full.groupby("user_idx").size().min() >= K_CORE
    assert full.groupby("item_idx").size().min() >= K_CORE


def test_split_is_chronological_one_row_per_user(processed):
    d = processed["data"]
    assert len(d.val) == len(d.test) == d.n_users
    last_train = d.train.groupby("user_idx")["timestamp"].max()
    assert (last_train.to_numpy() <= d.val["timestamp"].to_numpy()).all()
    assert (d.val["timestamp"].to_numpy() <= d.test["timestamp"].to_numpy()).all()
    assert (d.train["timestamp"].groupby(d.train["user_idx"]).apply(lambda s: s.is_monotonic_increasing)).all()


def test_no_target_leaks_into_history(processed):
    d = processed["data"]
    for u in range(0, d.n_users, 7):
        assert d.val["item_idx"][u] not in d.train_history[u]
        assert d.test["item_idx"][u] not in d.trainval_history[u]
        assert d.trainval_history[u][-1] == d.val["item_idx"][u]


def test_indices_are_contiguous_and_dtypes_match_the_contract(processed):
    d = processed["data"]
    assert sorted(d.users["user_idx"]) == list(range(d.n_users))
    assert sorted(d.items["item_idx"]) == list(range(d.n_items))
    assert d.train["item_idx"].dtype == np.int32 and d.train["rating"].dtype == np.float32
    assert d.train["timestamp"].dtype == np.int64


def test_item_metadata_handling(processed):
    d = processed["data"]
    assert "average_rating" not in d.items.columns and "rating_number" not in d.items.columns
    missing = d.items[~d.items["has_metadata"]]
    assert set(missing["parent_asin"]) == {"I0003", "I0004"}
    assert (missing["title"] == "").all()
    assert d.items["price"].dtype == np.float32
    assert (d.items.loc[d.items["has_metadata"], "title"] != "").all()


def test_item_features_are_complete_encoded_and_scaled(processed):
    d = processed["data"]
    it = d.items
    cols = ["price_missing", "price_log_z", "store_code", "main_category_code", "cat2_code", "cat3_code", "text"]
    assert not it[cols].isna().any().any()
    priced = it["price_missing"] == 0
    assert (priced == it["price"].notna()).all()
    assert (it.loc[~priced, "price_log_z"] == 0).all()  # missing price -> scaled mean, flagged by price_missing
    assert it.loc[priced, "price_log_z"].mean() == pytest.approx(0, abs=1e-4)
    assert it.loc[priced, "price_log_z"].std() == pytest.approx(1, abs=0.02)
    for col, n in d.meta["item_feature_cardinalities"].items():
        assert 0 <= it[col].min() and it[col].max() < n
    assert (it["cat3_code"] == 0).all()  # synthetic categories have only two levels
    assert (it.loc[it["has_metadata"], "cat2_code"] > 0).all()
    assert (it.loc[~it["has_metadata"], "cat2_code"] == 0).all()  # no metadata -> unknown
    assert (it.loc[~it["has_metadata"], "text"] == "").all()
    assert (it.loc[it["has_metadata"], "text"].str.startswith("Game ")).all()


def test_bins_cover_everyone(processed):
    d = processed["data"]
    assert set(d.users["activity_bin"]) <= {"cold", "warm", "heavy"}
    assert set(d.items["pop_bin"]) <= {"cold", "head", "mid", "tail"}
    assert (d.items.loc[d.items["pop_bin"] == "cold", "n_train"] == 0).all()
    assert (d.items["n_train"].to_numpy() == np.bincount(d.train["item_idx"], minlength=d.n_items)).all()


def test_build_is_deterministic(processed, tmp_path):
    reviews, meta = make_raw(tmp_path)
    again = build(reviews, meta, tmp_path / "again")
    assert again["data_version"] == processed["meta"]["data_version"]


def test_dev_sample_is_a_valid_smaller_dataset(processed, tmp_path):
    reviews, meta = processed["raw"]
    m = build(reviews, meta, tmp_path / "dev", max_users=100)
    dev = load(tmp_path / "dev")
    assert dev.n_users <= 100 and dev.n_users < processed["data"].n_users
    full = pd.concat([dev.train, dev.val, dev.test])
    assert full.groupby("user_idx").size().min() >= K_CORE and full.groupby("item_idx").size().min() >= K_CORE
    assert m["data_version"] != processed["meta"]["data_version"]


def test_helpers(processed):
    d = processed["data"]
    h = build_histories([d.train], d.n_users)
    assert len(h) == d.n_users and sum(len(x) for x in h) == len(d.train)
    m = to_csr(d.train, d.n_users, d.n_items)
    assert m.shape == (d.n_users, d.n_items) and m.nnz == len(d.train)


def test_load_refuses_tampered_data(processed, tmp_path):
    import shutil

    shutil.copytree(processed["dir"], tmp_path / "copy")
    train = pd.read_parquet(tmp_path / "copy" / "train.parquet")
    train.loc[0, "item_idx"] = (train.loc[0, "item_idx"] + 1) % processed["data"].n_items
    train.to_parquet(tmp_path / "copy" / "train.parquet", index=False)
    with pytest.raises(ValueError, match="do not match meta.json"):
        load(tmp_path / "copy")
