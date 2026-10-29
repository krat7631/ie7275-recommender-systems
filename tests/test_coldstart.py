import json
import shutil

import matplotlib
import numpy as np
import pandas as pd
import pytest

from recsys import harness
from recsys.config import COLD_REVEAL
from recsys.data import load
from recsys.models.base import Recommender
from recsys.models.popularity import Popularity
from recsys.plots import cold_start_curve

matplotlib.use("Agg")


class TargetGate(Recommender):
    """Knows the answer, but only reveals it when it was shown at least `need` purchases."""

    name = "target_gate"

    def __init__(self, need=2, **kw):
        super().__init__(need=need, **kw)
        self.need = need

    def fit(self, data):
        self.test_item, self.n_items = data.test["item_idx"].to_numpy(), data.n_items

    def score(self, user_idxs, histories):
        s = np.zeros((len(user_idxs), self.n_items), dtype=np.float32)
        for row, (u, h) in enumerate(zip(user_idxs, histories)):
            if len(h) >= self.need:
                s[row, self.test_item[u]] = 1.0
        return s


class ValOracle(Recommender):
    name = "val_oracle"

    def fit(self, data):
        self.val_item, self.n_items = data.val["item_idx"].to_numpy(), data.n_items

    def score(self, user_idxs, histories):
        s = np.zeros((len(user_idxs), self.n_items), dtype=np.float32)
        s[np.arange(len(user_idxs)), self.val_item[user_idxs]] = 1.0
        return s


class Recorder(Recommender):
    name = "recorder"

    def fit(self, data):
        self.n_items, self.seen = data.n_items, []

    def score(self, user_idxs, histories):
        self.seen += list(zip(user_idxs, histories))
        return np.zeros((len(user_idxs), self.n_items), dtype=np.float32)


def test_new_users_are_a_fixed_tenth_with_no_training_rows(cold):
    d, base = cold["data"], cold["base"]
    assert abs(len(d.new_users) - round(0.1 * base.n_users)) <= 1
    assert (np.diff(d.new_users) > 0).all()
    assert not d.train["user_idx"].isin(d.new_users).any()
    assert len(d.train) + len(d.new_interactions) == len(base.train)
    assert d.val.equals(base.val) and d.test.equals(base.test)


def test_cold_build_is_deterministic(cold, processed, tmp_path):
    from recsys.coldstart import build_coldstart

    again = build_coldstart(processed["dir"], tmp_path / "again")
    assert again["data_version"] == cold["meta"]["data_version"]
    assert cold["meta"]["data_version"] != processed["meta"]["data_version"]
    assert cold["meta"]["base_data_version"] == processed["meta"]["data_version"]


def test_new_user_sequences_are_the_original_histories(cold):
    d, base = cold["data"], cold["base"]
    for u in d.new_users:
        assert np.array_equal(d.new_user_history[u], base.trainval_history[u])
        assert base.test["item_idx"][u] not in d.new_user_history[u]
    warm = np.setdiff1d(np.arange(d.n_users), d.new_users)[:5]
    assert all(len(d.new_user_history[u]) == 0 for u in warm)


def test_counts_and_bins_are_recomputed_for_the_reduced_training_set(cold):
    d, base = cold["data"], cold["base"]
    assert (d.items["n_train"].to_numpy() == np.bincount(d.train["item_idx"], minlength=d.n_items)).all()
    assert (d.users.loc[d.new_users, "n_train"] == 0).all()
    assert (d.users.loc[d.new_users, "activity_bin"] == "new").all()
    warm = d.eval_users
    assert (d.users.loc[warm, "activity_bin"].to_numpy() == base.users.loc[warm, "activity_bin"].to_numpy()).all()
    assert (d.items.loc[d.items["pop_bin"] == "cold", "n_train"] == 0).all()


def test_standard_evaluation_skips_new_users(cold):
    d = cold["data"]
    m = Popularity()
    m.fit(d)
    out = harness.run_split(m, d, "test")
    assert np.array_equal(out.users, d.eval_users) and not np.isin(out.users, d.new_users).any()


def test_model_is_shown_exactly_the_first_n_purchases(cold):
    d = cold["data"]
    for n in COLD_REVEAL:
        rec = Recorder()
        rec.fit(d)
        harness.run_new_users(rec, d, n)
        assert len(rec.seen) == len(d.new_users)
        assert all(np.array_equal(h, d.new_user_history[u][:n]) for u, h in rec.seen)


def test_information_gate_opens_at_the_right_n(cold):
    d = cold["data"]
    m = TargetGate(need=2)
    m.fit(d)
    recall = [harness.evaluate_new_users(m, d, n)["recall@5"] for n in COLD_REVEAL]
    assert recall == [0.0, 0.0, 1.0, 1.0]


def test_other_purchases_are_never_candidates_even_when_not_revealed(cold):
    d = cold["data"]
    m = ValOracle()  # the val item is in the user's full sequence, so it must be masked at every n
    m.fit(d)
    assert all(harness.evaluate_new_users(m, d, n)["recall@20"] == 0.0 for n in COLD_REVEAL)


def test_popularity_ignores_history_so_it_is_flat_across_n(cold):
    d = cold["data"]
    m = Popularity()
    m.fit(d)
    vals = {harness.evaluate_new_users(m, d, n)["recall@10"] for n in COLD_REVEAL}
    assert len(vals) == 1


def test_new_user_metrics_have_the_standard_keys_with_no_rating(cold):
    d = cold["data"]
    m = Popularity()
    m.fit(d)
    cold_metrics = harness.evaluate_new_users(m, d, 1)
    assert set(cold_metrics) == set(harness.evaluate(m, d, "test")) and cold_metrics["rmse"] is None


def test_run_experiment_writes_the_cold_start_section(cold, tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "RESULTS_DIR", tmp_path)
    d = cold["data"]
    res = harness.run_experiment(Popularity(), d, tag="c")
    saved = json.loads((tmp_path / "popularity_c.json").read_text())
    assert saved["n_eval_users"] == len(d.eval_users) < d.n_users
    assert set(saved["cold_start"]["by_revealed"]) == {str(n) for n in COLD_REVEAL}
    assert saved["cold_start"]["n_new_users"] == len(d.new_users)
    assert sum(v["n"] for v in saved["test_slices"]["user_bin"].values()) == len(d.eval_users)
    per_user = np.load(tmp_path / "per_user" / "popularity_c.npz")
    assert all(len(per_user[f"rank_new_n{n}"]) == len(d.new_users) for n in COLD_REVEAL)
    assert res["cold_start"]["by_revealed"]["0"]["recall@10"] == saved["cold_start"]["by_revealed"]["0"]["recall@10"]


def test_new_user_evaluation_needs_a_cold_start_dataset(processed):
    m = Popularity()
    m.fit(processed["data"])
    with pytest.raises(ValueError, match="no new users"):
        harness.run_new_users(m, processed["data"], 1)


def test_load_refuses_new_users_that_have_training_rows(cold, tmp_path):
    shutil.copytree(cold["dir"], tmp_path / "copy")
    inter = pd.read_parquet(tmp_path / "copy" / "new_user_interactions.parquet")
    train = pd.read_parquet(tmp_path / "copy" / "train.parquet")
    pd.concat([train, inter.iloc[:1]]).to_parquet(tmp_path / "copy" / "train.parquet", index=False)
    with pytest.raises(ValueError):  # fingerprint or the new-user check, either way it must not load
        load(tmp_path / "copy")


def test_cold_start_curve_renders(cold, tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "RESULTS_DIR", tmp_path)
    d = cold["data"]
    harness.run_experiment(Popularity(), d)
    gate = TargetGate(need=2)
    harness.run_experiment(gate, d)
    ax = cold_start_curve(tmp_path)
    assert len(ax.lines) == 2
