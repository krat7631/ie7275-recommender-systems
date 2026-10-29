import json

import numpy as np
import pytest

from recsys import config, harness
from recsys.models.base import Recommender
from recsys.models.popularity import Popularity


class ValOracle(Recommender):
    """Puts the validation item first and everything else tied. Cheats on val, must be blind on test."""

    name = "val_oracle"

    def fit(self, data):
        self.val_item = data.val["item_idx"].to_numpy()
        self.n_items = data.n_items

    def score(self, user_idxs, histories):
        s = np.zeros((len(user_idxs), self.n_items), dtype=np.float32)
        s[np.arange(len(user_idxs)), self.val_item[user_idxs]] = 1.0
        return s


class TargetOracle(Recommender):
    name = "test_oracle"

    def fit(self, data):
        self.test_item = data.test["item_idx"].to_numpy()
        self.n_items = data.n_items

    def score(self, user_idxs, histories):
        s = np.zeros((len(user_idxs), self.n_items), dtype=np.float32)
        s[np.arange(len(user_idxs)), self.test_item[user_idxs]] = 1.0
        return s


class Constant(Recommender):
    name = "constant"

    def fit(self, data):
        self.n_items = data.n_items

    def score(self, user_idxs, histories):
        return np.zeros((len(user_idxs), self.n_items), dtype=np.float32)


def test_val_oracle_wins_val_and_is_blind_on_test(processed):
    d = processed["data"]
    m = ValOracle()
    m.fit(d)
    assert harness.evaluate(m, d, "val")["recall@5"] == 1.0
    # on test the val item is part of the history and gets masked, so the oracle has no information left
    assert harness.evaluate(m, d, "test")["recall@20"] == 0.0


def test_test_oracle_wins_test(processed):
    d = processed["data"]
    m = TargetOracle()
    m.fit(d)
    assert harness.evaluate(m, d, "test")["mrr"] == 1.0


def test_constant_model_gets_no_credit_for_ties(processed):
    d = processed["data"]
    m = Constant()
    m.fit(d)
    r = harness.evaluate(m, d, "test")
    assert r["recall@20"] == 0.0
    assert r["rmse"] is None


def test_popularity_beats_chance_and_has_a_sane_rmse(processed):
    d = processed["data"]
    m = Popularity()
    m.fit(d)
    r = harness.evaluate(m, d, "test")
    assert r["recall@20"] > 1.5 * 20 / d.n_items
    assert 0.5 < r["rmse"] < 3.0
    assert 0 < r["coverage@10"] <= 1


def test_max_users_subset_is_fixed(processed):
    d = processed["data"]
    m = Popularity()
    m.fit(d)
    a = harness.evaluate(m, d, "val", max_users=50)
    b = harness.evaluate(m, d, "val", max_users=50)
    assert a == b


def test_score_shape_and_type_are_enforced(processed):
    d = processed["data"]

    class Bad(Constant):
        name = "bad"

        def score(self, user_idxs, histories):
            return np.zeros((len(user_idxs), d.n_items - 1), dtype=np.float32)

    m = Bad()
    m.fit(d)
    with pytest.raises(ValueError):
        harness.evaluate(m, d, "val")


def test_scoring_does_not_corrupt_model_state(processed):
    d = processed["data"]
    m = Popularity()
    m.fit(d)
    before = m.counts.copy()
    harness.evaluate(m, d, "test")
    assert (m.counts == before).all()


def test_run_experiment_writes_the_results_contract(processed, tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "RESULTS_DIR", tmp_path)
    d = processed["data"]
    res = harness.run_experiment(Popularity(), d, tag="smoke")
    saved = json.loads((tmp_path / "popularity_smoke.json").read_text())
    assert saved["run_name"] == "popularity_smoke" and saved["data_version"] == d.data_version
    expected = {f"{m}@{k}" for m in ("precision", "recall", "f1", "ndcg", "coverage", "novelty", "avg_pop", "gini") for k in config.K_LIST}
    expected |= {"mrr", "rmse", "mae"}
    assert set(saved["val"]) == set(saved["test"]) == expected
    assert set(saved["cost"]) == {"fit_seconds", "score_ms_per_user", "n_params"}
    assert set(saved["test_slices"]) == {"user_bin", "item_bin"}
    assert saved["cold_start"] is None and saved["n_eval_users"] == d.n_users
    assert sum(v["n"] for v in saved["test_slices"]["user_bin"].values()) == d.n_users
    per_user = np.load(tmp_path / "per_user" / "popularity_smoke.npz")
    assert len(per_user["rank_test"]) == d.n_users
    assert res["test"]["recall@10"] == saved["test"]["recall@10"]


def test_popularity_rating_shrinkage_interpolates_between_item_mean_and_global_mean(processed):
    d = processed["data"]
    g = float(d.train["rating"].mean())
    heavy = Popularity(shrinkage=1e9)
    heavy.fit(d)
    assert np.allclose(heavy.item_mean, g, atol=1e-3)  # huge shrinkage: everything collapses to the global mean
    none = Popularity(shrinkage=0)
    none.fit(d)
    i = int(np.argmax(none.counts))
    raw = d.train.loc[d.train["item_idx"] == i, "rating"].mean()
    assert none.item_mean[i] == pytest.approx(raw, abs=1e-4)  # no shrinkage: the plain item mean
