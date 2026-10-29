import json

import numpy as np
import pytest

from recsys.analysis import bootstrap_means, comparison_table, paired_p_values, per_user_values


def test_per_user_values_match_the_metric_definitions():
    ranks = np.array([1, 3, 11])
    assert per_user_values(ranks, "recall@10").tolist() == [1, 1, 0]
    assert per_user_values(ranks, "ndcg@10") == pytest.approx([1.0, 0.5, 0.0])
    assert per_user_values(ranks, "mrr") == pytest.approx([1.0, 1 / 3, 1 / 11])


def test_bootstrap_uses_the_same_resamples_for_every_model():
    rng = np.random.default_rng(0)
    a = rng.random(500)
    out = bootstrap_means({"a": a, "a_copy": a.copy(), "shifted": a + 0.1}, n_boot=200)
    assert np.array_equal(out["a"], out["a_copy"])
    assert np.allclose(out["shifted"] - out["a"], 0.1)  # paired: the shift survives resampling exactly
    assert abs(out["a"].mean() - a.mean()) < 0.01


def test_bootstrap_interval_covers_the_mean_and_shrinks_with_more_users():
    rng = np.random.default_rng(1)
    widths = []
    for n in (200, 5000):
        v = (rng.random(n) < 0.3).astype(float)
        boots = bootstrap_means({"m": v}, n_boot=500)["m"]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        assert lo <= v.mean() <= hi
        widths.append(hi - lo)
    assert widths[1] < widths[0] / 3


def _fake_results(tmp_path, ranks_by_model):
    (tmp_path / "per_user").mkdir(parents=True)
    for name, ranks in ranks_by_model.items():
        (tmp_path / f"{name}_final.json").write_text(json.dumps({"model": name, "tag": "final", "run_name": f"{name}_final"}))
        np.savez(tmp_path / "per_user" / f"{name}_final.npz", rank_test=ranks, user_idx=np.arange(len(ranks)))


def test_comparison_and_paired_tests_on_planted_differences(tmp_path):
    rng = np.random.default_rng(2)
    n = 4000
    good = np.where(rng.random(n) < 0.20, rng.integers(1, 10, n), rng.integers(11, 1000, n))
    bad = np.where(rng.random(n) < 0.05, rng.integers(1, 10, n), rng.integers(11, 1000, n))
    _fake_results(tmp_path, {"good": good, "bad": bad, "good_twin": good.copy()})
    table = comparison_table(tmp_path, metrics=("recall@10", "ndcg@10"), n_boot=300)
    assert table.loc[0, "model"] in ("good", "good_twin") and table.iloc[-1]["model"] == "bad"
    row = table.set_index("model").loc["good"]
    assert row["recall@10_lo"] <= row["recall@10"] <= row["recall@10_hi"]
    diff, pval = paired_p_values(tmp_path, "ndcg@10", n_boot=300)
    assert pval.loc["good", "bad"] < 0.01 and diff.loc["good", "bad"] > 0
    assert pval.loc["good", "good_twin"] == 1.0  # identical per-user results: no evidence of a difference
