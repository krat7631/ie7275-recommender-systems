import numpy as np
import pytest

from recsys.metrics import beyond_accuracy, mask_history, ranking_metrics, rating_metrics, target_ranks, topk_items


def test_rank_is_pessimistic_on_ties():
    scores = np.array([[0.1, 0.9, 0.5, 0.5, 0.2]], dtype=np.float32)
    assert target_ranks(scores, np.array([2]))[0] == 3  # items 1, 2, 3 all score >= 0.5
    assert target_ranks(scores, np.array([1]))[0] == 1


def test_constant_scores_get_worst_rank():
    scores = np.zeros((4, 50), dtype=np.float32)
    assert (target_ranks(scores, np.array([0, 7, 20, 49])) == 50).all()


def test_history_is_removed_from_ranking():
    scores = np.array([[0.1, 0.9, 0.5, 0.4, 0.2]], dtype=np.float32)
    hist = [np.array([1, 4])]
    mask_history(scores, hist)
    assert np.isneginf(scores[0, [1, 4]]).all()
    assert target_ranks(scores, np.array([3]))[0] == 2  # 0.9 is masked, so only 0.5 beats 0.4
    mask_history(scores, [np.array([], dtype=np.int64)])  # empty history is fine


def test_rank_ignores_masked_items_but_not_target():
    scores = np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    mask_history(scores, [np.array([0, 1])])
    assert target_ranks(scores, np.array([2]))[0] == 1


def test_nan_scores_raise():
    with pytest.raises(ValueError):
        target_ranks(np.array([[0.1, np.nan]], dtype=np.float32), np.array([0]))


def test_ranking_metrics_known_values():
    m = ranking_metrics(np.array([1, 3, 11]), ks=(10,))
    assert m["recall@10"] == pytest.approx(2 / 3)
    assert m["precision@10"] == pytest.approx(0.2 / 3)
    assert m["ndcg@10"] == pytest.approx((1 + 1 / np.log2(4)) / 3)
    assert m["f1@10"] == pytest.approx((2 * 0.1 * 1 / 1.1) * 2 / 3)
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 11) / 3)


def test_topk_order_and_shape():
    scores = np.array([[0.1, 0.9, 0.5, 0.7], [0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
    top = topk_items(scores, 3)
    assert top.tolist() == [[1, 3, 2], [0, 1, 2]]


def test_beyond_accuracy_known_values():
    top = np.array([[0, 1], [0, 2]])
    counts = np.array([3, 1, 1, 0])
    m = beyond_accuracy(top, 4, counts, ks=(2,))
    assert m["coverage@2"] == pytest.approx(3 / 4)
    assert m["avg_pop@2"] == pytest.approx(2.0)
    assert m["gini@2"] == pytest.approx(0.375)  # exposure counts [2, 1, 1, 0]
    p = np.array([3, 1, 1, 1]) / 5
    assert m["novelty@2"] == pytest.approx((-np.log2(p))[top].mean())


def test_rating_metrics_clip_to_star_range():
    m = rating_metrics(np.array([6.0, 0.0]), np.array([5.0, 1.0]))
    assert m["rmse"] == 0.0 and m["mae"] == 0.0


def test_random_scorer_matches_chance():
    rng = np.random.default_rng(0)
    n_users, n_items = 20000, 1000
    scores = rng.random((n_users, n_items), dtype=np.float32)
    ranks = target_ranks(scores, rng.integers(0, n_items, n_users))
    assert ranking_metrics(ranks, ks=(10,))["recall@10"] == pytest.approx(10 / n_items, abs=0.003)
