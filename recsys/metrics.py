"""The only metric implementations in the project. Models must never compute their own.

Protocol: each user has ONE held-out target item. The model scores the full catalog, items in the
user's history are removed from the ranking, and we record the rank of the target (1 = best).
Every ranking metric below is a function of that rank.

Ties: rank = 1 + number of other candidate items with score >= the target's score. A model that
returns identical scores for everything therefore gets the worst possible rank, not a lucky one.
Because there is a single relevant item, recall@k equals hit rate@k, and precision@k and f1@k are
deterministic functions of it, so they carry no independent information.
"""
import numpy as np

from recsys.config import K_LIST


def mask_history(scores: np.ndarray, histories: list[np.ndarray]) -> None:
    """In place: set the score of every already-seen item to -inf."""
    lens = np.fromiter((len(h) for h in histories), dtype=np.int64, count=len(histories))
    if lens.sum() == 0:
        return
    rows = np.repeat(np.arange(len(histories)), lens)
    cols = np.concatenate(histories)
    scores[rows, cols] = -np.inf


def target_ranks(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """1-based rank of each row's target item, pessimistic on ties. scores: [B, n_items]."""
    if np.isnan(scores).any():
        raise ValueError("model.score() returned NaN")
    target_score = scores[np.arange(len(targets)), targets]
    return (scores >= target_score[:, None]).sum(axis=1).astype(np.int32)


def topk_items(scores: np.ndarray, k: int) -> np.ndarray:
    """Top-k item indices per row, best first. Order among exactly tied scores is unspecified."""
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1).astype(np.int32)


def ranking_metrics(ranks: np.ndarray, ks=K_LIST) -> dict[str, float]:
    out = {}
    for k in ks:
        hit = (ranks <= k).astype(np.float64)
        precision = hit / k
        recall = hit
        denom = precision + recall
        f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
        ndcg = np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
        out[f"precision@{k}"] = float(precision.mean())
        out[f"recall@{k}"] = float(recall.mean())
        out[f"f1@{k}"] = float(f1.mean())
        out[f"ndcg@{k}"] = float(ndcg.mean())
    out["mrr"] = float((1.0 / ranks).mean())
    return out


def rating_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Predictions are clipped to the 1-5 star range before scoring."""
    err = np.clip(pred, 1.0, 5.0) - true
    return {"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err)))}


def _gini(counts: np.ndarray) -> float:
    c = np.sort(counts.astype(np.float64))
    n, total = len(c), c.sum()
    if total == 0:
        return 0.0
    return float(2 * np.sum(np.arange(1, n + 1) * c) / (n * total) - (n + 1) / n)


def beyond_accuracy(topk: np.ndarray, n_items: int, item_train_count: np.ndarray, ks=K_LIST) -> dict[str, float]:
    """Diversity / popularity-bias measures over the recommended lists of all evaluated users.

    coverage@k : share of the catalog that appears in at least one user's top-k
    novelty@k  : mean of -log2(p(item)) over recommended items, p = train count / total train
                 interactions (items with zero train count use a count of 1)
    avg_pop@k  : mean train interaction count of the recommended items
    gini@k     : Gini index of how often each item is recommended (0 = even, 1 = one item only)
    """
    total = max(int(item_train_count.sum()), 1)
    p = np.maximum(item_train_count, 1) / total
    info = -np.log2(p)
    out = {}
    for k in ks:
        top = topk[:, :k]
        out[f"coverage@{k}"] = float(len(np.unique(top)) / n_items)
        out[f"novelty@{k}"] = float(info[top].mean())
        out[f"avg_pop@{k}"] = float(item_train_count[top].mean())
        out[f"gini@{k}"] = _gini(np.bincount(top.ravel(), minlength=n_items))
    return out
