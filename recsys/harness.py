"""Evaluation harness shared by every model.

evaluate(model, data, split) is cheap enough to call inside a training loop for early stopping.
run_experiment(model, data) is what recsys.run calls: fit, evaluate val + test, write results.

On the cold-start dataset (data/processed_coldstart) run_experiment also scores the new users:
the model is shown only the first n purchases of each new user and must rank their last purchase.
The candidate list excludes all of the user's other purchases for every n, so the target and the
candidates are identical across n and only the information given to the model changes.
"""
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from recsys.config import COLD_REVEAL, K_LIST, RESULTS_DIR, SEED, get_device, set_seed
from recsys.data import SplitData
from recsys.metrics import beyond_accuracy, mask_history, rating_metrics, ranking_metrics, target_ranks, topk_items
from recsys.models.base import Recommender


@dataclass
class SplitOutput:
    users: np.ndarray  # user_idx evaluated
    ranks: np.ndarray  # rank of the target item per user, 1 = best
    topk: np.ndarray  # [n_users, max(K_LIST)] recommended item_idx
    rating_pred: np.ndarray | None
    score_seconds: float


def _eval_users(data: SplitData, max_users: int | None, pool: np.ndarray | None = None) -> np.ndarray:
    pool = data.eval_users if pool is None else pool
    if max_users is None or max_users >= len(pool):
        return pool
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(pool, size=max_users, replace=False))


def _score_and_rank(model: Recommender, data: SplitData, users: np.ndarray, targets: np.ndarray,
                    model_hist: list, mask_hist: list, batch_size: int) -> SplitOutput:
    """model_hist is what the model sees, mask_hist is removed from the candidate list. Both indexed by user_idx."""
    ranks = np.empty(len(users), dtype=np.int32)
    topk = np.empty((len(users), max(K_LIST)), dtype=np.int32)
    score_seconds = 0.0
    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        t0 = time.perf_counter()
        raw = model.score(batch, [model_hist[u] for u in batch])
        score_seconds += time.perf_counter() - t0
        if not isinstance(raw, np.ndarray):
            raise TypeError("model.score() must return a numpy array (call .cpu().numpy() on torch tensors)")
        if raw.shape != (len(batch), data.n_items):
            raise ValueError(f"model.score() returned {raw.shape}, expected {(len(batch), data.n_items)}")
        scores = np.array(raw, dtype=np.float32, copy=True)  # copy: masking must not touch model state
        mask_history(scores, [mask_hist[u] for u in batch])
        sl = slice(start, start + len(batch))
        ranks[sl] = target_ranks(scores, targets[batch])
        topk[sl] = topk_items(scores, max(K_LIST))
    return SplitOutput(users, ranks, topk, None, score_seconds)


def run_split(model: Recommender, data: SplitData, split: str, batch_size: int = 1024, max_users: int | None = None) -> SplitOutput:
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    target_df = data.val if split == "val" else data.test
    histories = data.train_history if split == "val" else data.trainval_history
    targets = target_df["item_idx"].to_numpy()
    users = _eval_users(data, max_users)

    out = _score_and_rank(model, data, users, targets, histories, histories, batch_size)
    if model.supports_rating:
        out.rating_pred = np.asarray(model.predict(users, targets[users]), dtype=np.float64)
    return out


def run_new_users(model: Recommender, data: SplitData, n_revealed: int, batch_size: int = 1024, max_users: int | None = None) -> SplitOutput:
    """Cold-start evaluation: show the model the first n_revealed purchases of each new user, rank their last one."""
    if data.new_users is None:
        raise ValueError("no new users in this dataset; set RECSYS_DATA_DIR to the cold-start dataset")
    full = data.new_user_history
    users = _eval_users(data, max_users, pool=data.new_users)
    if min(len(full[u]) for u in users) < n_revealed:
        raise ValueError(f"n_revealed={n_revealed} is longer than some new user's history")
    revealed = [h[:n_revealed] for h in full]
    return _score_and_rank(model, data, users, data.test["item_idx"].to_numpy(), revealed, full, batch_size)


def _summarize(out: SplitOutput, data: SplitData, split: str | None) -> dict:
    metrics = ranking_metrics(out.ranks)
    metrics.update(beyond_accuracy(out.topk, data.n_items, data.item_train_count))
    if out.rating_pred is not None and split is not None:
        target_df = data.val if split == "val" else data.test
        metrics.update(rating_metrics(out.rating_pred, target_df["rating"].to_numpy()[out.users]))
    else:
        metrics.update({"rmse": None, "mae": None})
    return metrics


def evaluate(model: Recommender, data: SplitData, split: str = "val", batch_size: int = 1024, max_users: int | None = None) -> dict:
    """All metrics for one split. Pass max_users (e.g. 2000) for a fast, fixed-subset check during training."""
    return _summarize(run_split(model, data, split, batch_size, max_users), data, split)


def evaluate_new_users(model: Recommender, data: SplitData, n_revealed: int, batch_size: int = 1024, max_users: int | None = None) -> dict:
    """All metrics for the new users when only their first n_revealed purchases are shown (cold-start dataset only)."""
    return _summarize(run_new_users(model, data, n_revealed, batch_size, max_users), data, None)


def _slices(data: SplitData, ranks: np.ndarray, users: np.ndarray) -> dict:
    labels = {
        "user_bin": data.users["activity_bin"].to_numpy()[users],
        "item_bin": data.items["pop_bin"].to_numpy()[data.test["item_idx"].to_numpy()[users]],
    }
    out = {}
    for name, lab in labels.items():
        out[name] = {}
        for b in pd.unique(lab):
            m = lab == b
            out[name][str(b)] = {"n": int(m.sum()), **ranking_metrics(ranks[m])}
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def run_experiment(model: Recommender, data: SplitData, tag: str = "", fit: bool = True) -> dict:
    if not model.name:
        raise ValueError("model.name must be set")
    set_seed(SEED)
    device = get_device()

    if fit:
        t0 = time.perf_counter()
        model.fit(data)
        model.fit_seconds_ = time.perf_counter() - t0
    # fit=False evaluates an already fitted model (e.g. a scoring variant); its recorded training time is reused
    fit_seconds = getattr(model, "fit_seconds_", float("nan"))

    val_out = run_split(model, data, "val")
    test_out = run_split(model, data, "test")

    cold, cold_outs = None, {}
    if data.new_users is not None:
        cold_outs = {n: run_new_users(model, data, n) for n in COLD_REVEAL}
        cold = {
            "n_new_users": len(data.new_users),
            "by_revealed": {str(n): _summarize(o, data, None) for n, o in cold_outs.items()},
        }

    run_name = f"{model.name}_{tag}" if tag else model.name
    result = {
        "model": model.name,
        "tag": tag,
        "run_name": run_name,
        "config": model.config,
        "data_version": data.data_version,
        "seed": SEED,
        "device": device,
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_users": data.n_users,
        "n_items": data.n_items,
        "n_eval_users": len(test_out.users),
        "val": _summarize(val_out, data, "val"),
        "test": _summarize(test_out, data, "test"),
        "test_slices": _slices(data, test_out.ranks, test_out.users),
        "cold_start": cold,
        "curve": getattr(model, "curve_", None),  # per-epoch training curve of iterative models, for the notebooks
        "best_epoch": getattr(model, "best_epoch_", None),
        "cost": {
            "fit_seconds": round(fit_seconds, 2),
            "score_ms_per_user": round(test_out.score_seconds / len(test_out.users) * 1000, 4),
            "n_params": model.num_parameters(),
        },
    }
    if os.environ.get("RECSYS_SAVE_MODEL") and tag == "final":
        model.save(os.environ["RECSYS_SAVE_MODEL"])  # opt-in: keep the fitted model, e.g. for the demo
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "per_user").mkdir(exist_ok=True)
    (RESULTS_DIR / f"{run_name}.json").write_text(json.dumps(result, indent=2, default=_json_default))
    arrays = {"user_idx": test_out.users, "rank_val": val_out.ranks, "rank_test": test_out.ranks}
    if cold_outs:
        arrays["new_user_idx"] = data.new_users
        arrays.update({f"rank_new_n{n}": o.ranks for n, o in cold_outs.items()})
    np.savez_compressed(RESULTS_DIR / "per_user" / f"{run_name}.npz", **arrays)
    return result
