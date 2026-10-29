"""Negative sampling shared by the models that train on sampled negatives (BPR, NeuMF, SASRec)."""
import numpy as np


def interaction_keys(users: np.ndarray, items: np.ndarray, n_items: int) -> np.ndarray:
    """Sorted integer keys user * n_items + item of every known interaction, for fast membership tests."""
    return np.sort(users.astype(np.int64) * n_items + items.astype(np.int64))


def sample_negatives(rng: np.random.Generator, users: np.ndarray, n_items: int, sorted_keys: np.ndarray) -> np.ndarray:
    """One uniformly random item per entry of `users`, never an item that user has in `sorted_keys` (rejection sampling)."""
    users = users.astype(np.int64)
    neg = rng.integers(0, n_items, size=len(users))
    bad = _known(sorted_keys, users * n_items + neg)
    while bad.any():
        idx = np.flatnonzero(bad)
        neg[idx] = rng.integers(0, n_items, size=len(idx))
        bad[idx] = _known(sorted_keys, users[idx] * n_items + neg[idx])
    return neg


def _known(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
    pos = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    return sorted_keys[pos] == keys
