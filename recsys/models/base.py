"""The interface every model implements. Copy recsys/models/popularity.py as a starting point."""
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from recsys.data import SplitData
from recsys.metrics import mask_history, topk_items


class Recommender(ABC):
    # Must equal the file name, e.g. recsys/models/knn.py -> name = "knn".
    name: str = ""
    # True only if predict() returns star-rating estimates (enables RMSE/MAE).
    supports_rating: bool = False

    def __init__(self, **config):
        # Hyperparameters only, JSON-serialisable. Saved into the results file.
        self.config = config

    @abstractmethod
    def fit(self, data: SplitData) -> None:
        """Train on data.train. data.val may be used for early stopping or tuning, never data.test."""

    @abstractmethod
    def score(self, user_idxs: np.ndarray, histories: list[np.ndarray]) -> np.ndarray:
        """Scores for the FULL catalog: numpy float32 array of shape [len(user_idxs), n_items], higher = better.

        histories[i] is the chronological item_idx array the model may condition on for user_idxs[i]
        (train items when evaluating validation, train+val items when evaluating test). Models that
        learn a vector per user at fit time may ignore it; sequence and neighbourhood models use it.
        Do not mask the history yourself, the harness does that. Must be callable at any point after
        fit() has started producing a usable model, so it can be used for early stopping.
        """

    def predict(self, user_idxs: np.ndarray, item_idxs: np.ndarray) -> np.ndarray:
        """Star-rating estimates for (user, item) pairs. Only needed if supports_rating is True."""
        raise NotImplementedError

    def num_parameters(self) -> int | None:
        """Trainable parameter count for the cost table, or None if not meaningful."""
        return None

    def recommend(self, user_idx: int, k: int = 10, history: np.ndarray | None = None) -> list[int]:
        """Top-k item_idx for one user, excluding their history. Used by the demo."""
        history = np.asarray([] if history is None else history, dtype=np.int64)
        scores = np.array(self.score(np.array([user_idx]), [history]), dtype=np.float32)
        mask_history(scores, [history])
        return topk_items(scores, k)[0].tolist()

    def save(self, path) -> None:
        """Pickle the fitted model (torch networks are moved to the CPU first) so it can be reloaded without retraining."""
        net, dev = getattr(self, "net", None), getattr(self, "dev", None)
        if net is not None:
            net.to("cpu")
            self.dev = "cpu"
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            if net is not None:
                net.to(dev)
                self.dev = dev

    @staticmethod
    def load(path) -> "Recommender":
        """Reload a model saved with save(); torch networks go to the best available device."""
        from recsys.config import get_device

        with open(path, "rb") as f:
            model = pickle.load(f)
        if getattr(model, "net", None) is not None:
            model.dev = get_device()
            model.net.to(model.dev)
        return model
