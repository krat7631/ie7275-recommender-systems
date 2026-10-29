"""Reference model: recommend the most-interacted items to everyone.

Also the template for the other models. For rating prediction it uses a shrunk item mean,
(sum of item ratings + m * global mean) / (item count + m), which is the RMSE baseline the
rating-prediction models have to beat. m = 10 was chosen on validation RMSE (m = 5 to 10 is
flat: 1.1797 vs 1.1795; no shrinkage gives 1.1957; the global mean gives 1.226).
"""
import numpy as np

from recsys.data import SplitData
from recsys.models.base import Recommender


class Popularity(Recommender):
    name = "popularity"
    supports_rating = True

    def __init__(self, shrinkage: float = 10, **kw):
        super().__init__(shrinkage=shrinkage, **kw)
        self.shrinkage = shrinkage

    def fit(self, data: SplitData) -> None:
        item = data.train["item_idx"].to_numpy()
        rating = data.train["rating"].to_numpy()
        self.counts = np.bincount(item, minlength=data.n_items).astype(np.float32)
        sums = np.bincount(item, weights=rating, minlength=data.n_items)
        global_mean = float(rating.mean())
        m = self.shrinkage
        self.item_mean = ((sums + m * global_mean) / (self.counts + m)).astype(np.float32)

    def score(self, user_idxs: np.ndarray, histories: list[np.ndarray]) -> np.ndarray:
        return np.tile(self.counts, (len(user_idxs), 1))

    def predict(self, user_idxs: np.ndarray, item_idxs: np.ndarray) -> np.ndarray:
        return self.item_mean[item_idxs]
