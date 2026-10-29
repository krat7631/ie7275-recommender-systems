"""Item-item collaborative filtering with cosine similarity (neighbourhood model).

Algorithm: build the implicit user-item matrix from data.train, compute the cosine similarity between item columns
with scipy sparse matrices, and keep only the top k most similar items for each item (the full 25,612 x 25,612
matrix would be 2.6 GB). A user's score for an item is the sum of its similarity to the items in the user's history.

Hyperparameter: k (neighbours kept per item, default 50).

Training time: about 0.2 s on the full data (the similarity product and the top-k selection are the whole fit).

Unseen users: the model reads the history it is given, so a new user with 1 to 3 revealed purchases is scored from
those purchases. A user with an empty history gets popularity scores (data.item_train_count).
"""
import numpy as np
import scipy.sparse as sp

from recsys.data import SplitData, to_csr
from recsys.models.base import Recommender


class Knn(Recommender):
    name = "knn"

    def __init__(self, k: int = 50, **kw):
        super().__init__(k=k, **kw)
        self.k = k

    def fit(self, data: SplitData) -> None:
        R = to_csr(data.train, data.n_users, data.n_items).tocsc()
        norms = np.sqrt(np.asarray(R.multiply(R).sum(axis=0)).ravel()) + 1e-9
        R = R @ sp.diags(1.0 / norms)
        S = (R.T @ R).tocsr()
        S.setdiag(0)
        S.eliminate_zeros()

        rows, cols, vals = [], [], []
        for i in range(S.shape[0]):
            lo, hi = S.indptr[i], S.indptr[i + 1]
            v, c = S.data[lo:hi], S.indices[lo:hi]
            if len(v) > self.k:
                top = np.argpartition(-v, self.k - 1)[: self.k]
                v, c = v[top], c[top]
            rows.append(np.full(len(v), i))
            cols.append(c)
            vals.append(v)
        self.S = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=S.shape).astype(np.float32)
        self.n_items = data.n_items
        self.pop = data.item_train_count.astype(np.float32)

    def score(self, user_idxs: np.ndarray, histories: list[np.ndarray]) -> np.ndarray:
        lens = np.array([len(h) for h in histories])
        H = sp.csr_matrix(
            (np.ones(lens.sum(), dtype=np.float32), np.concatenate(histories), np.r_[0, np.cumsum(lens)]),
            shape=(len(histories), self.n_items),
        )
        scores = (H @ self.S).toarray()
        scores[lens == 0] = self.pop
        return scores

    def num_parameters(self) -> int:
        return int(self.S.nnz)
