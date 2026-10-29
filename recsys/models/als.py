"""Implicit-feedback ALS matrix factorization (Hu, Koren and Volinsky, 2008), written from scratch.

Every training interaction is a positive (preference 1) with confidence 1 + alpha; every other pair has preference 0
and confidence 1. User factors X and item factors Y are fitted by alternating closed-form regularised least squares:
for a user u with items I_u,   x_u = (Y^T Y + alpha * Y_I^T Y_I + reg * I)^-1 * (1 + alpha) * sum of y_i for i in I_u,
which uses Y^T C_u Y = Y^T Y + Y^T (C_u - I) Y so that each solve only touches the user's own items. The solves are
batched with numpy over users of similar degree, so one iteration is a few seconds on the full data.
The score of an item for a user is x_u . y_i.

Hyperparameters: factors (64), reg (0.1), alpha (40), iterations (15), foldin (False), foldin_all (False).

Unseen users (no training interactions): with foldin=False the model returns popularity scores (data.item_train_count),
because it has no vector for them. With foldin=True a vector is computed from the purchases in the history it is given,
with the same closed-form update and the item factors held fixed; an empty history still gets popularity.

foldin_all=True computes every user's vector from the history the harness gives it instead of using the vector learned in
fit(). When the test item is scored that history includes the validation purchase, which the learned vectors never saw, so
this removes the protocol advantage that history-based models (KNN, SASRec) have over per-user-vector models. The item
factors are the same, nothing is retrained.

Training time: see the results files (cost.fit_seconds). self.curve_ holds the validation NDCG@10 after each iteration.
"""
import numpy as np

from recsys.config import SEED
from recsys.data import SplitData, to_csr
from recsys.harness import evaluate
from recsys.models.base import Recommender


def _solve_side(F: np.ndarray, indptr: np.ndarray, indices: np.ndarray, alpha: float, reg: float, budget: int = 20_000_000,
                FtF: np.ndarray | None = None) -> np.ndarray:
    """New factors for every row of a CSR matrix, given the fixed factors F of the other side."""
    n, f = len(indptr) - 1, F.shape[1]
    if FtF is None:
        FtF = F.T.astype(np.float64) @ F.astype(np.float64) + reg * np.eye(f)
    deg = np.diff(indptr)
    order = np.argsort(deg, kind="stable")
    out = np.zeros((n, f), dtype=np.float32)
    start = 0
    while start < n:
        # grow the chunk while (rows x max degree x factors) stays within the memory budget
        end = start + 1
        while end < n and (end + 1 - start) * deg[order[end]] * f <= budget and end - start < 8192:
            end += 1
        rows = order[start:end]
        degs = deg[rows]
        d = int(degs.max())
        if d > 0:
            B = len(rows)
            r = np.repeat(np.arange(B), degs)
            offs = np.arange(degs.sum()) - np.repeat(np.cumsum(degs) - degs, degs)
            src = np.repeat(indptr[rows], degs) + offs
            idx = np.zeros((B, d), dtype=np.int64)
            mask = np.zeros((B, d), dtype=np.float32)
            idx[r, offs] = indices[src]
            mask[r, offs] = 1.0
            G = F[idx] * mask[..., None]  # [B, d, f]; padded slots are zero vectors
            A = FtF + alpha * np.matmul(G.transpose(0, 2, 1), G).astype(np.float64)  # batched BLAS, not einsum
            b = (1.0 + alpha) * G.sum(axis=1, dtype=np.float64)
            out[rows] = np.linalg.solve(A, b[..., None])[..., 0].astype(np.float32)
        start = end
    return out


class Als(Recommender):
    name = "als"

    def __init__(self, factors: int = 64, reg: float = 0.1, alpha: float = 40.0, iterations: int = 15, foldin: bool = False,
                 foldin_all: bool = False, **kw):
        super().__init__(factors=factors, reg=reg, alpha=alpha, iterations=iterations, foldin=foldin, foldin_all=foldin_all, **kw)
        self.factors, self.reg, self.alpha, self.iterations = factors, reg, alpha, iterations
        self.foldin, self.foldin_all = foldin, foldin_all

    def fit(self, data: SplitData) -> None:
        R = to_csr(data.train, data.n_users, data.n_items).tocsr()
        Rt = R.T.tocsr()
        self.n_items = data.n_items
        self.pop = data.item_train_count.astype(np.float32)
        self.seen = np.bincount(data.train["user_idx"].to_numpy(), minlength=data.n_users) > 0

        rng = np.random.default_rng(SEED)
        self.X = np.zeros((data.n_users, self.factors), dtype=np.float32)
        self.Y = (0.1 * rng.standard_normal((data.n_items, self.factors))).astype(np.float32)
        self._prepare()
        self.curve_ = []
        for it in range(1, self.iterations + 1):
            self.X = _solve_side(self.Y, R.indptr, R.indices, self.alpha, self.reg)
            self.Y = _solve_side(self.X, Rt.indptr, Rt.indices, self.alpha, self.reg)
            self._prepare()
            self.curve_.append({"iteration": it, "val_ndcg@10": evaluate(self, data, "val", max_users=2000)["ndcg@10"]})

    def _prepare(self) -> None:
        self.YtY = self.Y.T.astype(np.float64) @ self.Y.astype(np.float64) + self.reg * np.eye(self.factors)

    def _vectors_from_history(self, histories: list[np.ndarray]) -> np.ndarray:
        """Closed-form user vectors for the given histories with the item factors held fixed (zero vector if empty)."""
        lens = np.array([len(h) for h in histories])
        indptr = np.r_[0, np.cumsum(lens)]
        indices = np.concatenate(histories) if lens.sum() else np.zeros(0, dtype=np.int64)
        return _solve_side(self.Y, indptr, indices, self.alpha, self.reg, FtF=self.YtY)

    def score(self, user_idxs: np.ndarray, histories: list[np.ndarray]) -> np.ndarray:
        lens = np.array([len(h) for h in histories])
        X = np.zeros((len(user_idxs), self.factors), dtype=np.float32) if self.X is None else self.X[user_idxs]  # None: slim copy
        if self.foldin_all:
            rows = np.arange(len(user_idxs))  # every user's vector comes from the history, learned vectors are not read
        elif self.foldin:
            rows = np.flatnonzero(~self.seen[user_idxs])
        else:
            rows = np.zeros(0, dtype=int)
        if len(rows):
            X[rows] = self._vectors_from_history([histories[r] for r in rows])
        scores = X @ self.Y.T
        unseen = ~self.seen[user_idxs]
        if not (self.foldin or self.foldin_all):
            scores[unseen] = self.pop  # no vector for them and no fold-in: popularity
        else:
            scores[(lens == 0) & (unseen | self.foldin_all)] = self.pop  # fold-in needs a history to work from
        return scores.astype(np.float32, copy=False)

    def num_parameters(self) -> int:
        return int((self.X.shape[0] + self.Y.shape[0]) * self.factors)
