"""Biased matrix factorization for rating prediction (FunkSVD with biases), written from scratch in PyTorch.

Predicted rating = mu + b_user + b_item + p_user . q_item, where mu is the mean training rating. Trained with Adam on
recsys.config.get_device(), mini-batches of ratings, squared-error loss plus an L2 penalty `reg` on the factors and
biases of the rows in the batch. After every epoch the validation RMSE (2,000 fixed users) decides early stopping;
the best epoch's weights are restored. self.curve_ holds {epoch, train_loss, val_rmse}.

Hyperparameters: factors (32), lr (0.005), reg (0.05), epochs (30), patience (3), batch_size (8192).

This is the rating-prediction model: predict() returns star ratings (clipped to 1-5 by the harness) and RMSE/MAE are
reported for it. For the ranking metrics score() ranks every item by its predicted rating, so it is optimised for the
wrong objective on purpose: the comparison shows how a rating-accuracy model does at top-N recommendation.

Unseen users (no training interactions) get b_user = 0 and p_user = 0 in predict() (so the prediction is mu + b_item)
and popularity scores (data.item_train_count) in score(); the model has no way to use their revealed purchases.

Training time: see the results files (cost.fit_seconds).
"""
import numpy as np
import torch

from recsys.config import get_device
from recsys.data import SplitData
from recsys.harness import evaluate
from recsys.models.base import Recommender


class Svd(Recommender):
    name = "svd"
    supports_rating = True

    def __init__(self, factors: int = 32, lr: float = 0.005, reg: float = 0.05, epochs: int = 30,
                 patience: int = 3, batch_size: int = 8192, **kw):
        super().__init__(factors=factors, lr=lr, reg=reg, epochs=epochs, patience=patience, batch_size=batch_size, **kw)
        self.factors, self.lr, self.reg, self.epochs = factors, lr, reg, epochs
        self.patience, self.batch_size = patience, batch_size

    def _sync(self, P, Q, bu, bi) -> None:
        """Copy the current weights to numpy so predict()/score() work during training."""
        self.P = P.detach().cpu().numpy().copy()
        self.Q = Q.detach().cpu().numpy().copy()
        self.bu = bu.detach().cpu().numpy().copy()
        self.bi = bi.detach().cpu().numpy().copy()
        self.P[~self.seen] = 0.0
        self.bu[~self.seen] = 0.0

    def fit(self, data: SplitData) -> None:
        dev = get_device()
        users = data.train["user_idx"].to_numpy()
        self.seen = np.bincount(users, minlength=data.n_users) > 0
        self.pop = data.item_train_count.astype(np.float32)
        u = torch.as_tensor(users, dtype=torch.long, device=dev)
        i = torch.as_tensor(data.train["item_idx"].to_numpy(), dtype=torch.long, device=dev)
        r = torch.as_tensor(data.train["rating"].to_numpy(), dtype=torch.float32, device=dev)
        self.mu = float(r.mean())

        P = torch.nn.Parameter(0.1 * torch.randn(data.n_users, self.factors, device=dev))
        Q = torch.nn.Parameter(0.1 * torch.randn(data.n_items, self.factors, device=dev))
        bu = torch.nn.Parameter(torch.zeros(data.n_users, device=dev))
        bi = torch.nn.Parameter(torch.zeros(data.n_items, device=dev))
        opt = torch.optim.Adam([P, Q, bu, bi], lr=self.lr)

        self.curve_, best, best_state, stale = [], float("inf"), None, 0
        for epoch in range(1, self.epochs + 1):
            perm = torch.randperm(len(r), device=dev)
            total = torch.zeros((), device=dev)
            for s in range(0, len(r), self.batch_size):
                b = perm[s : s + self.batch_size]
                pu, qi = P[u[b]], Q[i[b]]
                pred = self.mu + bu[u[b]] + bi[i[b]] + (pu * qi).sum(1)
                penalty = (pu.pow(2).sum(1) + qi.pow(2).sum(1) + bu[u[b]].pow(2) + bi[i[b]].pow(2)).mean()
                loss = (pred - r[b]).pow(2).mean() + self.reg * penalty
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.detach() * len(b)
            self._sync(P, Q, bu, bi)
            val_rmse = evaluate(self, data, "val", max_users=2000)["rmse"]
            self.curve_.append({"epoch": epoch, "train_loss": float(total) / len(r), "val_rmse": val_rmse})
            if val_rmse < best - 1e-5:
                best, stale = val_rmse, 0
                best_state = (P.detach().clone(), Q.detach().clone(), bu.detach().clone(), bi.detach().clone())
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self._sync(*best_state)
        self.best_epoch_ = int(np.argmin([c["val_rmse"] for c in self.curve_])) + 1

    def predict(self, user_idxs: np.ndarray, item_idxs: np.ndarray) -> np.ndarray:
        return self.mu + self.bu[user_idxs] + self.bi[item_idxs] + (self.P[user_idxs] * self.Q[item_idxs]).sum(1)

    def score(self, user_idxs: np.ndarray, histories: list[np.ndarray]) -> np.ndarray:
        scores = self.mu + self.bu[user_idxs][:, None] + self.bi[None, :] + self.P[user_idxs] @ self.Q.T
        scores[~self.seen[user_idxs]] = self.pop
        return scores.astype(np.float32, copy=False)

    def num_parameters(self) -> int:
        return int(self.P.size + self.Q.size + self.bu.size + self.bi.size)
