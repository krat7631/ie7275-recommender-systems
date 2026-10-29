"""BPR matrix factorization (Rendle et al., 2009), written from scratch in PyTorch.

score(u, i) = p_u . q_i + b_i. Each epoch, for every training interaction (u, i) one negative item j is sampled
uniformly among the items the user has no interaction with in data.train (collisions are rejection-sampled), and the
model minimises -log(sigmoid(score(u, i) - score(u, j))) plus an L2 penalty `reg` on the factors and item biases of the
rows in the batch. Trained with Adam on recsys.config.get_device(). After every epoch the validation NDCG@10 (2,000
fixed users) decides early stopping and the best epoch's weights are restored. self.curve_ holds
{epoch, loss, val_ndcg@10}.

Hyperparameters: factors (64), lr (0.005), reg (1e-5), epochs (50), batch_size (4096), patience (5).

Unseen users (no training interactions) have no learned vector, so score() returns popularity (data.item_train_count)
for them; the model cannot use their revealed purchases.

Training time: see the results files (cost.fit_seconds).
"""
import numpy as np
import torch

from recsys.config import SEED, get_device
from recsys.data import SplitData
from recsys.harness import evaluate
from recsys.models.base import Recommender
from recsys.sampling import interaction_keys, sample_negatives


class Bpr(Recommender):
    name = "bpr"

    def __init__(self, factors: int = 64, lr: float = 0.005, reg: float = 1e-5, epochs: int = 50,
                 batch_size: int = 4096, patience: int = 5, **kw):
        super().__init__(factors=factors, lr=lr, reg=reg, epochs=epochs, batch_size=batch_size, patience=patience, **kw)
        self.factors, self.lr, self.reg, self.epochs = factors, lr, reg, epochs
        self.batch_size, self.patience = batch_size, patience

    def _sync(self, P, Q, b) -> None:
        self.P = P.detach().cpu().numpy().copy()
        self.Q = Q.detach().cpu().numpy().copy()
        self.b = b.detach().cpu().numpy().copy()

    def fit(self, data: SplitData) -> None:
        dev = get_device()
        rng = np.random.default_rng(SEED)
        users = data.train["user_idx"].to_numpy().astype(np.int64)
        items = data.train["item_idx"].to_numpy().astype(np.int64)
        sorted_keys = interaction_keys(users, items, data.n_items)
        self.seen = np.bincount(users, minlength=data.n_users) > 0
        self.pop = data.item_train_count.astype(np.float32)

        P = torch.nn.Parameter(0.1 * torch.randn(data.n_users, self.factors, device=dev))
        Q = torch.nn.Parameter(0.1 * torch.randn(data.n_items, self.factors, device=dev))
        b = torch.nn.Parameter(torch.zeros(data.n_items, device=dev))
        opt = torch.optim.Adam([P, Q, b], lr=self.lr)
        u_all = torch.as_tensor(users, device=dev)
        i_all = torch.as_tensor(items, device=dev)

        self.curve_, best, best_state, stale = [], -1.0, None, 0
        for epoch in range(1, self.epochs + 1):
            j_all = torch.as_tensor(sample_negatives(rng, users, data.n_items, sorted_keys), device=dev)
            perm = torch.as_tensor(rng.permutation(len(users)), device=dev)
            total = torch.zeros((), device=dev)
            for s in range(0, len(users), self.batch_size):
                k = perm[s : s + self.batch_size]
                pu, qi, qj = P[u_all[k]], Q[i_all[k]], Q[j_all[k]]
                diff = (pu * qi).sum(1) + b[i_all[k]] - (pu * qj).sum(1) - b[j_all[k]]
                penalty = (pu.pow(2).sum(1) + qi.pow(2).sum(1) + qj.pow(2).sum(1) + b[i_all[k]].pow(2) + b[j_all[k]].pow(2)).mean()
                loss = -torch.nn.functional.logsigmoid(diff).mean() + self.reg * penalty
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.detach() * len(k)
            self._sync(P, Q, b)
            val = evaluate(self, data, "val", max_users=2000)["ndcg@10"]
            self.curve_.append({"epoch": epoch, "loss": float(total) / len(users), "val_ndcg@10": val})
            if val > best + 1e-6:
                best, stale = val, 0
                best_state = (P.detach().clone(), Q.detach().clone(), b.detach().clone())
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self._sync(*best_state)
        self.best_epoch_ = int(np.argmax([c["val_ndcg@10"] for c in self.curve_])) + 1

    def score(self, user_idxs: np.ndarray, histories: list[np.ndarray]) -> np.ndarray:
        scores = self.P[user_idxs] @ self.Q.T + self.b[None, :]
        scores[~self.seen[user_idxs]] = self.pop
        return scores.astype(np.float32, copy=False)

    def num_parameters(self) -> int:
        return int(self.P.size + self.Q.size + self.b.size)
