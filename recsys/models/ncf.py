"""Neural collaborative filtering, NeuMF (He et al., 2017), written from scratch in PyTorch.

Two towers share one prediction: a GMF tower (elementwise product of a user and an item embedding) and an MLP tower
(concatenated user and item embeddings through ReLU hidden layers), joined by one final linear layer to a single logit.
Trained on implicit feedback: every training interaction is a positive, `neg` negatives per positive are sampled each
epoch uniformly among the items the user has no interaction with, binary cross-entropy with logits, Adam. After every
epoch the validation NDCG@10 (2,000 fixed users) decides early stopping and the best epoch's weights are restored.
self.curve_ holds {epoch, loss, val_ndcg@10}.

Hyperparameters: factors (32, GMF embedding size), mlp_factors (32), layers ([64, 32, 16]), lr (0.001), neg (4),
batch_size (4096), epochs (30), patience (3), reg (0, L2 penalty on the embeddings of the rows in each batch) and
dropout (0, applied after each MLP hidden layer while training). With 7.7M parameters and 625k interactions the
unregularised network memorises the training data within a few epochs, so reg and dropout are what make it generalise.

Scoring all 25,612 items for a batch of users is done in item chunks. The GMF part is one matrix product and the first
MLP layer is split into a user part and an item part (W [u; i] = W_u u + W_i i), so the [users x items x hidden] tensor is
only ever built one chunk at a time (about 250 MB at most) instead of many GB.

Unseen users (no training interactions) have no learned embeddings, so score() returns popularity
(data.item_train_count) for them; the model cannot use their revealed purchases.

Training time and the time of a full evaluation pass: see the results files (cost.fit_seconds, cost.score_ms_per_user).
"""
import numpy as np
import torch
from torch import nn

from recsys.config import SEED, get_device
from recsys.data import SplitData
from recsys.harness import evaluate
from recsys.models.base import Recommender
from recsys.sampling import interaction_keys, sample_negatives


class _NeuMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, factors: int, mlp_factors: int, layers: list[int], dropout: float = 0.0):
        super().__init__()
        self.fg, self.fm = factors, mlp_factors
        self.drop = nn.Dropout(dropout)
        self.Ug, self.Ig = nn.Embedding(n_users, factors), nn.Embedding(n_items, factors)
        self.Um, self.Im = nn.Embedding(n_users, mlp_factors), nn.Embedding(n_items, mlp_factors)
        dims = [2 * mlp_factors] + list(layers)
        self.mlp = nn.ModuleList([nn.Linear(dims[k], dims[k + 1]) for k in range(len(layers))])
        self.out = nn.Linear(factors + dims[-1], 1)
        for emb in (self.Ug, self.Ig, self.Um, self.Im):
            nn.init.normal_(emb.weight, std=0.05)

    def forward(self, u: torch.Tensor, i: torch.Tensor, with_penalty: bool = False):
        ug, ig, um, im = self.Ug(u), self.Ig(i), self.Um(u), self.Im(i)
        h = torch.cat([um, im], dim=-1)
        for lin in self.mlp:
            h = self.drop(torch.relu(lin(h)))
        logit = self.out(torch.cat([ug * ig, h], dim=-1)).squeeze(-1)
        if with_penalty:
            return logit, (ug.pow(2).sum(-1) + ig.pow(2).sum(-1) + um.pow(2).sum(-1) + im.pow(2).sum(-1)).mean()
        return logit

    @torch.no_grad()
    def score_all(self, u: torch.Tensor, cap: int = 64_000_000) -> torch.Tensor:
        """Logits of every item for each user in u, shape [len(u), n_items], without building the full pair tensor."""
        wg, wm = self.out.weight[0, : self.fg], self.out.weight[0, self.fg :]
        gmf = (self.Ug(u) * wg) @ self.Ig.weight.T
        W1 = self.mlp[0].weight
        a = self.Um(u) @ W1[:, : self.fm].T  # [B, h1] user part of the first layer
        b = self.Im.weight @ W1[:, self.fm :].T + self.mlp[0].bias  # [N, h1] item part
        chunk = max(1, cap // (a.shape[0] * a.shape[1]))
        parts = []
        for s in range(0, b.shape[0], chunk):
            h = torch.relu(a[:, None, :] + b[None, s : s + chunk, :])
            for lin in self.mlp[1:]:
                h = torch.relu(lin(h))
            parts.append(h @ wm)
        return gmf + torch.cat(parts, dim=1) + self.out.bias


class Ncf(Recommender):
    name = "ncf"

    def __init__(self, factors: int = 32, mlp_factors: int = 32, layers: list | tuple = (64, 32, 16), lr: float = 0.001,
                 neg: int = 4, batch_size: int = 4096, epochs: int = 30, patience: int = 3, reg: float = 0.0,
                 dropout: float = 0.0, **kw):
        layers = list(layers)
        super().__init__(factors=factors, mlp_factors=mlp_factors, layers=layers, lr=lr, neg=neg, batch_size=batch_size,
                         epochs=epochs, patience=patience, reg=reg, dropout=dropout, **kw)
        self.factors, self.mlp_factors, self.layers, self.lr, self.neg = factors, mlp_factors, layers, lr, neg
        self.batch_size, self.epochs, self.patience, self.reg, self.dropout = batch_size, epochs, patience, reg, dropout

    def fit(self, data: SplitData) -> None:
        self.dev = get_device()
        rng = np.random.default_rng(SEED)
        users = data.train["user_idx"].to_numpy().astype(np.int64)
        items = data.train["item_idx"].to_numpy().astype(np.int64)
        keys = interaction_keys(users, items, data.n_items)
        self.seen = np.bincount(users, minlength=data.n_users) > 0
        self.pop = data.item_train_count.astype(np.float32)

        self.net = _NeuMF(data.n_users, data.n_items, self.factors, self.mlp_factors, self.layers, self.dropout).to(self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        bce = nn.BCEWithLogitsLoss()
        n = len(users)
        labels = torch.as_tensor(np.r_[np.ones(n), np.zeros(n * self.neg)], dtype=torch.float32, device=self.dev)
        u_all = torch.as_tensor(np.r_[users, np.tile(users, self.neg)], device=self.dev)

        self.curve_, best, best_state, stale = [], -1.0, None, 0
        for epoch in range(1, self.epochs + 1):
            neg_items = sample_negatives(rng, np.tile(users, self.neg), data.n_items, keys)
            i_all = torch.as_tensor(np.r_[items, neg_items], device=self.dev)
            perm = torch.as_tensor(rng.permutation(len(labels)), device=self.dev)
            self.net.train()
            total = torch.zeros((), device=self.dev)
            for s in range(0, len(labels), self.batch_size):
                k = perm[s : s + self.batch_size]
                logit, penalty = self.net(u_all[k], i_all[k], with_penalty=True)
                loss = bce(logit, labels[k]) + self.reg * penalty
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.detach() * len(k)
            val = evaluate(self, data, "val", max_users=2000)["ndcg@10"]
            self.curve_.append({"epoch": epoch, "loss": float(total) / len(labels), "val_ndcg@10": val})
            if val > best + 1e-6:
                best, stale = val, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_epoch_ = int(np.argmax([c["val_ndcg@10"] for c in self.curve_])) + 1

    def score(self, user_idxs: np.ndarray, histories: list[np.ndarray]) -> np.ndarray:
        self.net.eval()
        scores = self.net.score_all(torch.as_tensor(user_idxs, device=self.dev)).cpu().numpy()
        scores[~self.seen[user_idxs]] = self.pop
        return scores.astype(np.float32, copy=False)

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))
