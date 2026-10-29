"""SASRec: self-attentive sequential recommendation (Kang and McAuley, 2018), written from scratch in PyTorch.

A user's chronological purchases are the input sequence: item embedding plus a learned positional embedding, a stack of
pre-norm causal self-attention blocks (each position sees only earlier positions), and the hidden state at a position
is dotted with the item embeddings to score every item as the next purchase. Item ids are shifted by +1 so 0 is padding;
sequences are left-padded to max_len and the last max_len items are used.

Training: for every user the chronological items in data.train_history; input is items[:-1] and the target at each
position is the next item, truncated to the last max_len positions. One negative per position is sampled uniformly
among the items the user has no interaction with, and the loss is binary cross-entropy over the non-padded positions
(as in the paper), Adam, dropout. After every epoch the validation NDCG@10 (2,000 fixed users) decides early stopping and
the best epoch's weights are restored. self.curve_ holds {epoch, loss, val_ndcg@10}.

Hyperparameters: max_len (20: the data analysis shows 95% of users have at most 18 items of history and the median is
5), hidden (64), blocks (2), heads (2), dropout (0.2), lr (0.001), batch_size (256), epochs (50), patience (5).

Unseen users: the model works from the history it is given, so a new user with 1 to 3 revealed purchases is scored from
those purchases without any special code. A user with an empty history gets popularity scores (data.item_train_count).

Training time: see the results files (cost.fit_seconds).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recsys.config import SEED, get_device
from recsys.data import SplitData
from recsys.harness import evaluate
from recsys.models.base import Recommender
from recsys.sampling import interaction_keys, sample_negatives


class _Block(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.ln1, self.ln2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.qkv, self.proj = nn.Linear(hidden, 3 * hidden), nn.Linear(hidden, hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        q, k, v = (t.reshape(B, T, self.heads, H // self.heads).transpose(1, 2) for t in self.qkv(self.ln1(x)).chunk(3, dim=-1))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, H)))
        return x + self.drop(self.ffn(self.ln2(x)))


class _SASRec(nn.Module):
    def __init__(self, n_items: int, hidden: int, blocks: int, heads: int, dropout: float, max_len: int):
        super().__init__()
        self.hidden, self.max_len = hidden, max_len
        self.item = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.pos = nn.Embedding(max_len, hidden)
        self.blocks = nn.ModuleList([_Block(hidden, heads, dropout) for _ in range(blocks)])
        self.ln = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        for emb in (self.item, self.pos):
            nn.init.xavier_normal_(emb.weight)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: [B, T] item ids (0 = padding, on the left). Returns [B, T, hidden]."""
        B, T = seq.shape
        real = (seq != 0)[..., None]
        x = self.drop(self.item(seq) * math.sqrt(self.hidden) + self.pos(torch.arange(T, device=seq.device))) * real
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=seq.device))
        # a position may attend to earlier real positions, and always to itself so padded rows never see an empty set
        allowed = (causal[None, None] & (seq != 0)[:, None, None, :]) | torch.eye(T, dtype=torch.bool, device=seq.device)[None, None]
        for block in self.blocks:
            x = block(x, allowed) * real
        return self.ln(x)


class Sasrec(Recommender):
    name = "sasrec"

    def __init__(self, max_len: int = 20, hidden: int = 64, blocks: int = 2, heads: int = 2, dropout: float = 0.2,
                 lr: float = 0.001, batch_size: int = 256, epochs: int = 50, patience: int = 5, **kw):
        super().__init__(max_len=max_len, hidden=hidden, blocks=blocks, heads=heads, dropout=dropout, lr=lr,
                         batch_size=batch_size, epochs=epochs, patience=patience, **kw)
        self.max_len, self.hidden, self.blocks, self.heads, self.dropout = max_len, hidden, blocks, heads, dropout
        self.lr, self.batch_size, self.epochs, self.patience = lr, batch_size, epochs, patience

    def fit(self, data: SplitData) -> None:
        self.dev = get_device()
        rng = np.random.default_rng(SEED)
        T = self.max_len
        self.pop = data.item_train_count.astype(np.float32)
        keys = interaction_keys(data.train["user_idx"].to_numpy(), data.train["item_idx"].to_numpy(), data.n_items)

        uid, X, Y = [], [], []
        for u, seq in enumerate(data.train_history):
            if len(seq) < 2:
                continue
            inp, tgt = seq[:-1][-T:] + 1, seq[1:][-T:] + 1
            x, y = np.zeros(T, dtype=np.int64), np.zeros(T, dtype=np.int64)
            x[T - len(inp):], y[T - len(tgt):] = inp, tgt
            uid.append(u), X.append(x), Y.append(y)
        uid, X, Y = np.array(uid), np.stack(X), np.stack(Y)
        X_t, Y_t = torch.as_tensor(X, device=self.dev), torch.as_tensor(Y, device=self.dev)

        self.net = _SASRec(data.n_items, self.hidden, self.blocks, self.heads, self.dropout, T).to(self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.98))
        self.curve_, best, best_state, stale = [], -1.0, None, 0
        for epoch in range(1, self.epochs + 1):
            neg = sample_negatives(rng, np.repeat(uid, T), data.n_items, keys).reshape(len(uid), T) + 1
            N_t = torch.as_tensor(neg, device=self.dev)
            perm = torch.as_tensor(rng.permutation(len(uid)), device=self.dev)
            self.net.train()
            total, count = torch.zeros((), device=self.dev), torch.zeros((), device=self.dev)
            for s in range(0, len(uid), self.batch_size):
                k = perm[s : s + self.batch_size]
                h = self.net.encode(X_t[k])
                pos_logit = (h * self.net.item(Y_t[k])).sum(-1)
                neg_logit = (h * self.net.item(N_t[k])).sum(-1)
                mask = Y_t[k] != 0
                loss = (-F.logsigmoid(pos_logit) - F.logsigmoid(-neg_logit))[mask].mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, count = total + loss.detach() * mask.sum(), count + mask.sum()
            val = evaluate(self, data, "val", max_users=2000)["ndcg@10"]
            self.curve_.append({"epoch": epoch, "loss": float(total / count), "val_ndcg@10": val})
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
        T = self.max_len
        seq = np.zeros((len(histories), T), dtype=np.int64)
        empty = np.zeros(len(histories), dtype=bool)
        for r, h in enumerate(histories):
            h = h[-T:]
            empty[r] = len(h) == 0
            if len(h):
                seq[r, T - len(h):] = h + 1
        with torch.no_grad():
            last = self.net.encode(torch.as_tensor(seq, device=self.dev))[:, -1, :]
            scores = (last @ self.net.item.weight[1:].T).cpu().numpy()
        scores[empty] = self.pop
        return scores.astype(np.float32, copy=False)

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))
