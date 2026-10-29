"""Content-based hybrid recommender on item-text embeddings.

fit(): every item's text (data.items["text"]: title, features and description) is embedded with the sentence-transformers
model all-MiniLM-L6-v2 on recsys.config.get_device(), L2-normalised, and cached in data/cache/ (the file name carries a
hash of the item list, so the full, cold-start and dev datasets, which index items differently, never share a cache).
Nothing is trained: the encoder is frozen.

score(): a user's profile is the mean embedding of the items in the history the model is given; the score of an item is
cosine(profile, item embedding) + lam * prior, where prior = log1p(item train count) / its maximum, a mild popularity
prior. Because the score is built from item text, items that never appeared in training can be recommended, which
collaborative models cannot do.

Hyperparameter: lam (0.1; lam = 0 is pure content similarity).

Unseen users: the model works from the history it is given, so a new user with 1 to 3 revealed purchases is scored from
those purchases. A user with an empty history gets popularity scores (data.item_train_count).

Encoder: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions). Embedding time: see self.embed_seconds_ (computed
once, then loaded from the cache).
"""
import hashlib
import time

import numpy as np

from recsys.config import ROOT, get_device
from recsys.data import SplitData
from recsys.models.base import Recommender

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def item_embeddings(data: SplitData) -> tuple[np.ndarray, float]:
    key = hashlib.sha1("|".join(data.items["parent_asin"]).encode()).hexdigest()[:10]
    cache = ROOT / "data" / "cache" / f"item_emb_minilm_{key}.npy"
    if cache.exists():
        return np.load(cache), 0.0
    from sentence_transformers import SentenceTransformer  # only needed the first time

    t0 = time.perf_counter()
    encoder = SentenceTransformer(ENCODER, device=get_device())
    emb = encoder.encode(data.items["text"].tolist(), batch_size=128, normalize_embeddings=True,
                         show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    seconds = time.perf_counter() - t0
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    return emb, seconds


class Content(Recommender):
    name = "content"

    def __init__(self, lam: float = 0.1, **kw):
        super().__init__(lam=lam, **kw)
        self.lam = lam

    def fit(self, data: SplitData) -> None:
        self.emb, self.embed_seconds_ = item_embeddings(data)
        self.pop = data.item_train_count.astype(np.float32)
        logp = np.log1p(self.pop)
        self.prior = (logp / max(float(logp.max()), 1e-9)).astype(np.float32)

    def score(self, user_idxs: np.ndarray, histories: list[np.ndarray]) -> np.ndarray:
        profile = np.zeros((len(histories), self.emb.shape[1]), dtype=np.float32)
        empty = np.zeros(len(histories), dtype=bool)
        for r, h in enumerate(histories):
            if len(h):
                profile[r] = self.emb[h].mean(axis=0)
            else:
                empty[r] = True
        profile /= np.linalg.norm(profile, axis=1, keepdims=True) + 1e-9
        scores = profile @ self.emb.T + self.lam * self.prior[None, :]
        scores[empty] = self.pop
        return scores.astype(np.float32, copy=False)

    def num_parameters(self) -> int:
        return 0  # frozen encoder, nothing is trained
