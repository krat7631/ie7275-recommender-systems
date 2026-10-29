"""The queryable top-10 demonstration: give it a user and it recommends products.

    from recsys.demo import demo, recommend
    recommend(model, data, "AGCI7FAH4GL5FI65HYLKWTMFZ2CQ")     # Amazon user_id or integer user_idx, top 10 as a table
    demo(model, data, 12345)                                    # history, recommendations, and where the real next purchase ranked

The model sees everything known about the user except the held-out test purchase (their train and validation items),
and items they already own are never recommended.
"""
import numpy as np
import pandas as pd

from recsys.data import SplitData
from recsys.models.base import Recommender


def _user_idx(data: SplitData, user) -> int:
    if isinstance(user, (int, np.integer)):
        return int(user)
    hit = data.users.index[data.users["user_id"] == user]
    if len(hit) == 0:
        raise KeyError(f"unknown user {user!r}")
    return int(data.users.loc[hit[0], "user_idx"])


def _titles(data: SplitData, item_idxs) -> pd.DataFrame:
    cols = ["title", "store", "price", "pop_bin"]
    out = data.items.iloc[np.asarray(item_idxs, dtype=int)][cols].reset_index(drop=True)
    out["title"] = out["title"].str.slice(0, 80)
    return out


def recommend(model: Recommender, data: SplitData, user, k: int = 10) -> pd.DataFrame:
    """Top-k recommendations for one user as a table (rank, title, store, price, popularity bin)."""
    u = _user_idx(data, user)
    ids = model.recommend(u, k, data.trainval_history[u])
    table = _titles(data, ids)
    table.insert(0, "rank", np.arange(1, len(ids) + 1))
    return table


def demo(model: Recommender, data: SplitData, user, k: int = 10, show_history: int = 5) -> None:
    """Print what the user bought lately, the model's top-k, and where their actual next purchase would have ranked."""
    u = _user_idx(data, user)
    history = data.trainval_history[u]
    print(f"user {data.users.loc[u, 'user_id']} (user_idx {u}), {len(history)} known purchases; latest {min(show_history, len(history))}:")
    print(_titles(data, history[-show_history:]).to_string(index=False), "\n")
    name = model.name + ("+history" if getattr(model, "foldin_all", False) else "")
    print(f"top {k} recommendations from {name}:")
    print(recommend(model, data, u, k).to_string(index=False), "\n")

    scores = np.array(model.score(np.array([u]), [history]), dtype=np.float32)
    scores[0, history] = -np.inf
    target = int(data.test["item_idx"][u])
    rank = int((scores[0] >= scores[0, target]).sum())
    t = _titles(data, [target]).iloc[0]
    print(f"their actual next purchase: {t['title']!r} ({t['pop_bin']} item)")
    print(f"it ranked #{rank} of {data.n_items:,} items" + ("  -> in the top 10" if rank <= k else ""))
