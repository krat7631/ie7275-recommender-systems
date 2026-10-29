"""Loading the processed dataset. Every model must get its data through load(); never re-read raw files."""
import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.config import DATA_DIR

INTERACTION_DTYPES = {
    "user_idx": "int32",
    "item_idx": "int32",
    "rating": "float32",
    "timestamp": "int64",
}


def fingerprint(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> str:
    """Short hash of the three splits. Stored in meta.json and in every results file."""
    h = hashlib.sha256()
    for part in (train, val, test):
        h.update(part["user_idx"].to_numpy().tobytes())
        h.update(part["item_idx"].to_numpy().tobytes())
    return h.hexdigest()[:12]


def build_histories(frames: list[pd.DataFrame], n_users: int) -> list[np.ndarray]:
    """Per-user chronological item arrays (index = user_idx).

    Frames are concatenated in order, and rows keep their file order within a user, so
    build_histories([train, val]) puts the validation item last for every user.
    """
    df = pd.concat(frames, ignore_index=True)
    users = df["user_idx"].to_numpy()
    items = df["item_idx"].to_numpy()
    order = np.argsort(users, kind="stable")
    counts = np.bincount(users, minlength=n_users)
    return np.split(items[order], np.cumsum(counts)[:-1])


def to_csr(df: pd.DataFrame, n_users: int, n_items: int, values: str = "implicit") -> sp.csr_matrix:
    """User x item sparse matrix with the full catalog shape.

    values='implicit' gives 1.0 per interaction, values='rating' gives the star rating.
    """
    v = np.ones(len(df), dtype=np.float32) if values == "implicit" else df["rating"].to_numpy(np.float32)
    return sp.csr_matrix((v, (df["user_idx"].to_numpy(), df["item_idx"].to_numpy())), shape=(n_users, n_items))


@dataclass(eq=False)
class SplitData:
    train: pd.DataFrame  # user_idx, item_idx, rating, timestamp; sorted by user then time
    val: pd.DataFrame  # exactly one row per user: second-to-last interaction
    test: pd.DataFrame  # exactly one row per user: last interaction
    users: pd.DataFrame  # user_idx, user_id, n_train, activity_bin
    items: pd.DataFrame  # item_idx, parent_asin, title, ..., n_train, pop_bin
    meta: dict
    # Only in the cold-start dataset (data/processed_coldstart): the held-out "new users" and the
    # train-period rows that were removed from `train` for them.
    new_users: np.ndarray | None = None
    new_interactions: pd.DataFrame | None = None

    @property
    def n_users(self) -> int:
        return self.meta["n_users"]

    @property
    def n_items(self) -> int:
        return self.meta["n_items"]

    @property
    def data_version(self) -> str:
        return self.meta["data_version"]

    @cached_property
    def train_history(self) -> list[np.ndarray]:
        """What a model may condition on when predicting the validation item."""
        return build_histories([self.train], self.n_users)

    @cached_property
    def trainval_history(self) -> list[np.ndarray]:
        """What a model may condition on when predicting the test item."""
        return build_histories([self.train, self.val], self.n_users)

    @cached_property
    def item_train_count(self) -> np.ndarray:
        return self.items["n_train"].to_numpy()

    @cached_property
    def eval_users(self) -> np.ndarray:
        """Users scored in the standard val/test evaluation: everyone, except the new users in a cold-start dataset."""
        if self.new_users is None:
            return np.arange(self.n_users)
        return np.setdiff1d(np.arange(self.n_users), self.new_users)

    @cached_property
    def new_user_history(self) -> list[np.ndarray]:
        """Each new user's full chronological sequence before the test item (empty array for everyone else)."""
        if self.new_users is None:
            raise ValueError("this is not a cold-start dataset")
        val_rows = self.val[self.val["user_idx"].isin(self.new_users)]
        return build_histories([self.new_interactions, val_rows], self.n_users)


def _check(name: str, df: pd.DataFrame) -> None:
    for col, dtype in INTERACTION_DTYPES.items():
        if col not in df.columns or str(df[col].dtype) != dtype:
            raise ValueError(f"{name}.parquet: column {col!r} must be {dtype}, got {df.get(col, pd.Series()).dtype}")


def load(data_dir: str | Path | None = None) -> SplitData:
    d = Path(data_dir) if data_dir else DATA_DIR
    meta = json.loads((d / "meta.json").read_text())
    frames = {n: pd.read_parquet(d / f"{n}.parquet") for n in ("train", "val", "test")}
    for n, df in frames.items():
        _check(n, df)

    n_users, n_items = meta["n_users"], meta["n_items"]
    for n in ("val", "test"):
        u = frames[n]["user_idx"].to_numpy()
        if len(u) != n_users or not np.array_equal(u, np.arange(n_users)):
            raise ValueError(f"{n}.parquet must hold exactly one row per user, ordered by user_idx")
    for n, df in frames.items():
        if df["item_idx"].max() >= n_items or df["item_idx"].min() < 0:
            raise ValueError(f"{n}.parquet: item_idx outside [0, n_items)")

    found = fingerprint(frames["train"], frames["val"], frames["test"])
    if found != meta["data_version"]:
        raise ValueError(f"data files do not match meta.json (found {found}, expected {meta['data_version']}); "
                         "restore the data folder from the original copy instead of editing it")

    users = pd.read_parquet(d / "users.parquet")
    items = pd.read_parquet(d / "items.parquet")
    if len(items) != n_items or len(users) != n_users:
        raise ValueError("users.parquet / items.parquet disagree with meta.json")

    new_users = new_inter = None
    if (d / "new_users.parquet").exists():
        new_users = pd.read_parquet(d / "new_users.parquet")["user_idx"].to_numpy()
        new_inter = pd.read_parquet(d / "new_user_interactions.parquet")
        _check("new_user_interactions", new_inter)
        if np.isin(frames["train"]["user_idx"].to_numpy(), new_users).any():
            raise ValueError("new users must have no rows in train.parquet")
    return SplitData(frames["train"], frames["val"], frames["test"], users, items, meta, new_users, new_inter)
