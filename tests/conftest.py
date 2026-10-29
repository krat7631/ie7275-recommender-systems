"""Synthetic raw-format data with planted defects, so each preprocessing rule can be checked exactly."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recsys.coldstart import build_coldstart  # noqa: E402
from recsys.data import load  # noqa: E402
from recsys.preprocess import build  # noqa: E402

PLANTED = {"malformed_json": 5, "missing": 4, "bad_rating": 3, "bad_timestamp": 2, "duplicates": 10}


def make_raw(dirpath: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_users, n_items = 300, 120
    pop = 1.0 / np.arange(1, n_items + 1)
    pop /= pop.sum()
    base_ts = 1_400_000_000_000
    lines = []
    for u in range(n_users):
        k = int(rng.integers(6, 30))
        items = rng.choice(n_items, size=k, replace=False, p=pop)
        times = np.sort(rng.integers(0, 10**10, size=k)) + base_ts
        for it, t in zip(items, times):
            lines.append({"user_id": f"U{u:04d}", "parent_asin": f"I{it:04d}", "rating": float(rng.integers(1, 6)),
                          "timestamp": int(t), "text": "x"})
    # users and items too sparse to survive the 5-core filter
    for u in range(5):
        for it in range(2):
            lines.append({"user_id": f"SPARSEU{u}", "parent_asin": f"I{it:04d}", "rating": 3.0, "timestamp": base_ts + 5, "text": "x"})
    for it in range(4):
        lines.append({"user_id": "U0000", "parent_asin": f"SPARSEI{it}", "rating": 3.0, "timestamp": base_ts + 7, "text": "x"})

    # planted defects
    dups = []
    for d in range(PLANTED["duplicates"]):
        src = lines[d * 7]
        dups.append({**src, "timestamp": src["timestamp"] + 1000, "rating": 1.0})  # later repeat of same pair
    bad = [
        {"user_id": None, "parent_asin": "I0001", "rating": 4.0, "timestamp": base_ts + 1},
        {"user_id": "UX", "parent_asin": "I0001", "rating": None, "timestamp": base_ts + 1},
        {"user_id": "UX", "parent_asin": "I0001", "rating": 4.0, "timestamp": None},
        {"user_id": "UX", "parent_asin": "", "rating": 4.0, "timestamp": base_ts + 1},
        {"user_id": "UX", "parent_asin": "I0001", "rating": 0.0, "timestamp": base_ts + 1},
        {"user_id": "UX", "parent_asin": "I0001", "rating": 7.0, "timestamp": base_ts + 1},
        {"user_id": "UX", "parent_asin": "I0001", "rating": -1.0, "timestamp": base_ts + 1},
        {"user_id": "UX", "parent_asin": "I0001", "rating": 4.0, "timestamp": 5},
        {"user_id": "UX", "parent_asin": "I0001", "rating": 4.0, "timestamp": 99_999_999_999_999},
    ]
    all_lines = [json.dumps(x) for x in lines + dups + bad]
    all_lines += ["not json at all", "{broken", "", "[1, 2]", '{"user_id": "U0"'][: PLANTED["malformed_json"]]
    rng.shuffle(all_lines)
    reviews = dirpath / "reviews.jsonl"
    reviews.write_text("\n".join(all_lines) + "\n")

    meta_lines = []
    for it in range(n_items):
        if it in (3, 4):  # no metadata for these items
            continue
        meta_lines.append(json.dumps({
            "parent_asin": f"I{it:04d}", "title": f"Game {it}", "store": "S", "main_category": "Video Games",
            "categories": ["Video Games", "PC"], "features": ["fast", "fun"], "description": ["a game"],
            "price": ["$12.99", None, "N/A", 5.5][it % 4], "average_rating": 4.2, "rating_number": 9,
        }))
    meta = dirpath / "meta.jsonl"
    meta.write_text("\n".join(meta_lines) + "\n")
    return reviews, meta


@pytest.fixture(scope="session")
def processed(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    reviews, meta = make_raw(d)
    out = d / "processed"
    meta_json = build(reviews, meta, out)
    return {"dir": out, "meta": meta_json, "data": load(out), "raw": (reviews, meta)}


@pytest.fixture(scope="session")
def cold(processed, tmp_path_factory):
    out = tmp_path_factory.mktemp("cold") / "processed_coldstart"
    meta = build_coldstart(processed["dir"], out)
    return {"dir": out, "meta": meta, "data": load(out), "base": processed["data"]}
