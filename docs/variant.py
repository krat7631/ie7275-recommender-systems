"""Evaluate a saved, already fitted model with changed scoring options, without retraining it.

    RECSYS_DATA_DIR=data/processed_coldstart python3 docs/variant.py results/models/als_cold.pkl foldin foldin=true

The fitted model is loaded, the given attributes (and its recorded config) are overridden, and the usual run_experiment
evaluation is written to results/<model>_<tag>.json. Its training time is the one recorded when the model was fitted.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recsys.data import load  # noqa: E402
from recsys.harness import run_experiment  # noqa: E402
from recsys.models.base import Recommender  # noqa: E402

if __name__ == "__main__":
    path, tag, overrides = sys.argv[1], sys.argv[2], dict(a.split("=", 1) for a in sys.argv[3:])
    model = Recommender.load(path)
    for key, value in overrides.items():
        value = json.loads(value)
        setattr(model, key, value)
        model.config[key] = value
    result = run_experiment(model, load(), tag=tag, fit=False)
    t = result["test"]
    print(f"{result['run_name']}: test recall@10={t['recall@10']:.4f} ndcg@10={t['ndcg@10']:.4f} mrr={t['mrr']:.4f}")
    if result["cold_start"]:
        print("  new users, recall@10 by purchases revealed: " + "  ".join(f"n={n}: {m['recall@10']:.4f}" for n, m in result["cold_start"]["by_revealed"].items()))
