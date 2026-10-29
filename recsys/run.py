"""Run one model through the shared harness.

    python -m recsys.run popularity
    python -m recsys.run knn --set k=50 --tag k50
    python -m recsys.run knn --set k=50 --tag final     # the settings you choose; this run goes in the report

Loads recsys/models/<model>.py, instantiates its Recommender subclass with the --set values as
hyperparameters, fits, evaluates on val and test, and writes results/<model>[_<tag>].json.
"""
import argparse
import importlib
import inspect
import json
import re

from recsys.data import load
from recsys.harness import run_experiment
from recsys.models.base import Recommender


def _parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def _find_class(module) -> type[Recommender]:
    classes = [
        c
        for _, c in inspect.getmembers(module, inspect.isclass)
        if issubclass(c, Recommender) and c is not Recommender and c.__module__ == module.__name__
    ]
    if len(classes) != 1:
        raise SystemExit(f"{module.__name__} must define exactly one Recommender subclass, found {len(classes)}")
    return classes[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="file name in recsys/models/, without .py")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="hyperparameters")
    ap.add_argument("--tag", default="", help="suffix for the results file, e.g. a hyperparameter setting")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]*", args.tag):
        raise SystemExit("--tag may only contain letters, digits, '.', '_' and '-'")

    cls = _find_class(importlib.import_module(f"recsys.models.{args.model}"))
    if cls.name != args.model:
        raise SystemExit(f"{cls.__name__}.name is {cls.name!r} but the file is {args.model}.py; they must match")
    config = {k: _parse_value(v) for k, v in (s.split("=", 1) for s in args.set)}

    data = load()  # which dataset: set RECSYS_DATA_DIR, see README.md
    print(f"data {data.data_version}: {data.n_users} users, {data.n_items} items, {len(data.train)} train rows")
    result = run_experiment(cls(**config), data, tag=args.tag)

    t, c = result["test"], result["cost"]
    print(f"{result['run_name']}: test recall@10={t['recall@10']:.4f} ndcg@10={t['ndcg@10']:.4f} "
          f"mrr={t['mrr']:.4f} rmse={t['rmse']} | fit {c['fit_seconds']}s, {c['score_ms_per_user']} ms/user")
    if result["cold_start"]:
        by = result["cold_start"]["by_revealed"]
        print("  new users, recall@10 by purchases revealed: " + "  ".join(f"n={n}: {m['recall@10']:.4f}" for n, m in by.items()))


if __name__ == "__main__":
    main()
