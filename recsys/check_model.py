"""Conformance check for a model file. Run it before you hand a model back.

    python -m recsys.check_model knn

Fits the model on the small cold-start dev dataset (data/processed_dev_coldstart, takes seconds) and checks
the project rules that a machine can verify. Prints PASS / WARN / FAIL and exits with status 1 on any FAIL.
"""
import ast
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from recsys.config import COLD_REVEAL, ROOT, SEED, set_seed
from recsys.data import load
from recsys.harness import evaluate, evaluate_new_users
from recsys.models.base import Recommender
from recsys.models.popularity import Popularity

ALLOWED_IMPORTS = set(sys.stdlib_module_names) | {
    "numpy", "pandas", "scipy", "sklearn", "torch", "pyarrow", "matplotlib", "seaborn", "tqdm", "recsys",
}
Result = tuple[str, str, str]  # (level, check, detail)


def static_checks(path: Path, name: str) -> list[Result]:
    out: list[Result] = []
    tree = ast.parse(path.read_text())

    bad = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods = [node.module.split(".")[0]]
        else:
            continue
        bad |= {m for m in mods if m not in ALLOWED_IMPORTS and not (m == "sentence_transformers" and name == "content")}
    out.append(("FAIL", "imports", f"not allowed: {sorted(bad)}") if bad else ("PASS", "imports", "only allowed packages"))

    touches_test = any(isinstance(n, ast.Attribute) and n.attr == "test" for n in ast.walk(tree))
    out.append(("FAIL", "test split", "uses `.test`; models must never touch the test split") if touches_test
               else ("PASS", "test split", "never touched"))

    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    reads_files = [s for s in strings if any(k in s for k in ("data/raw", "test.parquet", ".jsonl", "processed"))]
    out.append(("FAIL", "data access", f"reads data outside load(): {reads_files[:3]}") if reads_files
               else ("PASS", "data access", "only through load()"))

    devices = [s for s in strings if s in ("cuda", "cpu")]
    if devices:
        out.append(("WARN", "device", "hard-coded 'cuda'/'cpu'; use recsys.config.get_device()"))
    if not ast.get_docstring(tree):
        out.append(("WARN", "docstring", "add a module docstring: algorithm, hyperparameters, training time, unseen-user behaviour"))
    return out


def load_class(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"recsys.models._checked_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    classes = [c for _, c in inspect.getmembers(module, inspect.isclass)
               if issubclass(c, Recommender) and c is not Recommender and c.__module__ == module.__name__]
    if len(classes) != 1:
        raise ValueError(f"{path.name} must define exactly one Recommender subclass, found {len(classes)}")
    return classes[0]


def _scores_ok(label: str, s, n_rows: int, n_items: int) -> list[Result]:
    if not isinstance(s, np.ndarray):
        return [("FAIL", label, "score() must return a numpy array (call .cpu().numpy() on torch tensors)")]
    if s.shape != (n_rows, n_items):
        return [("FAIL", label, f"shape {s.shape}, expected {(n_rows, n_items)}")]
    if not np.isfinite(s).all():
        return [("FAIL", label, "contains NaN or infinity")]
    return []


def dynamic_checks(cls, data) -> tuple[list[Result], dict]:
    out: list[Result] = []
    info: dict = {}
    set_seed(SEED)
    model = cls()
    try:
        json.dumps(model.config)
        out.append(("PASS", "config", "JSON-serialisable"))
    except TypeError:
        out.append(("FAIL", "config", "model.config must be JSON-serialisable (hyperparameters only)"))

    t0 = time.perf_counter()
    try:
        model.fit(data)
    except Exception as e:  # noqa: BLE001
        return out + [("FAIL", "fit", f"crashed: {type(e).__name__}: {e}")], info
    info["fit_seconds"] = round(time.perf_counter() - t0, 2)

    warm = data.eval_users[:64]
    hist = [data.train_history[u] for u in warm]
    hist_copy = [h.copy() for h in hist]
    try:
        s = model.score(warm, hist)
    except Exception as e:  # noqa: BLE001
        return out + [("FAIL", "score (known users)", f"crashed: {type(e).__name__}: {e}")], info
    problems = _scores_ok("score (known users)", s, len(warm), data.n_items)
    if problems:
        return out + problems, info
    out.append(("PASS", "score (known users)", f"shape {s.shape}, finite"))

    out.append(("PASS", "not constant", "scores differ across items") if (s.std(axis=1) > 0).mean() > 0.9
               else ("FAIL", "not constant", "equal scores for most users; ties get the worst rank"))
    again = model.score(warm, hist)
    out.append(("PASS", "deterministic", "same scores on a repeat call") if np.allclose(s, again, atol=1e-5)
               else ("FAIL", "deterministic", "scores change between calls (dropout still on? call model.eval() / torch.no_grad())"))
    single = model.score(warm[:1], hist[:1])
    out.append(("PASS", "batch-independent", "one user alone scores like in a batch") if np.allclose(s[:1], single, rtol=1e-3, atol=1e-4)
               else ("FAIL", "batch-independent", "a user's scores depend on who else is in the batch"))
    out.append(("PASS", "inputs untouched", "histories not modified") if all(np.array_equal(a, b) for a, b in zip(hist, hist_copy))
               else ("FAIL", "inputs untouched", "score() modified the histories it was given"))

    new = data.new_users[:32]
    pop = data.item_train_count.astype(np.float64)
    for n in COLD_REVEAL:
        nh = [data.new_user_history[u][:n] for u in new]
        try:
            sn = model.score(new, nh)
        except Exception as e:  # noqa: BLE001 - any crash on unseen users is a failure
            out.append(("FAIL", f"new users, n={n}", f"crashed: {type(e).__name__}: {e}"))
            continue
        bad = _scores_ok(f"new users, n={n}", sn, len(new), data.n_items)
        if bad:
            out += bad
            continue
        if n == 0:
            if (sn.std(axis=1) == 0).any():
                out.append(("FAIL", "new users, n=0", "constant scores for a user with no history; return popularity (data.item_train_count)"))
            else:
                rho = float(np.nanmean([spearmanr(sn[i], pop)[0] for i in range(len(new))]))
                out.append(("PASS", "new users, n=0", f"popularity-like (rank correlation {rho:.2f})") if rho >= 0.9
                           else ("WARN", "new users, n=0", f"rank correlation with popularity is {rho:.2f}; the project rules ask for popularity here"))
        else:
            out.append(("PASS", f"new users, n={n}", "scores returned"))

    if model.supports_rating:
        pred = np.asarray(model.predict(warm, data.val["item_idx"].to_numpy()[warm]))
        out.append(("PASS", "predict", "finite, one rating per pair") if pred.shape == (len(warm),) and np.isfinite(pred).all()
                   else ("FAIL", "predict", "must return one finite rating per (user, item) pair"))

    try:
        val = evaluate(model, data, "val", max_users=300)
        for n in COLD_REVEAL:
            evaluate_new_users(model, data, n, max_users=100)
    except Exception as e:  # noqa: BLE001
        return out + [("FAIL", "harness", f"evaluate() crashed: {type(e).__name__}: {e}")], info
    base = Popularity()
    base.fit(data)
    base_val = evaluate(base, data, "val", max_users=300)
    info["val_ndcg@10"], info["popularity_val_ndcg@10"] = round(val["ndcg@10"], 4), round(base_val["ndcg@10"], 4)
    out.append(("PASS", "harness", "evaluate() and evaluate_new_users() run end to end"))
    out.append(("PASS", "beats popularity", f"val NDCG@10 {info['val_ndcg@10']} vs {info['popularity_val_ndcg@10']}")
               if val["ndcg@10"] > base_val["ndcg@10"]
               else ("WARN", "beats popularity", f"val NDCG@10 {info['val_ndcg@10']} vs popularity {info['popularity_val_ndcg@10']} "
                                                  "on the small dev data; check again on the full data"))
    return out, info


def run_checks(path: Path, data_dir: Path | None = None) -> list[Result]:
    path = Path(path)
    name = path.stem
    results = static_checks(path, name)
    try:
        cls = load_class(path, name)
    except Exception as e:  # noqa: BLE001
        return results + [("FAIL", "import", f"{type(e).__name__}: {e}")]
    if cls.name != name:
        return results + [("FAIL", "name", f"{cls.__name__}.name is {cls.name!r} but the file is {name}.py")]
    results.append(("PASS", "name", f"{cls.__name__}, name = {name!r}"))
    try:
        dyn, info = dynamic_checks(cls, load(data_dir or ROOT / "data" / "processed_dev_coldstart"))
    except Exception as e:  # noqa: BLE001
        return results + [("FAIL", "run", f"{type(e).__name__}: {e}")]
    return results + dyn + ([("INFO", "fit time (dev data)", f"{info['fit_seconds']} s")] if "fit_seconds" in info else [])


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m recsys.check_model <model name, e.g. knn>   |   --all")
    from recsys.handoff import OWNERS

    names = ([m for o in OWNERS.values() for m in o["models"] if (ROOT / "recsys" / "models" / f"{m}.py").exists()]
             if sys.argv[1] == "--all" else [sys.argv[1]])
    total_fails = 0
    for name in names:
        results = run_checks(ROOT / "recsys" / "models" / f"{name}.py")
        if len(names) > 1:
            print(f"--- {name}")
        for level, check, detail in results:
            if len(names) == 1 or level in ("FAIL", "WARN", "INFO"):
                print(f"{level:4s}  {check:22s} {detail}")
        fails = sum(r[0] == "FAIL" for r in results)
        warns = sum(r[0] == "WARN" for r in results)
        total_fails += fails
        print(f"{'FAILED' if fails else 'OK'}: {fails} failure(s), {warns} warning(s)" + ("\n" if len(names) > 1 else ""))
    raise SystemExit(1 if total_fails else 0)


if __name__ == "__main__":
    main()
