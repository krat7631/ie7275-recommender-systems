"""Assemble the deliverable notebooks from one set of tagged cells.

    python3 docs/make_final_notebook.py

Writes IE7275_Project1_Final.ipynb (the single self-contained deliverable) and notebooks/02_classical_models.ipynb,
03_neural_and_sequential.ipynb and 04_slice_and_cold_start_analysis.ipynb (shorter notebooks that import the package).

The notebook contains ALL the code: its first cells write every module of the recsys package to disk (%%writefile), so it
runs in a folder that holds only the notebook, data/ and results/. By default it reads the saved results (RETRAIN = False);
set RETRAIN = True to refit every final model. Interpretations come from docs/final_findings.json, written after the results.
"""
import json
import pprint
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "IE7275_Project1_Final.ipynb"
FINDINGS = json.loads((ROOT / "docs" / "final_findings.json").read_text()) if (ROOT / "docs" / "final_findings.json").exists() else {}

MODULES = ["__init__.py", "config.py", "data.py", "metrics.py", "sampling.py", "models/__init__.py", "models/base.py", "harness.py",
           "preprocess.py", "coldstart.py", "report.py", "plots.py", "analysis.py", "demo.py", "models/popularity.py", "models/knn.py",
           "models/als.py", "models/svd.py", "models/bpr.py", "models/ncf.py", "models/sasrec.py", "models/content.py"]

IDEAS = {
    "popularity": "Recommends the items with the most training interactions to everybody. No personalisation: the baseline every other model has to beat. "
                  "For rating prediction it uses a shrunk item mean, (sum of item ratings + m * global mean) / (count + m) with m = 10 chosen on validation.",
    "knn": "Item-item collaborative filtering. Two items are similar if the same users interacted with both (cosine similarity on the implicit matrix); "
           "a user's score for an item is its summed similarity to the items in their history. Only the top k neighbours per item are stored.",
    "als": "Implicit-feedback matrix factorization (Hu, Koren and Volinsky, 2008). Every user and item gets a short vector fitted by alternating closed-form least squares, "
           "with confidence 1 + alpha on observed interactions. It cannot score users it never saw in training; the optional fold-in computes a vector from their revealed purchases.",
    "svd": "Biased matrix factorization for rating prediction: mu + user bias + item bias + user vector . item vector, trained with Adam on squared error. "
           "It is the model for the RMSE metric; for the ranking metrics it ranks items by predicted rating.",
    "bpr": "Bayesian personalised ranking matrix factorization (Rendle et al., 2009). Trained directly to rank: an item the user interacted with should score above a sampled item they did not.",
    "ncf": "Neural collaborative filtering, NeuMF (He et al., 2017). A GMF tower (elementwise product of embeddings) and an MLP tower (concatenated embeddings through hidden layers) "
           "are joined by one linear layer. Trained with sampled negatives and binary cross-entropy.",
    "sasrec": "Self-attentive sequential recommendation (Kang and McAuley, 2018): a transformer over the user's chronological purchases that predicts the next item. "
              "It scores users from the history it is given, so it handles new users with a few revealed purchases.",
    "content": "Content-based hybrid. Every item's text (title, features, description) is embedded with a frozen sentence-transformers encoder (all-MiniLM-L6-v2); "
               "a user's profile is the mean embedding of their history, plus a mild popularity prior. It can recommend items nobody has bought yet.",
}
ORDER = ["popularity", "knn", "als", "svd", "bpr", "ncf", "sasrec", "content"]

cells, tags = [], []


def md(s, tag=None):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))
    tags.append(tag)


def code(s, tag=None):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))
    tags.append(tag)


def finding(key, default="", tag=None):
    text = FINDINGS.get(key, default)
    if text:
        md(text, tag)


def shared_setup(root_expr: str) -> str:
    """Imports, plot style and data loading, identical in every notebook except where the project root is."""
    return f'''
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from recsys.config import RESULTS_DIR, COLD_REVEAL
from recsys.data import load

ORDER = {ORDER!r}   # models in the order of the comparison
mpl.rcParams.update({{"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10}})
BLUE, GRAY, ORANGE = "#2a6fb0", "#8a8f98", "#d9822b"
ROOT = {root_expr}
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"eda_{{name}}.png", dpi=150)
    plt.show()

data = load()
cold = load(ROOT / "data" / "processed_coldstart")
log = data.meta["preprocess_log"]
train, val, test, users, items = data.train, data.val, data.test, data.users, data.items
allx = pd.concat([train, val, test], ignore_index=True)
allx["year"] = pd.to_datetime(allx["timestamp"], unit="ms").dt.year
print(data.meta["dataset"], "| data version", data.data_version, "| cold-start version", cold.data_version)
'''


# ----------------------------------------------------------------------------- title and switches
md("""
# IE7275 Group Project 1: Personalized product recommendation on Amazon Reviews 2023 (Video Games)
**Devansh Nayak, Rishi Hirpara, Venkatashivasai Muppidi**

We build and compare eight recommenders, from a popularity baseline to a transformer, on one dataset with one shared evaluation protocol:
popularity, item-item KNN, ALS and BPR matrix factorization, SVD (rating prediction), neural collaborative filtering (NeuMF), SASRec, and a content-based hybrid.
One scoring variant of the trained ALS, `als+history`, is reported next to it (see the ALS section for why).

**This notebook is self-contained.** The first cells write the whole `recsys` package (data loading, metrics, evaluation harness, every model) to disk, so it needs only
this file and the `data/` and `results/` folders next to it. All numbers below are read from the saved result files produced by those same models, unless `RETRAIN = True`.
""")
code('''
# Switches
RETRAIN = False         # False: read the saved results in results/.  True: refit every final model (roughly an hour on a laptop).
RUN_PREPROCESS = False  # True: rebuild data/processed from the raw Amazon files in data/raw (a 3 GB download; see the data section).

import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
os.makedirs("recsys/models", exist_ok=True)
''')

md("## 0. The source code\nEach cell below writes one module of the package. They are shown in dependency order and are the exact code that produced every result in this notebook.")
for rel in MODULES:
    body = (ROOT / "recsys" / rel).read_text()
    code(f"%%writefile recsys/{rel}\n{body}")

code(shared_setup("Path.cwd()"))

# ----------------------------------------------------------------------------- data preparation and EDA, reused from notebook 01
md("# Part 1. Data preparation and exploratory analysis")
md("""
`RUN_PREPROCESS` rebuilds the processed data from the raw files with the `recsys.preprocess` and `recsys.coldstart` modules written above. The raw files are
`raw/review_categories/Video_Games.jsonl` (2.7 GB) and `raw/meta_categories/meta_Video_Games.jsonl` (0.4 GB) from `McAuley-Lab/Amazon-Reviews-2023` on Hugging Face.
""")
code('''
if RUN_PREPROCESS:
    from recsys.preprocess import build
    from recsys.coldstart import build_coldstart
    build("data/raw/Video_Games.jsonl", "data/raw/meta_Video_Games.jsonl", "data/processed")
    build_coldstart("data/processed", "data/processed_coldstart")
else:
    print("using the processed data already in data/ (set RUN_PREPROCESS = True to rebuild it from the raw files)")
''')
eda = nbf.read(str(ROOT / "notebooks" / "01_data_and_eda.ipynb"), 4)
start = next(i for i, c in enumerate(eda.cells) if c.cell_type == "markdown" and c.source.startswith("## 1. Dataset description"))
for c in eda.cells[start:]:
    src = c.source.replace("Path.cwd().parent / ", "Path.cwd() / ").replace("Path.cwd().parent", "Path.cwd()")
    src = src.replace("## 5. Findings", "## Findings of the exploratory analysis")
    cells.append(nbf.v4.new_markdown_cell(src) if c.cell_type == "markdown" else nbf.v4.new_code_cell(src))
    tags.append(None)  # keep tags aligned with cells: the part notebooks select cells by tag

# ----------------------------------------------------------------------------- evaluation protocol
md("# Part 2. The evaluation protocol")
md("""
Every model is scored by the same code (`recsys.metrics`, `recsys.harness`), which is what makes the comparison fair.
- **Split:** per user, leave-one-out by time. The last interaction is the test item, the second-to-last the validation item, the rest is training.
- **Ranking:** each model scores **all 25,612 items** for each user (no sampled negatives); items the user already has are removed; we record the rank of the held-out item. Full ranking is deliberate: sampled-negative evaluation is known to distort model rankings (Krichene and Rendle, 2020), so the proposal commits to ranking the whole catalog.
- **Ties:** rank = 1 + the number of other items scoring at least as high as the target, so a model that returns equal scores for everything gets the worst possible rank.
- **Metrics:** Precision, Recall, F1 and NDCG at k = 5, 10 and 20, MRR, and RMSE/MAE for the rating model. With one relevant item, recall@k is the hit rate and precision and F1 are functions of it, so Recall@10, NDCG@10 and MRR are the headline numbers.
- **Beyond accuracy:** catalog coverage, novelty, average popularity of recommended items and the Gini index of item exposure; **cost:** training time, scoring time per user, parameter count.
- **Selection:** every hyperparameter is chosen on the validation split only (NDCG@10; lowest RMSE for the rating model). The test split is never used inside any model.
""")
code('''
from recsys.metrics import ranking_metrics, target_ranks
print("three users whose held-out item ranked 1st, 3rd and 11th:")
print({k: round(v, 4) for k, v in ranking_metrics(np.array([1, 3, 11]), ks=(10,)).items()})
print("a model that scores every item the same gets the worst rank, not a lucky one:",
      int(target_ranks(np.zeros((1, 5), dtype=np.float32), np.array([2]))[0]), "out of 5")
''')

# ----------------------------------------------------------------------------- models
md("# Part 3. Models, hyperparameter sweeps and training curves")
md("Each section gives the idea, the validation-chosen sweep (all runs read from `results/`, the winner marked) and the training curve of the final run. The code of every model is in Part 0.")
code('''
from recsys.report import sweep_table

def curve_plot(model):
    r = json.load(open(RESULTS_DIR / f"{model}_final.json"))
    if not r.get("curve"):
        print(f"{model} has no iterative training curve")
        return
    c = pd.DataFrame(r["curve"])
    xcol = "epoch" if "epoch" in c else "iteration"
    ycols = [k for k in c.columns if k != xcol]
    fig, axes = plt.subplots(1, len(ycols), figsize=(4.6 * len(ycols), 3.4))
    for ax, y in zip(np.atleast_1d(axes), ycols):
        ax.plot(c[xcol], c[y], color=BLUE, lw=2, marker="o", ms=3)
        if r.get("best_epoch"):
            ax.axvline(r["best_epoch"], color=ORANGE, ls="--", label=f"best epoch {r['best_epoch']}")
            ax.legend(frameon=False)
        ax.set(xlabel=xcol, ylabel=y, title=f"{model}: {y}")
    plt.tight_layout(); plt.savefig(FIG / f"curve_{model}.png", dpi=150); plt.show()
''', "helpers")
for name in ORDER:
    md(f"## {name}\n{IDEAS[name]}", f"model:{name}")
    by = ", by='val_rmse'" if name == "svd" else ""
    cols = '["run"] + [c for c in sw.columns if c.startswith("cfg_")] + ["val_ndcg@10"' + (', "val_rmse"' if name in ("svd", "popularity") else "") + ', "test_recall@10", "fit_seconds", "selected"]'
    code(f'''
sw = sweep_table("{name}"{by})
print("{name}: {{}} runs, winner on validation: {{}}".format(len(sw), sw.loc[sw["selected"], "run"].iloc[0]))
sw[{cols}].head(12)
''', f"model:{name}")
    if name not in ("popularity", "knn", "content"):
        code(f'curve_plot("{name}")', f"model:{name}")
    if name == "als":
        code('''
rows = []
for run, label in (("als_final", "als: learned user vectors"), ("als_foldall", "als+history: user vectors computed from the history given")):
    r = json.load(open(RESULTS_DIR / f"{run}.json"))
    rows.append({"variant": label, "val_ndcg@10": r["val"]["ndcg@10"], "test_recall@10": r["test"]["recall@10"],
                 "test_ndcg@10": r["test"]["ndcg@10"], "test_mrr": r["test"]["mrr"]})
pd.DataFrame(rows).round(4)
''', "model:als")
    finding(f"model_{name}", tag=f"model:{name}")

md("## Final settings, and optional retraining")
fin = {}
for name in ORDER:
    p = ROOT / "results" / f"{name}_final.json"
    if p.exists():
        fin[name] = json.loads(p.read_text())["config"]
code(f'''
FINAL_CONFIGS = {pprint.pformat(fin, width=110, sort_dicts=False)}

def run_final(name, cfg, data_obj, results_dir):
    import importlib
    import recsys.harness as harness
    from recsys.harness import run_experiment
    from recsys.models.base import Recommender
    module = importlib.import_module(f"recsys.models.{{name}}")
    cls = next(c for _, c in vars(module).items() if isinstance(c, type) and issubclass(c, Recommender) and c is not Recommender and c.__module__ == module.__name__)
    harness.RESULTS_DIR = results_dir
    model = cls(**cfg)
    run_experiment(model, data_obj, tag="final")
    return model

if RETRAIN:   # refit every final model on the full data and on the cold-start data (writes results/*_final.json)
    import recsys.harness as harness
    from recsys.harness import run_experiment
    cold_data = load(Path("data") / "processed_coldstart")
    for name, cfg in FINAL_CONFIGS.items():
        t0 = time.perf_counter()
        model = run_final(name, cfg, data, RESULTS_DIR)
        if name == "als":     # the same fitted ALS, every user scored from the history given (no retraining)
            model.foldin_all = True; model.config["foldin_all"] = True
            run_experiment(model, data, tag="foldall", fit=False)
        model = run_final(name, cfg, cold_data, RESULTS_DIR / "processed_coldstart")
        if name == "als":     # fold the revealed purchases of new users into the fitted ALS
            model.foldin = True; model.config["foldin"] = True
            run_experiment(model, cold_data, tag="foldin", fit=False)
        print(name, "done in", round(time.perf_counter() - t0), "s")
    harness.RESULTS_DIR = RESULTS_DIR
else:
    print("RETRAIN = False: the tables and charts below read the saved final runs for", sorted(FINAL_CONFIGS))
''')

# ----------------------------------------------------------------------------- results
md("# Part 4. Results")
finding("results_intro")
code('''
from recsys.analysis import comparison_table, paired_p_values, cost_table
from recsys.plots import comparison_chart, cost_scatter, tradeoff_scatter

table = comparison_table(RESULTS_DIR, metrics=("recall@10", "ndcg@10", "mrr"), n_boot=2000)
shown = table.copy()
for m in ("recall@10", "ndcg@10", "mrr"):
    shown[m] = [f"{v:.4f} [{lo:.4f}, {hi:.4f}]" for v, lo, hi in zip(table[m], table[f"{m}_lo"], table[f"{m}_hi"])]
shown[["model", "recall@10", "ndcg@10", "mrr"]]
''')
code('''
ax = comparison_chart(table, "ndcg@10"); ax.figure.tight_layout(); ax.figure.savefig(FIG / "results_ndcg.png", dpi=150); plt.show()
''')
md("**Is one model really better than another?** Paired bootstrap over users (the same resampled users for every model). Below: NDCG@10 difference (row minus column) and the two-sided p-value.")
code('''
diff, pval = paired_p_values(RESULTS_DIR, "ndcg@10", n_boot=2000)
print("difference in NDCG@10, row minus column"); display(diff.round(4))
print("two-sided bootstrap p-value (0.0000 means no resample reversed the order)"); display(pval.round(4))
''')
finding("results_significance")
code('''
cost = cost_table(RESULTS_DIR)
cost[["model", "recall@10", "ndcg@10", "fit_seconds", "score_ms_per_user", "n_params", "coverage@10", "novelty@10", "avg_pop@10", "gini@10"]]
''')
code('''
fig = cost_scatter(cost); fig.savefig(FIG / "results_cost.png", dpi=150); plt.show()
fig = tradeoff_scatter(cost); fig.savefig(FIG / "results_tradeoff.png", dpi=150); plt.show()
''')
finding("results_tradeoffs")
md("### Back to the proposal: does model complexity pay off?")
finding("complexity")
finding("popularity_expectation")
md("### Rating prediction (RMSE)")
code('''
rows = []
for name in ORDER:
    p = RESULTS_DIR / f"{name}_final.json"
    if p.exists():
        r = json.loads(p.read_text())
        if r["test"]["rmse"] is not None:
            rows.append({"model": name, "val_rmse": r["val"]["rmse"], "test_rmse": r["test"]["rmse"], "test_mae": r["test"]["mae"]})
pd.DataFrame(rows).round(4)
''')
finding("results_rating")

md("# Part 5. Where the models fail: popularity slices, activity slices, cold items and cold-start users")
code('''
from recsys.plots import slice_heatmap, cold_start_curve
for kind in ("item_bin", "user_bin"):
    ax = slice_heatmap(RESULTS_DIR, kind, "recall@10"); ax.figure.savefig(FIG / f"slices_{kind}.png", dpi=150); plt.show()
''', "slices")
finding("slices", tag="slices")
code('''
rows = []
for name in ORDER:
    p = RESULTS_DIR / f"{name}_final.json"
    if p.exists():
        c = json.loads(p.read_text())["test_slices"]["item_bin"]["cold"]
        rows.append({"model": name, "recall@10 on never-seen items": c["recall@10"], "n": c["n"]})
pd.DataFrame(rows).round(4)
''', "cold_items")
finding("cold_items", tag="cold_items")
code('''
ax = cold_start_curve(RESULTS_DIR / "processed_coldstart"); ax.figure.savefig(FIG / "cold_start_curve.png", dpi=150); plt.show()
''', "cold_users")
finding("cold_users", tag="cold_users")

# ----------------------------------------------------------------------------- demo
md("# Part 6. The recommender in use")
finding("demo_intro", "Give the recommender a user and it returns ten products. The model sees the user's known purchases (everything except their held-out last one).")
code(FINDINGS.get("demo_code", "print('demo cell not generated yet')"))

md("# Part 7. Limitations and notes")
finding("limitations")
md("""
### Reference
- Krichene, W. and Rendle, S. (2020). On Sampled Metrics for Item Recommendation. Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20).
""")

assert len(cells) == len(tags), (len(cells), len(tags))  # every cell must have a tag entry


def write(path, selected_cells, title=None):
    nb = nbf.v4.new_notebook()
    nb.cells = ([nbf.v4.new_markdown_cell(title)] if title else []) + selected_cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(nb, str(path))
    print("wrote", path, len(nb.cells), "cells")


write(OUT, cells)

# ----------------------------------------------------------------------------- shorter notebooks that import the package
setup_cell = nbf.v4.new_code_cell(
    "import json, sys, time\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n"
    + shared_setup("Path.cwd().parent").strip("\n"))
helpers = [c for c, t in zip(cells, tags) if t == "helpers"]


def pick(*wanted):
    return [c for c, t in zip(cells, tags) if t in wanted]


closing = lambda names: nbf.v4.new_code_cell(f'''
rows = []
for name in ["popularity"] + {names!r}:
    r = json.load(open(RESULTS_DIR / f"{{name}}_final.json"))
    t, c = r["test"], r["cost"]
    rows.append({{"model": name, "recall@10": t["recall@10"], "ndcg@10": t["ndcg@10"], "mrr": t["mrr"], "rmse": t["rmse"],
                 "coverage@10": t["coverage@10"], "fit_seconds": c["fit_seconds"], "score_ms_per_user": c["score_ms_per_user"]}})
pd.DataFrame(rows).round(4)
''')
NB = ROOT / "notebooks"
write(NB / "02_classical_models.ipynb",
      [setup_cell] + helpers + pick(*[f"model:{m}" for m in ("knn", "als", "svd", "bpr", "content")])
      + [nbf.v4.new_markdown_cell("## Final runs against the popularity baseline"), closing(["knn", "als", "svd", "bpr", "content"])],
      "# Classical models: item-KNN, ALS, SVD, BPR and the content-based hybrid\nThe final self-contained notebook (`IE7275_Project1_Final.ipynb`) contains the code of every model; this notebook imports it from the `recsys` package.")
write(NB / "03_neural_and_sequential.ipynb",
      [setup_cell] + helpers + pick("model:ncf", "model:sasrec")
      + [nbf.v4.new_markdown_cell("## Final runs against the popularity baseline"), closing(["ncf", "sasrec"])],
      "# Neural and sequential models: NeuMF and SASRec")
write(NB / "04_slice_and_cold_start_analysis.ipynb",
      [setup_cell] + pick("slices", "cold_items", "cold_users"),
      "# Where the models fail: popularity slices, activity slices, cold items and cold-start users")
