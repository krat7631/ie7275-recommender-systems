"""Save a slim copy of the final ALS for the demo: item factors only, users scored from their history (foldin_all).

    python3 docs/save_demo_model.py

results/models/als_main.pkl holds the learned user vectors too (about 120 MB); the demo never reads them, so the slim copy
is a fifth of the size.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recsys.models.base import Recommender  # noqa: E402

model = Recommender.load("results/models/als_main.pkl")
model.X, model.foldin_all = None, True
model.config["foldin_all"] = True
model.save("results/models/als_demo.pkl")
print("saved results/models/als_demo.pkl")
