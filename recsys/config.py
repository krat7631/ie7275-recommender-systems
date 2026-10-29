"""Constants shared by every model and by the evaluation harness. Do not override locally."""
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# RECSYS_DATA_DIR lets you point at a different processed dataset (e.g. data/processed_dev).
DATA_DIR = Path(os.environ.get("RECSYS_DATA_DIR", ROOT / "data" / "processed"))

# Results from a non-default dataset go to their own folder so they never mix with real results.
RESULTS_DIR = ROOT / "results" if DATA_DIR.name == "processed" else ROOT / "results" / DATA_DIR.name

SEED = 42
K_CORE = 5
K_LIST = (5, 10, 20)
PRIMARY_METRIC = "ndcg@10"  # use this one for model selection and early stopping

# Cold-start experiment: a fixed share of users is held out of training entirely ("new users"),
# then only their first n purchases are revealed to the model.
COLD_FRAC = 0.1
COLD_REVEAL = (0, 1, 2, 3)


def get_device() -> str:
    """'mps' on Apple Silicon, 'cuda' if present, else 'cpu'."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
