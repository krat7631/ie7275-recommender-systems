"""Hyperparameter grids used for model selection. Every run is chosen on validation only (python -m recsys.report <model>).

    python3 docs/sweeps.py svd                              # the grid listed below
    python3 docs/sweeps.py als '{"reg": [10, 100]}' factors=128   # a custom grid, with fixed settings

Tags are built from the settings, e.g. svd_factors16_reg0.1.
"""
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GRIDS = {
    "svd": {"factors": [8, 16, 32, 64], "reg": [0.05, 0.1, 0.2, 0.5]},
    "bpr": {"factors": [32, 64, 128], "lr": [0.002, 0.005, 0.01]},
}


def run(model: str, grid: dict, fixed: dict | None = None) -> None:
    fixed = fixed or {}
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        settings = {**fixed, **dict(zip(keys, combo))}
        tag = "_".join(f"{k}{v}" for k, v in settings.items())
        cmd = [sys.executable, "-m", "recsys.run", model, "--set", *[f"{k}={v}" for k, v in settings.items()], "--tag", tag]
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        print((out.stdout.strip().splitlines() or [out.stderr.strip()[-300:]])[-1] if out.returncode == 0 else f"FAILED {tag}: {out.stderr.strip()[-300:]}", flush=True)


if __name__ == "__main__":
    model = sys.argv[1]
    grid, fixed = GRIDS.get(model, {}), {}
    for arg in sys.argv[2:]:
        if arg.startswith("{"):
            grid = json.loads(arg)
        else:
            k, v = arg.split("=", 1)
            fixed[k] = json.loads(v)
    run(model, grid, fixed)
