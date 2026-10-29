"""Run the chosen setting of every model as its final run: on the full data and on the cold-start data.

    python3 docs/run_finals.py            # every model in docs/final_settings.json
    python3 docs/run_finals.py knn als    # only these

Runs are sequential and should be done with nothing else running, because the cost table uses their training and scoring
times. docs/final_settings.json maps each model to the hyperparameters picked on the validation split. A model may list
"variants": scoring options evaluated from the saved fitted model (docs/variant.py) instead of being retrained, e.g. the
ALS fold-in on the cold-start data and ALS scored from the given history on the full data. "save": true keeps the fitted
model in results/models/ for the demo.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = json.loads((ROOT / "docs" / "final_settings.json").read_text())


def run(model: str, settings: dict, tag: str, cold: bool, save: str | None = None) -> None:
    cmd = [sys.executable, "-m", "recsys.run", model, "--tag", tag]
    if settings:
        cmd += ["--set", *[f"{k}={json.dumps(v)}" for k, v in settings.items()]]
    env = {**os.environ, **({"RECSYS_DATA_DIR": "data/processed_coldstart"} if cold else {}), **({"RECSYS_SAVE_MODEL": str(ROOT / save)} if save else {})}
    t0 = time.perf_counter()
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    last = (out.stdout.strip().splitlines() or [out.stderr[-300:]])
    print(f"[{'cold' if cold else 'full'}] {model}_{tag}: {time.perf_counter() - t0:.0f}s | " + " || ".join(last[-2:] if cold else last[-1:]), flush=True)
    if out.returncode:
        raise SystemExit(f"FAILED {model}: {out.stderr[-500:]}")


def variant(pickle_path: str, tag: str, overrides: dict, cold: bool) -> None:
    cmd = [sys.executable, "docs/variant.py", pickle_path, tag, *[f"{k}={json.dumps(v)}" for k, v in overrides.items()]]
    env = {**os.environ, **({"RECSYS_DATA_DIR": "data/processed_coldstart"} if cold else {})}
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    print(f"[{'cold' if cold else 'full'}] variant {tag}: " + " || ".join(out.stdout.strip().splitlines()[-2:]), flush=True)
    if out.returncode:
        raise SystemExit(f"FAILED variant {tag}: {out.stderr[-500:]}")


if __name__ == "__main__":
    models = sys.argv[1:] or [m for m in SETTINGS if not m.startswith("_")]
    for m in models:
        spec = SETTINGS[m]
        variants = spec.get("variants", [])
        run(m, spec["set"], "final", cold=False, save=f"results/models/{m}_main.pkl" if spec.get("save") or variants else None)
        run(m, spec["set"], "final", cold=True, save=f"results/models/{m}_cold.pkl" if variants else None)
        for v in variants:  # scoring variants of the model just fitted: evaluated from the saved model, not refitted
            variant(f"results/models/{m}_{v['on']}.pkl", v["tag"], v["set"], cold=v["on"] == "cold")
