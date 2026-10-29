"""Confirm this machine is set up and holds the same data as everyone else.

    python -m recsys.check_env
"""
import importlib
import sys

EXPECTED_VERSION = "4fe78b90381a"  # data/processed fingerprint, must match across the team
EXPECTED_COLD_VERSION = "6943ffc82a19"  # data/processed_coldstart
REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "pyarrow", "torch", "matplotlib", "tqdm", "pytest"]


def main() -> None:
    ok = True
    print(f"python {sys.version.split()[0]}")
    for name in REQUIRED:
        try:
            print(f"  {name:12s} {getattr(importlib.import_module(name), '__version__', 'ok')}")
        except ImportError:
            print(f"  {name:12s} MISSING  -> pip install -r requirements.txt")
            ok = False
    if not ok:
        raise SystemExit(1)

    from recsys.config import DATA_DIR, ROOT, get_device
    from recsys.data import load

    print(f"device: {get_device()}")
    data = load()  # raises if the files were altered
    print(f"data: {DATA_DIR.name} version {data.data_version}, {data.n_users} users, {data.n_items} items")
    if DATA_DIR.name == "processed" and data.data_version != EXPECTED_VERSION:
        raise SystemExit(f"WRONG DATA: expected version {EXPECTED_VERSION}. Re-copy data/processed from Devansh.")
    cold_dir = ROOT / "data" / "processed_coldstart"
    if not cold_dir.exists():
        raise SystemExit("data/processed_coldstart is missing. Re-copy the data folder from Devansh.")
    cold = load(cold_dir)
    print(f"cold-start data: version {cold.data_version}, {len(cold.new_users)} new users")
    if cold.data_version != EXPECTED_COLD_VERSION:
        raise SystemExit(f"WRONG COLD-START DATA: expected version {EXPECTED_COLD_VERSION}. Re-copy it from Devansh.")
    print("OK")


if __name__ == "__main__":
    main()
