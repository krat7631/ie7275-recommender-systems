"""Pack a teammate's work for Devansh, and unpack it safely.

    python -m recsys.handoff pack rishi          # teammate: checks everything, writes dist/handoff_rishi_<time>.zip
    python -m recsys.handoff unpack <zip file>   # Devansh: verifies the zip and copies the files into the project

Only files a person owns can travel: their model files, their notebooks, their figures and the results of their models.
Shared code can never be overwritten by a hand-off, and a zip made on the wrong data is rejected.
"""
import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

from recsys.check_model import run_checks
from recsys.config import ROOT

OWNERS = {
    "rishi": {"models": ("knn", "als", "bpr", "svd", "content"), "notebooks": ("02_classical_models.ipynb",)},
    "venkat": {"models": ("ncf", "sasrec"), "notebooks": ("03_neural_and_sequential.ipynb", "04_slice_and_cold_start_analysis.ipynb")},
}
TAG = r"[A-Za-z0-9._-]"


def is_allowed(owner: str, rel: str) -> bool:
    cfg = OWNERS[owner]
    models = "|".join(re.escape(m) for m in cfg["models"])
    notebooks = "|".join(re.escape(n) for n in cfg["notebooks"])
    patterns = [
        rf"recsys/models/(?:{models})\.py",
        rf"recsys/models/{owner}_[A-Za-z0-9_]+\.py",
        rf"notebooks/(?:{notebooks})",
        rf"figures/{owner}_{TAG}+\.png",
        rf"results/(?:processed_coldstart/)?(?:per_user/)?(?:{models})(?:_{TAG}+)?\.(?:json|npz)",
    ]
    return any(re.fullmatch(p, rel) for p in patterns)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _versions(root: Path) -> dict:
    return {d: json.loads((root / "data" / d / "meta.json").read_text())["data_version"] for d in ("processed", "processed_coldstart")}


def pack(owner: str, root: Path = ROOT) -> Path:
    if owner not in OWNERS:
        raise SystemExit(f"owner must be one of {sorted(OWNERS)}")
    versions = _versions(root)
    delivered = [m for m in OWNERS[owner]["models"] if (root / "recsys" / "models" / f"{m}.py").exists()]
    if not delivered:
        raise SystemExit("no model files found in recsys/models/")

    problems = []
    for m in delivered:
        fails = [r for r in run_checks(root / "recsys" / "models" / f"{m}.py", root / "data" / "processed_dev_coldstart") if r[0] == "FAIL"]
        problems += [f"{m}: check_model FAIL {c}: {d}" for _, c, d in fails]
        for folder, key in ((root / "results", "processed"), (root / "results" / "processed_coldstart", "processed_coldstart")):
            final = folder / f"{m}_final.json"
            if not final.exists():
                where = "the cold-start dataset" if key == "processed_coldstart" else "the full data"
                problems.append(f"{m}: missing {final.relative_to(root)}; run the chosen settings with --tag final on {where}")
            for f in folder.glob(f"{m}*.json"):
                if is_allowed(owner, f.relative_to(root).as_posix()) and json.loads(f.read_text())["data_version"] != versions[key]:
                    problems.append(f"{f.relative_to(root)} was made on different data than the kit; delete it and re-run")
    if problems:
        raise SystemExit("cannot pack yet:\n  " + "\n  ".join(problems))

    files = sorted(p for p in root.rglob("*") if p.is_file() and "data/raw" not in p.as_posix()
                   and is_allowed(owner, p.relative_to(root).as_posix()))
    manifest = {"owner": owner, "created": datetime.now().isoformat(timespec="seconds"), "models": delivered,
                "data_versions": versions, "files": {p.relative_to(root).as_posix(): _sha(p) for p in files}}
    out = root / "dist"
    out.mkdir(exist_ok=True)
    zpath = out / f"handoff_{owner}_{datetime.now():%Y%m%d_%H%M}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
        for p in files:
            z.write(p, p.relative_to(root).as_posix())
    print(f"packed {len(files)} files for models {delivered}\n-> {zpath}\nSend this one file to Devansh.")
    return zpath


def unpack(zip_path: Path, root: Path = ROOT) -> list[str]:
    with zipfile.ZipFile(zip_path) as z:
        manifest = json.loads(z.read("MANIFEST.json"))
        owner = manifest["owner"]
        if owner not in OWNERS:
            raise SystemExit(f"unknown owner {owner!r} in manifest")
        if manifest["data_versions"] != _versions(root):
            raise SystemExit(f"data versions differ: zip {manifest['data_versions']} vs project {_versions(root)}")
        members = [n for n in z.namelist() if n != "MANIFEST.json"]
        bad = [n for n in members if ".." in Path(n).parts or Path(n).is_absolute() or not is_allowed(owner, n)]
        if bad:
            raise SystemExit(f"refused: files {owner} may not hand off: {bad}")
        if set(members) != set(manifest["files"]):
            raise SystemExit("refused: zip contents do not match the manifest")
        for n in members:
            if hashlib.sha256(z.read(n)).hexdigest() != manifest["files"][n]:
                raise SystemExit(f"refused: {n} is corrupted (hash mismatch)")
        for n in members:
            target = root / n
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(n))
    print(f"unpacked {len(members)} files from {owner} (models {manifest['models']}). Next: pytest, check_model on each model, compare --final.")
    return members


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in ("pack", "unpack"):
        raise SystemExit("usage: python -m recsys.handoff pack <rishi|venkat>   |   unpack <zip file>")
    pack(sys.argv[2]) if sys.argv[1] == "pack" else unpack(Path(sys.argv[2]))


if __name__ == "__main__":
    main()
