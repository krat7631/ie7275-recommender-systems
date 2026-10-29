import json
import shutil
import zipfile

import matplotlib
import numpy as np
import pytest

from recsys import harness
from recsys.check_model import load_class, run_checks, static_checks
from recsys.data import load
from recsys.handoff import OWNERS, is_allowed, pack, unpack
from recsys.plots import slice_heatmap
from recsys.report import sweep_table

matplotlib.use("Agg")

GOOD = '''"""Toy item-item model for the tests."""
import numpy as np

from recsys.data import to_csr
from recsys.models.base import Recommender


class Toy(Recommender):
    name = "toy"

    def fit(self, data):
        R = to_csr(data.train, data.n_users, data.n_items)
        self.S = (R.T @ R).toarray().astype(np.float32)
        np.fill_diagonal(self.S, 0)
        self.pop = data.item_train_count.astype(np.float32)
        self.n_items = data.n_items

    def score(self, user_idxs, histories):
        out = np.empty((len(user_idxs), self.n_items), dtype=np.float32)
        for i, h in enumerate(histories):
            out[i] = self.S[h].sum(axis=0) + 1e-3 * self.pop if len(h) else self.pop
        return out
'''


def variant(tmp_path, old, new, fname="toy.py"):
    assert old in GOOD, old
    p = tmp_path / fname
    p.write_text(GOOD.replace(old, new))
    return p


def levels(results):
    return {(lvl, chk) for lvl, chk, _ in results}


# ---------------------------------------------------------------- check_model

def test_a_conforming_model_has_no_failures(cold, tmp_path):
    p = tmp_path / "toy.py"
    p.write_text(GOOD)
    res = run_checks(p, cold["dir"])
    assert not [r for r in res if r[0] == "FAIL"], res
    assert ("PASS", "new users, n=0") in levels(res)


@pytest.mark.parametrize("old,new,fname,expected", [
    ('self.n_items = data.n_items', 'self.n_items = data.n_items; _ = data.test', "toy.py", ("FAIL", "test split")),
    ('import numpy as np\n', 'import numpy as np\nimport implicit\n', "toy.py", ("FAIL", "imports")),
    ('self.n_items = data.n_items', 'self.n_items = data.n_items; open("data/raw/Video_Games.jsonl")', "toy.py", ("FAIL", "data access")),
    ('name = "toy"', 'name = "other"', "toy.py", ("FAIL", "name")),
    ('if len(h) else self.pop', 'if len(h) else np.zeros(self.n_items)', "toy.py", ("FAIL", "new users, n=0")),
    ('out[i] = self.S[h].sum(axis=0) + 1e-3 * self.pop if len(h) else self.pop', 'out[i] = self.S[h[0]] + 1e-3 * self.pop', "toy.py", ("FAIL", "new users, n=0")),
    ('+ 1e-3 * self.pop', '+ np.random.rand(self.n_items).astype(np.float32)', "toy.py", ("FAIL", "deterministic")),
    ('(len(user_idxs), self.n_items)', '(len(user_idxs), self.n_items - 1)', "toy.py", ("FAIL", "score (known users)")),
])
def test_each_rule_violation_is_caught(cold, tmp_path, old, new, fname, expected):
    res = run_checks(variant(tmp_path, old, new, fname), cold["dir"])
    assert expected in levels(res) or any(lvl == "FAIL" and expected[1] in chk for lvl, chk, _ in res), res


def test_static_scan_flags_hardcoded_device_as_a_warning(tmp_path):
    p = variant(tmp_path, 'import numpy as np\n', 'import numpy as np\nDEVICE = "cuda"\n')
    assert ("WARN", "device") in levels(static_checks(p, "toy"))


def test_sentence_transformers_is_only_allowed_in_content(tmp_path):
    p = variant(tmp_path, 'import numpy as np\n', 'import numpy as np\nimport sentence_transformers\n')
    assert ("FAIL", "imports") in levels(static_checks(p, "knn"))
    assert ("PASS", "imports") in levels(static_checks(p, "content"))


# ---------------------------------------------------------------- hand-off

@pytest.fixture
def mini(processed, cold, tmp_path, monkeypatch):
    """A tiny project folder with a conforming 'knn' model and its final results on both datasets."""
    root = tmp_path / "proj"
    for src, dst in ((processed["dir"], "processed"), (cold["dir"], "processed_coldstart"), (cold["dir"], "processed_dev_coldstart")):
        shutil.copytree(src, root / "data" / dst)
    (root / "recsys" / "models").mkdir(parents=True)
    path = root / "recsys" / "models" / "knn.py"
    path.write_text(GOOD.replace('name = "toy"', 'name = "knn"').replace("Toy", "Knn"))
    cls = load_class(path, "knn")
    for results, data in ((root / "results", load(root / "data" / "processed")),
                          (root / "results" / "processed_coldstart", load(root / "data" / "processed_coldstart"))):
        monkeypatch.setattr(harness, "RESULTS_DIR", results)
        harness.run_experiment(cls(), data, tag="final")
    (root / "notebooks").mkdir()
    (root / "notebooks" / "02_classical_models.ipynb").write_text("{}")
    return root


def test_pack_then_unpack_moves_exactly_the_owners_files(mini, processed, cold, tmp_path):
    z = pack("rishi", mini)
    with zipfile.ZipFile(z) as zf:
        names = set(zf.namelist())
    assert "recsys/models/knn.py" in names and "results/knn_final.json" in names
    assert "results/processed_coldstart/knn_final.json" in names and "notebooks/02_classical_models.ipynb" in names
    assert not any(n.startswith(("data/", "tests/")) or n.endswith(("harness.py", "base.py")) for n in names)

    other = tmp_path / "devansh"
    for src, dst in ((processed["dir"], "processed"), (cold["dir"], "processed_coldstart")):
        shutil.copytree(src, other / "data" / dst)
    unpack(z, other)
    assert (other / "recsys" / "models" / "knn.py").read_text() == (mini / "recsys" / "models" / "knn.py").read_text()
    assert (other / "results" / "per_user" / "knn_final.npz").exists()


def test_pack_refuses_a_model_without_final_results(mini):
    (mini / "results" / "processed_coldstart" / "knn_final.json").unlink()
    with pytest.raises(SystemExit, match="cold-start"):
        pack("rishi", mini)


def test_pack_refuses_results_from_different_data(mini):
    f = mini / "results" / "knn_final.json"
    r = json.loads(f.read_text())
    r["data_version"] = "000000000000"
    f.write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="different data"):
        pack("rishi", mini)


def test_pack_refuses_a_model_that_fails_the_conformance_check(mini):
    p = mini / "recsys" / "models" / "knn.py"
    p.write_text(p.read_text().replace("self.n_items = data.n_items", "self.n_items = data.n_items; _ = data.test"))
    with pytest.raises(SystemExit, match="check_model FAIL"):
        pack("rishi", mini)


def _forge(tmp_path, mini, members, manifest_extra=None, corrupt=None):
    manifest = {"owner": "rishi", "models": ["knn"], "data_versions": {
        d: json.loads((mini / "data" / d / "meta.json").read_text())["data_version"] for d in ("processed", "processed_coldstart")},
        "files": {m: "0" * 64 for m in members}}
    manifest.update(manifest_extra or {})
    z = tmp_path / "forged.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("MANIFEST.json", json.dumps(manifest))
        for m in members:
            zf.writestr(m, "x")
    return z


def test_unpack_refuses_shared_code(mini, tmp_path):
    with pytest.raises(SystemExit, match="may not hand off"):
        unpack(_forge(tmp_path, mini, ["recsys/harness.py"]), mini)


def test_unpack_refuses_another_owners_files_and_path_tricks(mini, tmp_path):
    with pytest.raises(SystemExit, match="may not hand off"):
        unpack(_forge(tmp_path, mini, ["recsys/models/ncf.py"]), mini)
    with pytest.raises(SystemExit, match="may not hand off"):
        unpack(_forge(tmp_path, mini, ["recsys/models/../harness.py"]), mini)


def test_unpack_refuses_corrupted_files_and_wrong_data_versions(mini, tmp_path):
    with pytest.raises(SystemExit, match="hash mismatch"):
        unpack(_forge(tmp_path, mini, ["recsys/models/knn.py"]), mini)
    with pytest.raises(SystemExit, match="data versions differ"):
        unpack(_forge(tmp_path, mini, ["recsys/models/knn.py"], {"data_versions": {"processed": "a", "processed_coldstart": "b"}}), mini)


def test_ownership_rules():
    assert is_allowed("rishi", "recsys/models/knn.py") and is_allowed("rishi", "recsys/models/rishi_utils.py")
    assert is_allowed("rishi", "results/als_foldin.json") and is_allowed("rishi", "results/processed_coldstart/per_user/als_final.npz")
    assert is_allowed("venkat", "notebooks/04_slice_and_cold_start_analysis.ipynb") and is_allowed("venkat", "figures/venkat_slices.png")
    assert not is_allowed("rishi", "recsys/models/ncf.py") and not is_allowed("venkat", "recsys/models/knn.py")
    assert not is_allowed("rishi", "results/processed_dev/knn_final.json")  # dev results never travel
    assert not is_allowed("rishi", "recsys/models/popularity.py") and not is_allowed("rishi", "recsys/models/base.py")
    assert not is_allowed("rishi", "PROJECT_RULES.md") and not is_allowed("rishi", "data/processed/train.parquet")
    assert set(OWNERS["rishi"]["models"]).isdisjoint(OWNERS["venkat"]["models"])


# ---------------------------------------------------------------- notebook helpers

def test_sweep_table_and_slice_heatmap_read_the_results(mini):
    t = sweep_table("knn", mini / "results")
    assert list(t["run"]) == ["knn_final"] and bool(t["selected"][0])
    ax = slice_heatmap(mini / "results", "item_bin")
    assert len(ax.get_yticklabels()) == 1
    slice_heatmap(mini / "results", "user_bin")
    assert sweep_table("knn", mini / "results", by="val_rmse").shape[0] == 1
    with pytest.raises(ValueError):
        sweep_table("nope", mini / "results")


def test_report_cli_prints_the_sweep(mini, monkeypatch, capsys):
    import sys

    from recsys import report

    monkeypatch.setattr(report, "RESULTS_DIR", mini / "results")
    monkeypatch.setattr(sys, "argv", ["report", "knn"])
    report.main()
    out = capsys.readouterr().out
    assert "knn_final" in out and "val_ndcg@10" in out and "selected" in out
