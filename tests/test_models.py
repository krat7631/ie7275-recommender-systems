"""Correctness tests for the algorithms themselves (the conformance checker only tests the interface)."""
import numpy as np
import scipy.sparse as sp

from recsys.models.als import Als, _solve_side


def _dense_als_update(F, R, alpha, reg):
    """Textbook update with explicit confidence matrices: x_u = (F^T C_u F + reg I)^-1 F^T C_u p_u."""
    n, f = R.shape[0], F.shape[1]
    out = np.zeros((n, f))
    for u in range(n):
        p = (R[u] > 0).astype(float)
        c = 1.0 + alpha * p
        out[u] = np.linalg.solve(F.T @ (c[:, None] * F) + reg * np.eye(f), F.T @ (c * p))
    return out


def test_batched_als_solver_matches_the_dense_textbook_formula():
    rng = np.random.default_rng(0)
    R = sp.random(40, 25, density=0.2, random_state=1, format="csr", data_rvs=lambda n: np.ones(n))
    R.sum_duplicates()
    R.data[:] = 1.0
    F = rng.standard_normal((25, 6)).astype(np.float32)
    got = _solve_side(F, R.indptr, R.indices, alpha=40.0, reg=0.1, budget=200)  # tiny budget forces many chunks
    want = _dense_als_update(F.astype(np.float64), R.toarray(), alpha=40.0, reg=0.1)
    assert np.allclose(got, want, atol=1e-4)


def test_als_rows_without_interactions_get_a_zero_vector():
    F = np.random.default_rng(0).standard_normal((5, 3)).astype(np.float32)
    indptr, indices = np.array([0, 2, 2, 3]), np.array([0, 1, 4])  # user 1 has no items
    out = _solve_side(F, indptr, indices, alpha=10.0, reg=0.1)
    assert (out[1] == 0).all() and (out[0] != 0).any()


def test_als_fit_improves_validation_and_foldin_uses_the_history(cold):
    d = cold["data"]
    m = Als(factors=8, iterations=6)
    m.fit(d)
    curve = [c["val_ndcg@10"] for c in m.curve_]
    assert curve[-1] > curve[0]  # converging, not diverging
    new = d.new_users[:5]
    hist = [d.new_user_history[u][:3] for u in new]
    plain = m.score(new, hist)
    assert (plain == m.pop).all()  # no fold-in: popularity for unseen users
    m.foldin = True
    folded = m.score(new, hist)
    assert not (folded == m.pop).all()  # fold-in: scores come from the revealed purchases
    assert (m.score(new, [np.array([], dtype=np.int64)] * 5) == m.pop).all()  # empty history: popularity


# ---------------------------------------------------------------- shared sampler and the neural models
import torch  # noqa: E402

from recsys.models.bpr import Bpr  # noqa: E402
from recsys.models.ncf import Ncf, _NeuMF  # noqa: E402
from recsys.models.sasrec import Sasrec, _SASRec  # noqa: E402
from recsys.models.svd import Svd  # noqa: E402
from recsys.sampling import interaction_keys, sample_negatives  # noqa: E402


def test_negative_sampler_never_returns_a_known_item():
    rng = np.random.default_rng(0)
    n_users, n_items = 30, 40
    users = rng.integers(0, n_users, 400)
    items = rng.integers(0, n_items, 400)
    keys = interaction_keys(users, items, n_items)
    known = set(zip(users.tolist(), items.tolist()))
    q_users = np.repeat(np.arange(n_users), 50)
    neg = sample_negatives(rng, q_users, n_items, keys)
    assert len(neg) == len(q_users) and not any((u, j) in known for u, j in zip(q_users.tolist(), neg.tolist()))
    assert neg.min() >= 0 and neg.max() < n_items


def test_chunked_neumf_scoring_equals_the_plain_forward_pass():
    torch.manual_seed(0)
    net = _NeuMF(n_users=7, n_items=23, factors=5, mlp_factors=6, layers=[8, 4]).eval()
    users = torch.tensor([0, 3, 6])
    fast = net.score_all(users, cap=300)  # tiny cap: forces many item chunks
    pairs_u = users.repeat_interleave(23)
    pairs_i = torch.arange(23).repeat(3)
    slow = net(pairs_u, pairs_i).reshape(3, 23)
    assert torch.allclose(fast, slow, atol=1e-5)


def test_sasrec_cannot_see_the_future_and_has_no_nan_on_padding():
    torch.manual_seed(0)
    net = _SASRec(n_items=30, hidden=16, blocks=2, heads=2, dropout=0.0, max_len=8).eval()
    a = torch.tensor([[0, 0, 0, 5, 9, 2, 7, 4]])
    b = a.clone()
    b[0, 6:] = torch.tensor([11, 13])  # change only the last two items
    ha, hb = net.encode(a), net.encode(b)
    assert torch.isfinite(ha).all()
    assert torch.allclose(ha[:, :6], hb[:, :6], atol=1e-5)  # earlier positions unaffected
    assert not torch.allclose(ha[:, 7], hb[:, 7], atol=1e-5)  # the last position does react


def test_svd_treats_unseen_users_as_specified(cold):
    d = cold["data"]
    m = Svd(factors=4, epochs=3)
    m.fit(d)
    new = d.new_users[:4]
    items = np.arange(4)
    assert np.allclose(m.predict(new, items), m.mu + m.bi[items], atol=1e-5)  # mu + b_item for unseen users
    assert (m.score(new, [np.array([], dtype=np.int64)] * 4) == m.pop).all()
    assert np.isfinite(m.curve_[-1]["val_rmse"])


def test_bpr_ncf_sasrec_fit_and_handle_new_users(cold):
    d = cold["data"]
    new = d.new_users[:6]
    hist = [d.new_user_history[u][:3] for u in new]
    for m in (Bpr(factors=4, epochs=3), Ncf(factors=4, mlp_factors=4, layers=[8, 4], epochs=2), Sasrec(hidden=16, blocks=1, epochs=2)):
        m.fit(d)
        s = m.score(new, hist)
        assert s.shape == (6, d.n_items) and np.isfinite(s).all()
        assert len(m.curve_) >= 1
        if m.name == "sasrec":
            assert not (s == m.pop).all()  # sasrec uses the revealed purchases
        else:
            assert (s == m.pop).all()  # no per-user vector: popularity
        assert (m.score(new, [np.array([], dtype=np.int64)] * 6) == m.pop).all()


def test_models_can_be_saved_and_reloaded_with_identical_scores(cold, tmp_path):
    from recsys.models.base import Recommender

    d = cold["data"]
    users = d.eval_users[:5]
    hist = [d.train_history[u] for u in users]
    for m in (Als(factors=4, iterations=2), Ncf(factors=4, mlp_factors=4, layers=[8, 4], epochs=1), Sasrec(hidden=16, blocks=1, epochs=1)):
        m.fit(d)
        before = m.score(users, hist)
        m.save(tmp_path / f"{m.name}.pkl")
        again = Recommender.load(tmp_path / f"{m.name}.pkl")
        assert np.allclose(before, again.score(users, hist), atol=1e-5)
        assert np.allclose(before, m.score(users, hist), atol=1e-5)  # the original still works after saving


def test_als_foldin_equals_the_dense_update_and_foldin_all_covers_seen_users(cold):
    d = cold["data"]
    m = Als(factors=6, iterations=3)
    m.fit(d)
    hist = d.new_user_history[d.new_users[0]][:3]
    R = np.zeros((1, d.n_items))
    R[0, hist] = 1.0
    want = _dense_als_update(m.Y.astype(np.float64), R, alpha=m.alpha, reg=m.reg)[0]
    got = m._vectors_from_history([hist])[0]
    assert np.allclose(got, want, atol=1e-4)

    users = d.eval_users[:4]
    hs = [d.trainval_history[u] for u in users]
    plain = m.score(users, hs)
    m.foldin_all = True
    from_history = m.score(users, hs)
    assert not np.allclose(plain, from_history)  # seen users are now scored from the history they are given
    assert np.isfinite(from_history).all()
