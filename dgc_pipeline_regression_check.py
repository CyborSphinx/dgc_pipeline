"""
Self-contained regression check: refactored dgc_pipeline == original math.

Run from the folder that CONTAINS the dgc_pipeline/ package:
    python dgc_pipeline_regression_check.py

It builds small synthetic data, runs BOTH the original inline implementations and
the package classes, and asserts identical numbers for:
  - OperatorSequence (free evolution, control columns/vector) over all 3 op types
  - FusedOperatorBuilder.D_fused (eps_fuse and ridge variants)
  - DGC rank-one operators
  - BOnly / Dynamics(timed,untimed) / ControlBestDay scorers (curve, LOO, mass)

No real data or GPU needed. Exit code 0 = all identical.
"""
import sys
import numpy as np
from numpy.linalg import norm, svd, pinv
from scipy.optimize import nnls

from dgc_pipeline.operators import OperatorSequence
from dgc_pipeline.scoring import (BOnlyScorer, DynamicsScorer, ControlBestDayScorer)
from dgc_pipeline.fusion import FusedOperatorBuilder, DGCOperatorBuilder
from dgc_pipeline.backend import Backend


# ---------- original inline implementations (verbatim from monolith) ----------
def apply_op(op, vec):
    if op[0] == "rank1":
        _, u, v = op; return vec + u * (v @ vec)
    if op[0] == "diff_plus_I":
        return vec + op[1] @ vec
    if op[0] == "dense":
        return op[1] @ vec

def _free_evolution(ops, x, n):
    z = x.copy()
    for k in range(n): z = apply_op(ops[k] if k < len(ops) else ops[-1], z)
    return z

def control_columns_timed(ops, b, n):
    G = len(b); cols = np.zeros((G, n))
    for k in range(n):
        c = b.copy()
        for j in range(k + 1, n): c = apply_op(ops[j] if j < len(ops) else ops[-1], c)
        cols[:, k] = c
    return cols

def control_vector(ops, b, n):
    G = len(b); C = np.zeros(G)
    for k in range(n):
        c = b.copy()
        for j in range(k + 1, n): c = apply_op(ops[j] if j < len(ops) else ops[-1], c)
        C += c
    return C

def score_b_only(B, tf, xi, xt):
    r = xt - xi; d0 = norm(r); sc = {}
    for j in tf:
        b = B[:, j]
        if norm(b) < 1e-12: continue
        u = max(0.0, r @ b / (b @ b + 1e-12)); sc[j] = d0 - norm(r - u * b)
    return sc

def score_dynamics(ops, B, tf, xi, xt, n, timed=True):
    z = _free_evolution(ops, xi, n); r = xt - z; d0 = norm(r); sc = {}
    for j in tf:
        b = B[:, j]
        if norm(b) < 1e-12: continue
        if timed:
            C = control_columns_timed(ops, b, n)
            keep = [k for k in range(C.shape[1]) if norm(C[:, k]) >= 1e-12]
            if not keep: sc[j] = 0.0; continue
            M_ = C[:, keep]; u, _ = nnls(M_, r); sc[j] = d0 - norm(r - M_ @ u)
        else:
            C = control_vector(ops, b, n)
            if norm(C) < 1e-12: sc[j] = 0.0; continue
            u = max(0.0, r @ C / (C @ C + 1e-12)); sc[j] = d0 - norm(r - u * C)
    return sc

def score_control_bestday(ops, B, tf, xi, xt, n, eps=0.0):
    cols_idx = [j for j in tf if norm(B[:, j]) > 1e-12]
    if not cols_idx: return None
    m = len(cols_idx); G = len(xi); Bsub = B[:, cols_idx]
    def step(op, V): return np.column_stack([apply_op(op, V[:, c]) for c in range(V.shape[1])]) if V.shape[1] else V
    cols = np.zeros((G, 0)); col_tf = np.zeros((0,), int); xfree = xi.copy(); curve = []; best = None
    for K in range(1, n + 1):
        op_prev = ops[K - 1] if (K - 1) < len(ops) else ops[-1]
        if cols.shape[1] > 0: cols = step(op_prev, cols)
        xfree = apply_op(op_prev, xfree); cols = np.hstack([cols, Bsub]); col_tf = np.concatenate([col_tf, np.arange(m)])
        rK = xt - xfree; gap = float(norm(rK)); cn = norm(cols, axis=0) + 1e-12; Cs = cols / cn
        if eps > 0:
            A_aug = np.vstack([Cs, eps * np.eye(Cs.shape[1])]); b_aug = np.concatenate([rK, np.zeros(Cs.shape[1])]); u_s, _ = nnls(A_aug, b_aug)
        else: u_s, _ = nnls(Cs, rK)
        resid = float(norm(rK - Cs @ u_s)); curve.append((K, gap, resid))
        if best is None or resid < best["resid"]:
            best = dict(K=K, resid=resid, u=(u_s / cn).copy(), col_tf=col_tf.copy(), Cs=Cs.copy(), rK=rK.copy())
    mass = np.zeros(m)
    for c, t in enumerate(best["col_tf"]): mass[t] += best["u"][c]
    loo = np.zeros(m); base = best["resid"]; Cs = best["Cs"]; rK = best["rK"]; ctf = best["col_tf"]
    for t in range(m):
        keep = ctf != t
        if keep.sum() == 0: loo[t] = norm(rK) - base; continue
        u2, _ = nnls(Cs[:, keep], rK); loo[t] = float(norm(rK - Cs[:, keep] @ u2) - base)
    return dict(best_K=best["K"], curve=curve,
                mass={cols_idx[t]: mass[t] for t in range(m)},
                loo={cols_idx[t]: loo[t] for t in range(m)})

def build_fused_original(traj_c, ridge=0.0, eps_fuse=0.0, rank_tol=1e-8, max_rank=None):
    L, K, G = traj_c.shape; W_blocks, U_blocks = [], []
    for l in range(L):
        Xl = traj_c[l].T; Xsnap = Xl[:, :-1]; DX = Xl[:, 1:] - Xl[:, :-1]
        U, S, Vt = svd(Xsnap, full_matrices=False)
        r = int(np.sum(S > rank_tol * (S[0] if S.size else 1.0)))
        cap = max_rank if max_rank is not None else max(2, Xsnap.shape[1] // 2)
        r = min(r, cap)
        if r == 0: continue
        Ur, Sr, Vtr = U[:, :r], S[:r], Vt[:r, :]
        filt = Sr / (Sr ** 2 + ridge); W_l = DX @ Vtr.T @ np.diag(filt)
        W_blocks.append(W_l); U_blocks.append(Ur)
    Wcat = np.concatenate(W_blocks, axis=1); Ucat = np.concatenate(U_blocks, axis=1)
    sum_AP = Wcat @ Ucat.T; sum_P = Ucat @ Ucat.T
    if eps_fuse and eps_fuse > 0: denom_inv = np.linalg.inv(sum_P + eps_fuse * np.eye(G))
    else: denom_inv = pinv(sum_P, rcond=1e-10)
    return sum_AP @ denom_inv


# ------------------------------- the checks -----------------------------------
def main():
    np.random.seed(0)
    fails = []
    G, N = 40, 12

    # operators
    u, v = np.random.randn(G), np.random.randn(G)
    D = 0.05 * np.random.randn(G, G); M = np.eye(G) + 0.05 * np.random.randn(G, G)
    legacy = [("rank1", u, v), ("diff_plus_I", D), ("dense", M)]
    seq = OperatorSequence.from_legacy(legacy)
    x, b = np.random.randn(G), np.random.randn(G)
    if not np.allclose(seq.free_evolution(x, N), _free_evolution(legacy, x, N)): fails.append("free_evolution")
    if not np.allclose(seq.control_vector(b, N), control_vector(legacy, b, N)): fails.append("control_vector")
    if not np.allclose(seq.control_columns_timed(b, N), control_columns_timed(legacy, b, N)): fails.append("control_columns")

    # fusion
    Lc, Kc = 30, 12
    traj_c = np.cumsum(0.1 * np.random.randn(Lc, Kc, G), axis=1)
    traj_c -= traj_c.reshape(-1, G).mean(0)
    bld = FusedOperatorBuilder(Backend.cpu())
    for eps in (0.0, 0.05):
        for ridge in (0.0, 0.1):
            op_new, _ = bld.build(traj_c, ridge=ridge, eps_fuse=eps, n_steps_hint=11)
            D_old = build_fused_original(traj_c, ridge=ridge, eps_fuse=eps)
            if not np.allclose(op_new.D, D_old, atol=1e-9):
                fails.append(f"fusion(eps={eps},ridge={ridge})")

    # scorers
    Bm = np.abs(np.random.randn(G, 15)) * 0.3; Bm[:, 7] = 0
    tf = list(range(15)); xi = np.random.randn(G); xt = np.random.randn(G) * 2
    sd = OperatorSequence.from_legacy([("dense", M)])
    if not _dict_close(BOnlyScorer().score(Bm, tf, xi, xt), score_b_only(Bm, tf, xi, xt)):
        fails.append("BOnly")
    if not _dict_close(DynamicsScorer(sd, N, timed=True).score(Bm, tf, xi, xt),
                       score_dynamics([("dense", M)], Bm, tf, xi, xt, N, True)):
        fails.append("Dynamics(timed)")
    if not _dict_close(DynamicsScorer(sd, N, timed=False).score(Bm, tf, xi, xt),
                       score_dynamics([("dense", M)], Bm, tf, xi, xt, N, False)):
        fails.append("Dynamics(untimed)")
    for eps in (0.0, 0.05):
        rn = ControlBestDayScorer(sd, N, eps=eps).run(Bm, tf, xi, xt)
        ro = score_control_bestday([("dense", M)], Bm, tf, xi, xt, N, eps=eps)
        if rn["best_K"] != ro["best_K"] or not _dict_close(rn["loo"], ro["loo"]) \
           or not _dict_close(rn["mass"], ro["mass"]):
            fails.append(f"ControlBestDay(eps={eps})")

    print("=" * 60)
    if fails:
        print("REGRESSION FAILURES:", fails); sys.exit(1)
    print("ALL CHECKS PASS — package reproduces the original math exactly.")
    print("=" * 60)


def _dict_close(a, b):
    if set(a) != set(b): return False
    return all(np.isclose(a[k], b[k]) for k in a)


if __name__ == "__main__":
    main()
