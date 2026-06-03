"""
Diagnostics for the difference-DMD / fusion / forcing-term model.

Three checks the user requested:

  1. fusion_vs_averaging(traj_c, panel, backend, eps_fuse=0.05)
       Compare FusedOperatorBuilder to the dumb baseline mean_l(A_l). Reports
       rho, R2_change on held-out lineages, gap. One-shot validation.

  2. forcing_svd(M, traj_c, panel)
       Stack per-step forcings f_k = x_{k+1} - M x_k and SVD. With 4 OSKM TFs,
       model says rank(F) <= 4. Realized rank tells us how much u_k varied
       (1 if Dox constant; up to 4 if per-cell stochastic).

  3. forcing_geometry(M, traj_c, panel, Bu_const)
       Per-step plots of dot product, cosine angle, and magnitude of f_k against
       u_hat := Bu_const/||Bu_const||. Three complementary views.
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import norm, svd


def fusion_vs_averaging(traj_c, panel, backend, eps_fuse=0.05, ridge=0.0,
                        rank_tol=1e-8, max_rank=None, heldout_frac=0.2,
                        rng_seed=0):
    """Compare FusedOperatorBuilder M to mean_l(A_l). Held-out R2_change."""
    from .fusion import FusedOperatorBuilder
    tc = traj_c[:, :, panel] if panel is not None else traj_c
    L, K, G = tc.shape
    rng = np.random.default_rng(rng_seed)
    idx = rng.permutation(L)
    n_held = max(1, int(heldout_frac * L))
    held, train = idx[:n_held], idx[n_held:]

    def lineage_A(Xl):
        Xsnap, DX = Xl[:, :-1], Xl[:, 1:] - Xl[:, :-1]
        U, S, Vt = svd(Xsnap, full_matrices=False)
        r = int(np.sum(S > rank_tol * (S[0] if S.size else 1.0)))
        cap = max_rank if max_rank is not None else max(2, Xsnap.shape[1] // 2)
        r = min(r, cap)
        if r == 0:
            return None
        Ur, Sr, Vtr = U[:, :r], S[:r], Vt[:r, :]
        filt = Sr / (Sr ** 2 + ridge)
        return (DX @ Vtr.T @ np.diag(filt)) @ Ur.T

    # streaming mean — never stack the L*G*G tensor (that's ~60 GB for L~4000, G~1500).
    # accumulate a running sum in a single (G,G) float64 array, divide at the end.
    A_sum = np.zeros((G, G), dtype=np.float64)
    n_train_used = 0
    for l in train:
        A_l = lineage_A(tc[l].T)
        if A_l is None:
            continue
        A_sum += A_l
        n_train_used += 1
    if n_train_used == 0:
        return {"error": "no train lineages produced an A_l"}
    M_avg = np.eye(G) + A_sum / n_train_used

    bld = FusedOperatorBuilder(backend)
    fused_op, _ = bld.build(tc[train], ridge=ridge, eps_fuse=eps_fuse,
                            rank_tol=rank_tol, max_rank=max_rank,
                            n_steps_hint=K - 1)
    M_fused = np.eye(G) + fused_op.D

    def r2_change(M):
        ss_res = 0.0; ss_tot = 0.0
        for l in held:
            Xl = tc[l].T
            change = Xl[:, 1:] - Xl[:, :-1]
            pred = (M - np.eye(G)) @ Xl[:, :-1]
            ss_res += norm(change - pred) ** 2
            ss_tot += norm(change) ** 2
        return 1.0 - ss_res / (ss_tot + 1e-12)

    rho_avg   = float(np.max(np.abs(np.linalg.eigvals(M_avg))))
    rho_fused = float(np.max(np.abs(np.linalg.eigvals(M_fused))))
    r2_avg   = r2_change(M_avg)
    r2_fused = r2_change(M_fused)
    return dict(
        n_train=len(train), n_held=n_held,
        rho_avg=rho_avg, rho_fused=rho_fused,
        r2_change_heldout_avg=r2_avg,
        r2_change_heldout_fused=r2_fused,
        gap_r2=r2_fused - r2_avg,
        verdict=("FUSED BETTER" if r2_fused > r2_avg + 0.01
                 else "TIE / averaging suffices" if abs(r2_fused - r2_avg) <= 0.01
                 else "AVERAGING BETTER"),
    )


def forcing_svd(M, traj_c, panel, n_top_singular=15, return_per_lineage=False):
    """
    Per-step forcings f_k = x_{k+1} - M x_k, stacked. With 4 TFs, rank cap = 4
    if the model holds. Realized rank reveals how much u_k varied.
    """
    tc = traj_c[:, :, panel] if panel is not None else traj_c
    L, K, G = tc.shape
    mean_c = tc.mean(axis=0)
    F_mean = np.stack([mean_c[k + 1] - M @ mean_c[k]
                       for k in range(K - 1)], axis=1)
    s_mean = svd(F_mean, compute_uv=False)
    out = dict(K=K, L=L, G=G,
               sv_mean=s_mean[:n_top_singular],
               sv_mean_normalised=(s_mean / (s_mean[0] + 1e-12))[:n_top_singular],
               effective_rank_mean=int(np.sum(s_mean > 0.05 * s_mean[0])))
    if return_per_lineage:
        F_all = np.concatenate(
            [np.stack([tc[l, k + 1] - M @ tc[l, k] for k in range(K - 1)], axis=1)
             for l in range(L)], axis=1)
        s_all = svd(F_all, compute_uv=False)
        out.update(sv_all=s_all[:n_top_singular],
                   sv_all_normalised=(s_all / (s_all[0] + 1e-12))[:n_top_singular],
                   effective_rank_all=int(np.sum(s_all > 0.05 * s_all[0])))
    return out


def forcing_geometry(M, traj_c, panel, Bu_const):
    """
    Per-step, per-lineage f_k against u_hat := Bu_const/||.||:
      dot, cos(angle), magnitude.  Returns (K-1, L) arrays + per-step summaries.
    """
    tc = traj_c[:, :, panel] if panel is not None else traj_c
    L, K, G = tc.shape
    u = np.asarray(Bu_const).ravel()
    if u.shape[0] != G and panel is not None:
        u = u[panel]
    u_hat = u / (norm(u) + 1e-12)
    dot = np.empty((K - 1, L)); ang = np.empty((K - 1, L)); mag = np.empty((K - 1, L))
    for k in range(K - 1):
        F = tc[:, k + 1, :] - tc[:, k, :] @ M.T
        m = norm(F, axis=1)
        d = F @ u_hat
        dot[k], mag[k], ang[k] = d, m, d / (m + 1e-12)
    return dict(dot=dot, cos=ang, mag=mag, K=K, L=L,
                summary=dict(
                    dot_median=np.median(dot, axis=1), dot_std=np.std(dot, axis=1),
                    cos_median=np.median(ang, axis=1), cos_std=np.std(ang, axis=1),
                    mag_median=np.median(mag, axis=1), mag_std=np.std(mag, axis=1),
                ))
