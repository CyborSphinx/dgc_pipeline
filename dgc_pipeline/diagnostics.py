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


def recover_input_and_validate(M, traj_c, panel, xbar,
                               n_pairs_holdout=200, lineage_sample=None,
                               seed=0, verbose=True, **_ignored_kwargs):
    """
    Health check for a fitted operator M. Prints (and returns):
      [A] R^2(M on pair differences) -- strict M test (b cancels in pair diffs).
      [B] Per-lineage roll-forward R^2 with constant Bu and time-varying bu_k.
      [C] Centered PCA on per-lineage residuals b^l_k -- forcing dimensionality.
      [D] ||Bu_const||, residual scatter.
      [E] Cross-lineage spread on Bu_const direction (validates shared-b).

    Note: per-lineage R^2 in [B] is inflated by the shared trajectory growth
    that every lineage trivially gets right since pred uses x^l_0. Read [B]
    together with [E] (per-lineage discrimination = cross-lineage spread of
    residuals) rather than alone.

    **_ignored_kwargs catches deprecated params like heldout_frac.
    """
    tc = traj_c[:, :, panel] if panel is not None else traj_c
    xb = xbar[panel] if (xbar is not None and panel is not None
                         and len(xbar) != tc.shape[2]) else xbar
    L, K, G = tc.shape
    out = {}

    # --- recover Bu_const and bu_per_step in RAW frame from mean trajectory ---
    mean_c = tc.mean(axis=0)
    if xb is not None:
        X = mean_c + xb[None, :]
    else:
        X = mean_c
        if verbose:
            print(f"  [warn] xbar is None -- Bu in centered frame "
                  f"(the A*xbar artifact will leak in)")
    bu_per_step = np.array([X[k + 1] - M @ X[k] for k in range(K - 1)])
    Bu_const = bu_per_step.mean(axis=0)
    resid_std = bu_per_step.std(axis=0)
    rel_scatter = float(norm(resid_std) / (norm(Bu_const) + 1e-12))
    out["bu_per_step"] = bu_per_step
    out["Bu_const"] = Bu_const
    out["rel_scatter"] = rel_scatter

    if verbose:
        print(f"  HEALTH CHECK on fitted operator (panel G={G}, L={L} lineages)")

    # --- [A] R^2 of M on pair differences ---
    rng = np.random.default_rng(seed)
    n_pairs = max(2, min(n_pairs_holdout, L))
    Ip = rng.integers(0, L, n_pairs)
    Jp = rng.integers(0, L, n_pairs)
    same = (Ip == Jp); Jp[same] = (Jp[same] + 1) % L
    ss_res = ss_tot = 0.0
    for p in range(n_pairs):
        y_traj = tc[Ip[p]] - tc[Jp[p]]
        pred = (M @ y_traj[:-1].T).T
        ss_res += norm(y_traj[1:] - pred) ** 2
        ss_tot += norm(y_traj[1:]) ** 2
    r2_M_pair = float(1.0 - ss_res / (ss_tot + 1e-12))
    out["r2_M_on_pair_diffs"] = r2_M_pair
    if verbose:
        print(f"\n  [A] R^2(M on pair differences) = {r2_M_pair:+.4f}  "
              f"({n_pairs} random pairs)")

    # --- subsample lineages for [B], [C], [E] ---
    if lineage_sample is not None and lineage_sample < L:
        rng2 = np.random.default_rng(seed + 1)
        lin_idx = rng2.choice(L, lineage_sample, replace=False)
    else:
        lin_idx = np.arange(L)
    L_use = len(lin_idx)
    xb_use = xb if xb is not None else np.zeros(G)
    Mt = M.T

    # --- [B] Per-lineage roll-forward R^2 ---
    X_init = tc[lin_idx, 0, :] + xb_use[None, :]
    pred_const = X_init.copy()
    pred_tv = X_init.copy()
    ss_res_const = ss_res_tv = ss_tot_lin = 0.0
    for k in range(K - 1):
        X_actual = tc[lin_idx, k + 1, :] + xb_use[None, :]
        pred_const = pred_const @ Mt + Bu_const
        pred_tv = pred_tv @ Mt + bu_per_step[k]
        deviation = X_actual - X_init
        ss_tot_lin += float(np.sum(deviation ** 2))
        ss_res_const += float(np.sum((X_actual - pred_const) ** 2))
        ss_res_tv += float(np.sum((X_actual - pred_tv) ** 2))
    r2_lin_const = float(1.0 - ss_res_const / (ss_tot_lin + 1e-12))
    r2_lin_tv = float(1.0 - ss_res_tv / (ss_tot_lin + 1e-12))
    out["r2_lineage_const_Bu"] = r2_lin_const
    out["r2_lineage_tv_bu"] = r2_lin_tv
    if verbose:
        print(f"\n  [B] Per-lineage roll-forward R^2 ({L_use} lineages):")
        print(f"    constant Bu:        R^2 = {r2_lin_const:+.4f}")
        print(f"    time-varying bu_k:  R^2 = {r2_lin_tv:+.4f}")

    # --- [C] Centered PCA on per-lineage forcings (b^l_k vectors) ---
    sum_f = np.zeros(G, dtype=np.float64)
    C_uncentered = np.zeros((G, G), dtype=np.float64)
    N_total = 0
    for k in range(K - 1):
        F_k = (tc[lin_idx, k + 1, :] + xb_use[None, :]) \
              - (tc[lin_idx, k, :] + xb_use[None, :]) @ Mt
        sum_f += F_k.sum(axis=0)
        C_uncentered += F_k.T @ F_k
        N_total += F_k.shape[0]
    mean_f = sum_f / N_total
    C_centered = C_uncentered - N_total * np.outer(mean_f, mean_f)
    eigvals = np.linalg.eigvalsh(C_centered)[::-1]
    eigvals = np.maximum(eigvals, 0.0)
    total_var = float(eigvals.sum())
    explained = eigvals / (total_var + 1e-12)
    sv = np.sqrt(eigvals)
    sv_norm = sv / (sv[0] + 1e-12)
    out["forcing_pca_sv"] = sv
    out["forcing_pca_explained"] = explained
    if verbose:
        print(f"\n  [C] Centered PCA on {N_total} per-lineage b^l_k vectors:")
        cum = 0.0
        for i in range(min(8, len(sv))):
            cum += explained[i]
            print(f"    PC[{i+1}]  sv_norm={sv_norm[i]:.4f}  "
                  f"explains {explained[i]*100:5.2f}%  cum={cum*100:5.2f}%")
        print(f"    Top 4 cumulative: {explained[:4].sum()*100:.1f}%   "
              f"Top 8: {explained[:8].sum()*100:.1f}%")

    # --- [D] Bu_const + scatter ---
    if verbose:
        print(f"\n  [D] ||Bu_const|| = {norm(Bu_const):.4f}   "
              f"residual scatter ||std||/||mean|| = {rel_scatter:.3f}")

    # --- [E] Cross-lineage spread on Bu_const direction ---
    if norm(Bu_const) < 1e-12:
        if verbose:
            print(f"\n  [E] Bu_const ~ 0 -- skipping cross-lineage spread")
        return out
    u_hat = Bu_const / norm(Bu_const)
    Xa_sub = tc[lin_idx] + xb_use[None, None, :]
    res_std = np.empty(K - 1)
    st_std  = np.empty(K - 1)
    for k in range(K - 1):
        Rk = Xa_sub[:, k + 1, :] - Xa_sub[:, k, :] @ Mt
        proj_k = Rk @ u_hat
        Sk = Xa_sub[:, k, :]
        state_proj_k = (Sk - Sk.mean(axis=0)) @ u_hat
        res_std[k] = proj_k.std()
        st_std[k]  = state_proj_k.std()
    ratio_med = float(np.median(res_std) / (np.median(st_std) + 1e-12))
    out["cross_lineage_ratio_median"] = ratio_med
    if verbose:
        print(f"\n  [E] Cross-lineage spread on Bu_const direction:")
        print(f"    state std median  = {np.median(st_std):.3f}")
        print(f"    resid std median  = {np.median(res_std):.3f}")
        print(f"    ratio (res/state) median = {ratio_med:.3f}")

    return out
