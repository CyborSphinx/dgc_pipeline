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


def recover_input_and_validate(M, traj_c, panel=None,
                               n_pairs_holdout=200, lineage_sample=None,
                               seed=0, verbose=True, **_ignored_kwargs):
    """
    Health check for a fitted operator M. All computations in CENTERED frame.

    Prints (and returns):
      [A] R^2(M on pair differences) -- strict M test (b cancels in pair diffs).
      [B] Per-lineage roll-forward R^2 with constant Bu and time-varying bu_k.
      [C] Centered PCA on per-lineage residuals b^l_k -- forcing dimensionality.
      [D] ||Bu_const_c||, residual scatter (centered frame).
      [E] Cross-lineage spread on Bu_const direction.

    M: (G, G) operator (fit on centered data, frame-invariant)
    traj_c: (L, K, G) CENTERED trajectories
    panel: optional gene-index subset

    Note: per-lineage R^2 in [B] is inflated by shared trajectory growth that
    every lineage trivially gets right (pred uses x^l_0). Read [B] together
    with [E].

    **_ignored_kwargs catches deprecated params (xbar, heldout_frac).
    """
    tc = traj_c[:, :, panel] if panel is not None else traj_c
    L, K, G = tc.shape
    out = {}

    # --- recover Bu_const and bu_per_step in CENTERED frame from mean trajectory ---
    mean_c = tc.mean(axis=0)                              # (K, G) centered mean
    bu_per_step = np.array([mean_c[k + 1] - M @ mean_c[k] for k in range(K - 1)])
    Bu_const = bu_per_step.mean(axis=0)
    resid_std = bu_per_step.std(axis=0)
    rel_scatter = float(norm(resid_std) / (norm(Bu_const) + 1e-12))
    out["bu_per_step"] = bu_per_step
    out["Bu_const"] = Bu_const
    out["rel_scatter"] = rel_scatter

    if verbose:
        print(f"  HEALTH CHECK on fitted operator (G={G}, L={L} lineages, centered)")

    # --- [A] R^2 of M on pair differences (centered; xbar cancels in diff anyway) ---
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
    Mt = M.T

    # --- [B] Per-lineage roll-forward R^2 (CENTERED frame) ---
    y_init = tc[lin_idx, 0, :]                            # (L_use, G) centered initial
    pred_const = y_init.copy()
    pred_tv = y_init.copy()
    ss_res_const = ss_res_tv = ss_tot_lin = 0.0
    for k in range(K - 1):
        y_actual = tc[lin_idx, k + 1, :]
        pred_const = pred_const @ Mt + Bu_const
        pred_tv = pred_tv @ Mt + bu_per_step[k]
        deviation = y_actual - y_init
        ss_tot_lin += float(np.sum(deviation ** 2))
        ss_res_const += float(np.sum((y_actual - pred_const) ** 2))
        ss_res_tv += float(np.sum((y_actual - pred_tv) ** 2))
    r2_lin_const = float(1.0 - ss_res_const / (ss_tot_lin + 1e-12))
    r2_lin_tv = float(1.0 - ss_res_tv / (ss_tot_lin + 1e-12))
    out["r2_lineage_const_Bu"] = r2_lin_const
    out["r2_lineage_tv_bu"] = r2_lin_tv
    if verbose:
        print(f"\n  [B] Per-lineage roll-forward R^2 ({L_use} lineages, centered):")
        print(f"    constant Bu:        R^2 = {r2_lin_const:+.4f}")
        print(f"    time-varying bu_k:  R^2 = {r2_lin_tv:+.4f}")

    # --- [C] Centered PCA on per-lineage forcings (b^l_k vectors, centered) ---
    sum_f = np.zeros(G, dtype=np.float64)
    C_uncentered = np.zeros((G, G), dtype=np.float64)
    N_total = 0
    for k in range(K - 1):
        F_k = tc[lin_idx, k + 1, :] - tc[lin_idx, k, :] @ Mt
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

    # --- [D] Bu_const + scatter (centered frame) ---
    if verbose:
        print(f"\n  [D] ||Bu_const_c|| = {norm(Bu_const):.4f}   "
              f"residual scatter ||std||/||mean|| = {rel_scatter:.3f}")

    # --- [E] Cross-lineage spread on Bu_const direction (centered) ---
    if norm(Bu_const) < 1e-12:
        if verbose:
            print(f"\n  [E] Bu_const ~ 0 -- skipping cross-lineage spread")
        return out
    u_hat = Bu_const / norm(Bu_const)
    tc_sub = tc[lin_idx]                                  # centered
    res_std = np.empty(K - 1)
    st_std  = np.empty(K - 1)
    for k in range(K - 1):
        Rk = tc_sub[:, k + 1, :] - tc_sub[:, k, :] @ Mt
        proj_k = Rk @ u_hat
        Sk = tc_sub[:, k, :]
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


def multi_horizon_comparison(traj_c, M, M_pure_dmd=None, panel=None,
                              horizons=None, seed=0, holdout_frac=0.0,
                              verbose=True):
    """
    Compare prediction quality across horizons for five models. CENTERED frame.

      (1) M + const Bu        : paired-fit M, time-averaged residual
      (2) M + tv b_t          : paired-fit M, per-step empirical residual
      (3) pure forcing const  : y_{k+1} = y_k + db_const  (NO operator)
      (4) pure forcing tv     : y_{k+1} = y_k + db_t      (NO operator)
      (5) pure DMD (no b)     : y_{k+1} = M_pure_dmd y_k  (no explicit forcing)

    Two metrics at each horizon h:

      R^2(h) = 1 - Σ_l ||y^l_h - pred^l_h||^2 / Σ_l ||y^l_h - mean_l(y^l_h)||^2
        Denominator is cross-lineage VARIANCE at step h. Standard per-horizon R^2;
        does NOT artificially grow with horizon (unlike the older
        ||y^l_h - y^l_0||^2 denominator, which inflates R^2 at long h).

      RMSE(h) = sqrt(mean_l ||y^l_h - pred^l_h||^2)
        Direct error magnitude in centered state units. SHOULD grow with h
        (error compounding); inspect this to verify the comparison is honest.

    holdout_frac:
      0.0 -> in-sample evaluation. tv baselines are TRIVIALLY high because b_t
        IS the mean per-step change of the same data we're predicting.
      0.2 -> hold out 20% of lineages. b_t, Bu_const, db_const, db_t are
        estimated from the TRAINING 80%, then predictions are evaluated on
        the held-out 20%. This is the honest out-of-sample comparison; tv b_t
        no longer trivially wins.

    The pure-DMD operator M_pure_dmd is used as-is (fit upstream by the user).
    If you want a fully OOS DMD comparison, refit M_pure_dmd on the training
    set externally and pass it here.

    Returns dict with keys 'R2', 'RMSE', 'train_idx', 'test_idx'.
    """
    tc = traj_c[:, :, panel] if panel is not None else traj_c
    L, K, G = tc.shape
    if horizons is None:
        horizons = [h for h in (1, 2, 4, 8, 16, 32, 36) if h < K]

    # ----- train/test split for OOS evaluation of b-derived quantities ------
    rng = np.random.default_rng(seed)
    if holdout_frac > 0.0:
        n_test = max(1, int(holdout_frac * L))
        test_idx = rng.choice(L, n_test, replace=False)
        train_mask = np.ones(L, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
    else:
        train_idx = np.arange(L)
        test_idx = np.arange(L)

    tc_train = tc[train_idx]
    tc_test = tc[test_idx]
    L_test = tc_test.shape[0]

    # ----- recover all b-related quantities from TRAINING data only ---------
    mean_c_train = tc_train.mean(axis=0)                  # (K, G)
    bu_per_step = np.array([mean_c_train[k + 1] - M @ mean_c_train[k]
                            for k in range(K - 1)])
    Bu_const = bu_per_step.mean(axis=0)
    db_per_step = np.diff(mean_c_train, axis=0)
    db_const = db_per_step.mean(axis=0)

    Mt = M.T
    M_dmd_T = M_pure_dmd.T if M_pure_dmd is not None else None

    def roll(M_op_T, b_step, b_const):
        y = np.empty((L_test, K, G))
        y[:, 0, :] = tc_test[:, 0, :]
        for k in range(K - 1):
            if M_op_T is not None:
                y[:, k + 1, :] = y[:, k, :] @ M_op_T
            else:
                y[:, k + 1, :] = y[:, k, :]
            if b_step is not None:
                y[:, k + 1, :] += b_step[k]
            elif b_const is not None:
                y[:, k + 1, :] += b_const
        return y

    models = {
        "M + const Bu":        roll(Mt, None, Bu_const),
        "M + tv b_t":          roll(Mt, bu_per_step, None),
        "pure forcing const":  roll(None, None, db_const),
        "pure forcing tv":     roll(None, db_per_step, None),
    }
    if M_pure_dmd is not None:
        models["pure DMD (no b)"] = roll(M_dmd_T, None, None)

    # ----- metrics --------------------------------------------------------
    results_r2 = {name: {} for name in models}
    results_rmse = {name: {} for name in models}
    var_check = {}  # cross-lineage variance at each h (for sanity)
    for h in horizons:
        y_actual_h = tc_test[:, h, :]
        mean_h = y_actual_h.mean(axis=0)
        ss_tot_var = float(np.sum((y_actual_h - mean_h) ** 2))     # cross-lin var
        var_check[h] = ss_tot_var / max(L_test, 1)
        for name, y_pred in models.items():
            err = y_actual_h - y_pred[:, h, :]
            ss_res = float(np.sum(err ** 2))
            results_r2[name][h] = 1.0 - ss_res / (ss_tot_var + 1e-12)
            results_rmse[name][h] = float(np.sqrt(np.mean(err ** 2)))

    if verbose:
        label = ("OOS" if holdout_frac > 0 else "IN-SAMPLE")
        print(f"  MULTI-HORIZON COMPARISON  ({L_test} test lineages, {label}, "
              f"centered frame)")
        if holdout_frac == 0:
            print(f"  WARNING: in-sample. tv b_t baselines are trivially ~1.0 "
                  f"because b_t IS the data's mean per-step change.")
            print(f"  Use holdout_frac=0.2 for an honest OOS comparison.\n")
        else:
            print(f"  b_t / Bu_const / db_* estimated from {len(train_idx)} "
                  f"training lineages; evaluated on {L_test} held-out.\n")

        print(f"  R^2(h) = 1 - Σ_l ||y_h - pred||^2 / Σ_l ||y_h - mean_l(y_h)||^2")
        print(f"  (cross-lineage variance denominator; does NOT inflate with h)\n")
        header = "  " + f"{'model':<24}" + "  ".join(f"h={h:>3}" for h in horizons)
        print(header)
        for name in models:
            row = "  ".join(f"{results_r2[name][h]:+.3f}" for h in horizons)
            print(f"  {name:<24}{row}")

        print(f"\n  RMSE(h) (absolute error, centered state units; SHOULD grow w/ h)\n")
        print(header)
        for name in models:
            row = "  ".join(f"{results_rmse[name][h]:>6.3f}" for h in horizons)
            print(f"  {name:<24}{row}")

        # sanity: how does cross-lineage variance evolve with h?
        print(f"\n  cross-lineage std(y_h) at each h (sanity check):")
        for h in horizons:
            print(f"    h={h:>3}: std={np.sqrt(var_check[h]):.4f}")

    return {"R2": results_r2, "RMSE": results_rmse,
            "train_idx": train_idx, "test_idx": test_idx,
            "var_check": var_check}


