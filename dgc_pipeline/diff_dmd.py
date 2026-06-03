"""
diff_dmd.py — difference-DMD: recover the natural state-transition operator by
cancelling the uniform (OSKM) input via pairwise differencing.
=============================================================================

PREMISE (the project's linear assumption):
  Single global linear model   x_{k+1} = (I + A) x_k + B u_k
  with B u_k a POPULATION CONSTANT (OSKM applied uniformly to all cells via Dox).

For a FIXED pair of trajectories (i, j), define the difference series
  Δx_k := x_k^i - x_k^j.
Then within that pair, across consecutive days:
  Δx_{k+1} = (I+A) Δx_k          [B u_k cancels EXACTLY, no need to know it]
So each pair's difference series is itself a valid one-step DMD time series.

CRITICAL (DMD validity):
  We must NOT concatenate columns from different series into one DMD solve --
  that would ask a single operator to predict the (meaningless) transition from
  one series' snapshot to another's. The ONLY valid construction is:
     fit ONE DMD operator per difference-series, then FUSE the operators.
  This is identical to how the raw-trajectory path fits one operator per lineage
  and fuses, so we REUSE that exact machinery: we build a (P, K, G) array of P
  difference-series and hand it to build_fused_operator unchanged.

Pairing regimes (to test the global-linear premise):
  - close : same-fate, nearest-neighbour trajectory pairs (most local)
  - all   : random trajectory pairs
  - cross : low-iPSC-fate vs high-iPSC-fate pairs (max dynamical contrast)
"""

import os
import numpy as np
from numpy.linalg import norm


def _pair_indices(traj, fate, regime, n_pairs, knn, rng):
    L = traj.shape[0]
    Xflat = traj.reshape(L, -1)
    # Pairs needed ~ L, NOT L^2: all Δ=series_i-series_j live in the rank-L span of
    # the originals, so O(L) well-chosen pairs capture the independent structure.
    # More pairs are redundant and blow up memory (each contributes ~rank cols).
    budget = n_pairs if n_pairs else L
    if regime == "close":
        I, J = _knn_same_fate_pairs(Xflat, fate, knn, rng)
    elif regime == "cross":
        I, J = _cross_fate_pairs(Xflat, fate, budget, rng)
    else:
        m = min(budget, L * (L - 1) // 2)
        I, J = rng.integers(0, L, size=m), rng.integers(0, L, size=m)
    # enforce the budget uniformly (close can overproduce via kNN)
    if len(I) > budget:
        sel = rng.choice(len(I), size=budget, replace=False)
        I, J = I[sel], J[sel]
    return I, J


def _knn_same_fate_pairs(X, fate, knn, rng, fate_bins=3):
    n = len(X)
    if n < 2:
        return np.array([], int), np.array([], int)
    if fate is None:
        strata = [np.arange(n)]
    else:
        q = np.quantile(fate, np.linspace(0, 1, fate_bins + 1))
        lab = np.clip(np.digitize(fate, q[1:-1]), 0, fate_bins - 1)
        strata = [np.where(lab == b)[0] for b in range(fate_bins)]
    I, J = [], []
    for idx in strata:
        if len(idx) < 2:
            continue
        Xs = X[idx]
        d2 = (np.sum(Xs ** 2, 1)[:, None] + np.sum(Xs ** 2, 1)[None, :]
              - 2 * Xs @ Xs.T)
        np.fill_diagonal(d2, np.inf)
        k = min(knn, len(idx) - 1)
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        for a in range(len(idx)):
            for b in nn[a]:
                I.append(idx[a]); J.append(idx[b])
    return np.array(I, int), np.array(J, int)


def _cross_fate_pairs(X, fate, n_pairs, rng, lo_q=0.33, hi_q=0.67):
    if fate is None:
        return np.array([], int), np.array([], int)
    lo = np.where(fate <= np.quantile(fate, lo_q))[0]
    hi = np.where(fate >= np.quantile(fate, hi_q))[0]
    if len(lo) == 0 or len(hi) == 0:
        return np.array([], int), np.array([], int)
    m = min(n_pairs, len(lo) * len(hi))
    return rng.choice(lo, size=m), rng.choice(hi, size=m)


def _difference_series(traj, I, J):
    """(P, K, G) array of difference SERIES: entry p is Δx_k = traj[I[p]]-traj[J[p]]."""
    return traj[I] - traj[J]


def _r2_on_series(M, series):
    """
    R^2 of one operator M predicting one-step transitions WITHIN each series
    (never across series boundaries). Within-series (x_k -> x_{k+1}) columns only.
      r2_step    : 1 - ||x_{k+1}-M x_k||^2 / ||x_{k+1}||^2   (flattered by persistence)
      r2_persist : 1 - ||x_{k+1}-x_k||^2  / ||x_{k+1}||^2    (identity baseline)
      r2_change  : 1 - ||(x_{k+1}-x_k)-(M-I)x_k||^2 / ||x_{k+1}-x_k||^2   (HONEST)
    """
    P, K, G = series.shape
    Xk = series[:, :-1, :].reshape(-1, G).T
    Xk1 = series[:, 1:, :].reshape(-1, G).T
    pred = M @ Xk
    ss_next = norm(Xk1) ** 2 + 1e-12
    r2_step = 1.0 - norm(Xk1 - pred) ** 2 / ss_next
    r2_persist = 1.0 - norm(Xk1 - Xk) ** 2 / ss_next
    change = Xk1 - Xk
    pred_change = (M - np.eye(G)) @ Xk
    ss_change = norm(change) ** 2 + 1e-12
    r2_change = 1.0 - norm(change - pred_change) ** 2 / ss_change
    return r2_step, r2_persist, r2_change


def recover_Bu_and_predict(M, mean_traj, traj_all=None, xbar=None):
    """
    With the natural operator M=I+A known, the forced model is x_{k+1}=M x_k + Bu_k.
    The per-step residual r_k := x_{k+1} - M x_k is the input term at step k.

    IMPORTANT — coordinate frame: the operator M and the trajectories here are in
    CENTERED coordinates (xbar subtracted upstream for the DMD fit). The input Bu lives
    in RAW coordinates. Computing the residual in centered coordinates gives
        r_centered = Bu + (M - I) xbar = Bu + A xbar,
    an A*xbar artifact that has nothing to do with the control input and can dominate it
    (xbar is the large positive expression baseline). So if xbar (panel-restricted) is
    provided we UN-CENTER first and recover Bu in raw coordinates:
        x_raw = x_centered + xbar,  Bu = x_raw_{k+1} - M x_raw_k.
    (Equivalently Bu = r_centered - A xbar.) The difference-DMD fit of M is unaffected
    by centering since xbar cancels in differences; only this input recovery must use
    the raw frame.

    CONSISTENCY ACROSS LINEAGES (scatter): if traj_all (L,K,G) is given, per-lineage
    residuals are projected onto u_hat = Bu_const/||Bu_const|| (also in raw frame).

    mean_traj: (K, G) centered. traj_all: (L, K, G) centered or None.
    xbar: (G,) panel-restricted mean to un-center; if None, residual stays centered
    (legacy, CONTAMINATED -- only for back-compat).
    """
    # un-center to raw coordinates so the recovered input has no A*xbar artifact
    if xbar is not None:
        X = mean_traj + xbar[None, :]
        Xa = (traj_all + xbar[None, None, :]) if traj_all is not None else None
    else:
        X = mean_traj
        Xa = traj_all
    K = X.shape[0]
    R = np.array([X[k + 1] - M @ X[k] for k in range(K - 1)])   # (K-1, G) residuals (raw)
    Bu_const = R.mean(axis=0)                                   # constant estimate
    resid_std = R.std(axis=0)
    rel_scatter = norm(resid_std) / (norm(Bu_const) + 1e-12)

    def _roll(use_const):
        x = X[0].copy(); pred = [x.copy()]
        for k in range(K - 1):
            u = Bu_const if use_const else R[k]
            x = M @ x + u; pred.append(x.copy())
        return np.array(pred)

    def _r2(pred):
        return 1.0 - norm(X[1:] - pred[1:]) ** 2 / (norm(X[1:] - X[0]) ** 2 + 1e-12)

    r2_const = _r2(_roll(True))
    r2_tv = _r2(_roll(False))
    r2_nat = 1.0 - norm(X[1:] - np.array([np.linalg.matrix_power(M, k+1) @ X[0]
                                          for k in range(K - 1)])) ** 2 \
        / (norm(X[1:] - X[0]) ** 2 + 1e-12)

    # per-lineage per-step residuals projected onto the shared input direction
    proj_by_step = None
    state_proj_by_step = None
    if Xa is not None and norm(Bu_const) > 1e-12:
        u_hat = Bu_const / norm(Bu_const)
        proj = np.empty((K - 1, Xa.shape[0]))
        state_proj = np.empty((K - 1, Xa.shape[0]))
        for k in range(K - 1):
            Rk = Xa[:, k + 1, :] - (Xa[:, k, :] @ M.T)         # (L, G) residuals
            proj[k] = Rk @ u_hat                               # input projection
            # raw STATE spread across lineages at this step (same projection),
            # mean-centered so we measure cross-lineage variance, not the level.
            Sk = Xa[:, k, :]                                   # (L, G) states
            state_proj[k] = (Sk - Sk.mean(axis=0)) @ u_hat
        proj_by_step = proj
        state_proj_by_step = state_proj
        # summary: cross-lineage std of residual vs of state, per step
        res_std = proj.std(axis=1)
        st_std = state_proj.std(axis=1)
        # report ratio: if residuals are tight RELATIVE to how spread the
        # trajectories themselves are, the input is a genuine shared signal,
        # not an averaging artifact.
        print(f"            cross-lineage spread (proj on u_hat): "
              f"state std median={np.median(st_std):.3f}, "
              f"residual std median={np.median(res_std):.3f}, "
              f"ratio(res/state)={np.median(res_std)/(np.median(st_std)+1e-12):.3f}")
        print(f"              (residual << state => tight input is REAL, not inherited "
              f"from near-identical trajectories)")
        # PER-STEP scatter list: is Bu state-independent at EVERY step, or only
        # on average? Watch the fast early steps (k=0..6), where a state-dependent
        # B(x) would show up as an inflated res/state ratio.
        mean_proj = proj.mean(axis=1)                      # mean Bu projection per step
        print("            per-step Bu consistency "
              "[k: day  res_std  state_std  ratio  mean_proj]:")
        for k in range(K - 1):
            ratio_k = res_std[k] / (st_std[k] + 1e-12)
            day_k = 0.5 * k
            flag = "  <-- high" if ratio_k > 0.2 else ""
            print(f"              {k:2d}: {day_k:4.1f}  {res_std[k]:7.4f}  "
                  f"{st_std[k]:8.4f}  {ratio_k:6.3f}  {mean_proj[k]:+8.3f}{flag}")

    return dict(Bu_const=Bu_const, rel_scatter=rel_scatter,
                r2_const=r2_const, r2_tv=r2_tv, r2_nat=r2_nat,
                resid_norm_by_step=np.linalg.norm(R, axis=1),
                dev_from_const=np.linalg.norm(R - Bu_const, axis=1),
                proj_by_step=proj_by_step,
                state_proj_by_step=state_proj_by_step)


def build_diff_dmd_operator(traj, fate=None, regimes=("close", "all", "cross"),
                            n_pairs=4000, knn=5, ridge=0.0, sigma_cap=None,
                            rank_tol=1e-8, n_steps_hint=38, seed=0,
                            fuse_fn=None, fuse_weight="count", max_rank=None,
                            eps_fuse=0.0, dropout=0.0, days=None, xbar_panel=None):
    """
    Per regime: pick trajectory pairs -> build per-pair DIFFERENCE SERIES ->
    fit ONE DMD per series and FUSE (via fuse_fn = build_fused_operator, the same
    routine used for raw trajectories; NO concatenation) -> read R^2 off the
    single fused operator on BOTH difference and state series.
    Returns ("diff_plus_I", D_fused) for the first regime, plus diagnostics.
    """
    if fuse_fn is None:
        raise ValueError("build_diff_dmd_operator requires fuse_fn=build_fused_operator")

    print("\n--- Difference-DMD (per-pair DMD then fuse; OSKM cancelled) ---")
    results = {}
    primary = None
    primary_Bu_const = None
    for reg in regimes:
        I, J = _pair_indices(traj, fate, reg, n_pairs, knn,
                             np.random.default_rng(seed))
        if len(I) == 0:
            print(f"  [{reg:>5}] no pairs (missing fate?) — skipped")
            continue
        dseries = _difference_series(traj, I, J)        # (P, K, G) difference SERIES

        # Fuse per-pair operators with the SAME machinery as the trajectory path:
        # build_fused_operator fits one DMD per series (axis 0) and fuses. We pass
        # the difference series as if trajectories. No cross-series concatenation.
        (_, D_fused), finfo = fuse_fn(dseries, rank_tol=rank_tol,
                                      weight=fuse_weight, gene_panel=None,
                                      ridge=ridge, n_steps_hint=n_steps_hint,
                                      sigma_cap=sigma_cap, max_rank=max_rank,
                                      eps_fuse=eps_fuse)
        G = D_fused.shape[0]
        M = np.eye(G) + D_fused

        d_step, d_persist, d_change = _r2_on_series(M, dseries)
        s_step, s_persist, s_change = _r2_on_series(M, traj)
        mean_traj = traj.mean(axis=0)                       # (K, G) observed mean

        rho = finfo.get("rho", float("nan"))
        results[reg] = dict(P=len(I), rho=rho, d_change=d_change, s_change=s_change)
        # concise one-line summary per regime
        print(f"  [{reg:>5}] pairs={len(I):>5} rho={rho:.3f}  "
              f"R2_diff(change)={d_change:+.3f}  R2_state(change)={s_change:+.3f}")

        if primary is None:
            primary = ("diff_plus_I", D_fused)
            # --- dropout: refit on a fraction of pairs, score on the held-out pairs ---
            if dropout and len(I) >= 20:
                rngd = np.random.default_rng(seed + 7)
                perm = rngd.permutation(len(I))
                cut = int(len(I) * (1.0 - dropout))
                Itr, Jtr = I[perm[:cut]], J[perm[:cut]]
                Ite, Jte = I[perm[cut:]], J[perm[cut:]]
                dtr = _difference_series(traj, Itr, Jtr)
                (_, Dtr), _ = fuse_fn(dtr, rank_tol=rank_tol, weight=fuse_weight,
                                      gene_panel=None, ridge=ridge,
                                      n_steps_hint=n_steps_hint, sigma_cap=sigma_cap,
                                      max_rank=max_rank, eps_fuse=eps_fuse)
                Mtr = np.eye(G) + Dtr
                dte = _difference_series(traj, Ite, Jte)
                _, _, r2_held = _r2_on_series(Mtr, dte)
                print(f"          dropout: fit on {cut} pairs, R2_diff(change) on "
                      f"{len(Ite)} HELD-OUT pairs = {r2_held:+.3f}  (vs {d_change:+.3f} in-sample)")

            # --- recover Bu (CONSTANT first) and forward-predict the state trajectory ---
            bd = recover_Bu_and_predict(M, mean_traj, traj_all=traj, xbar=xbar_panel)
            primary_Bu_const = bd["Bu_const"]      # recovered input v_bar (constant)
            print(f"          full-model state prediction (x_(k+1)=M x_k + Bu):")
            print(f"            constant Bu:  R2={bd['r2_const']:+.3f}   "
                  f"(natural-only, no Bu: R2={bd['r2_nat']:+.3f})")
            print(f"            residual scatter ||std||/||mean|| = {bd['rel_scatter']:.2f}  "
                  f"({'constant Bu OK' if bd['rel_scatter'] < 0.5 else 'HIGH -> consider time-varying'})")
            print(f"            time-varying Bu_k: R2={bd['r2_tv']:+.3f}  "
                  f"(gain over constant: {bd['r2_tv'] - bd['r2_const']:+.3f})")
            # SCATTER per step: each lineage's input residual projected onto the
            # shared input direction. Vertical SPREAD at a step = how (in)consistent
            # lineages are about the input there; tight = coherent shared signal.
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                P = bd["proj_by_step"]                         # (K-1, L) or None
                if P is None:
                    raise ValueError("no per-lineage projections")
                nstep = P.shape[0]
                xs = (np.array(days[:nstep]) if (days is not None and len(days) >= nstep)
                      else np.arange(nstep))
                fig, ax = plt.subplots(figsize=(9, 4))
                # subsample lineages for plotting if very many
                Lp = P.shape[1]
                idx = (np.random.default_rng(0).choice(Lp, 800, replace=False)
                       if Lp > 800 else np.arange(Lp))
                for s in range(nstep):
                    xj = xs[s] + (np.random.default_rng(s).random(len(idx)) - 0.5) * \
                        (0.6 * (xs[1] - xs[0]) if nstep > 1 else 0.3)
                    ax.scatter(xj, P[s, idx], s=3, alpha=0.12,
                               color="#3b82f6", edgecolors="none")
                # overlay per-step mean and +/-1 std
                mean_s = P.mean(axis=1); std_s = P.std(axis=1)
                ax.plot(xs, mean_s, color="#dc2626", lw=1.6, marker="o", ms=3,
                        label="input mean", zorder=5)
                ax.fill_between(xs, mean_s - std_s, mean_s + std_s, color="#dc2626",
                                alpha=0.18, label="input ±1 std (across lineages)", zorder=4)
                # overlay the cross-lineage STATE spread (mean-centered) as a band:
                # if trajectories are widely spread here but the input band is thin,
                # the tight input is REAL, not inherited from identical trajectories.
                SP = bd.get("state_proj_by_step")
                if SP is not None:
                    st_std = SP.std(axis=1)
                    ax.fill_between(xs, -st_std, st_std, color="#6b7280", alpha=0.15,
                                    label="state ±1 std (lineage spread)", zorder=2)
                ax.axhline(0, color="k", lw=0.5, alpha=0.4)
                ax.set_xlabel("day" if days is not None else "step")
                ax.set_ylabel("per-lineage input projection  $r_k^i \\cdot \\hat u$")
                ax.set_title("Per-step input across lineages (tight spread => coherent "
                             "shared input; watch for change at Dox-off ~day 8)")
                ax.legend(loc="best", fontsize=8)
                fig.tight_layout()
                outp = os.path.join(os.getcwd(), "Bu_consistency_scatter.png")
                fig.savefig(outp, dpi=130); plt.close(fig)
                print(f"            [scatter saved: {outp}]")
            except Exception as e:
                print(f"            [scatter skipped: {e}]")

    if not results:
        raise RuntimeError("difference-DMD produced no operator (check fate/inputs)")

    if len(results) >= 2:
        r2s = {k: v["d_change"] for k, v in results.items()}
        spread = max(r2s.values()) - min(r2s.values())
        print(f"\n  GLOBAL-LINEAR PREMISE TEST (R^2-change on difference series):")
        print(f"    spread across regimes = {spread:.3f}")
        if spread < 0.05:
            print("    -> regimes AGREE: single global A supported (any valid pair OK).")
        else:
            print("    -> regimes DIVERGE: evidence of state-dependence (not globally linear).")
        if "close" in r2s and "cross" in r2s:
            tag = ("consistent" if abs(r2s["close"] - r2s["cross"]) < 0.05
                   else "CROSS DEGRADES -> nonlinearity")
            print(f"       close={r2s['close']:+.3f} vs cross={r2s['cross']:+.3f} ({tag})")
        best = max(r2s.values())
        print(f"    best change-explained R^2 = {best:+.3f}  "
              f"({'linear dynamics carry signal' if best > 0.2 else 'LINEAR MODEL FITS ~NOTHING beyond persistence'})")

    return primary, {"diff_dmd_results": results,
                     "Bu_const": primary_Bu_const,
                     "rho": results[regimes[0]]["rho"] if regimes[0] in results else None}
