"""
Operator builders: the dimension-aware lineage fusion (the project's core
contribution) and the DGC rank-one mean-path comparator.

FusedOperatorBuilder implements CHOICE B fusion:
  for each lineage l, fit DX_l = A_l X_l (low-rank DMD on differences); let P_l
  project onto row(A_l). Then
      D_fused = ( sum_l w_l A_l P_l ) ( sum_l w_l P_l )^+
  computed via skinny factors (no per-lineage GxG) and an optional Tikhonov floor
  eps_fuse on the denominator (which is what actually fixes the fusion-manufactured
  instability, vs the abandoned sigma_cap post-clip).

The maths is byte-for-byte the original build_fused_operator; it now takes a
Backend for pinv/matmul instead of reaching for module globals, and returns a
DiffPlusIOperator instead of a ("diff_plus_I", D) tuple. All the diagnostic prints
are preserved (the spectra/rho/non-normality reporting the analysis relies on).

DGCOperatorBuilder builds the per-step rank-one A_k on the mean trajectory.
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import norm, svd

from .backend import Backend
from .operators import DiffPlusIOperator, Rank1Operator


class DGCOperatorBuilder:
    """DGC-style per-step rank-one operators on the mean (bulk) trajectory."""

    def build(self, traj_c):
        """traj_c: (L,K,G) centered. Returns (list[Rank1Operator], Xc_mean (G,K))."""
        Xc = traj_c.mean(axis=0).T              # (G, K)
        G, K = Xc.shape
        ops = []
        for k in range(K - 1):
            xk = Xc[:, k]
            d = xk @ xk
            if d < 1e-12:
                ops.append(Rank1Operator(np.zeros(G), xk))
            else:
                ops.append(Rank1Operator(Xc[:, k + 1] - xk, xk / d))
        return ops, Xc


class FusedOperatorBuilder:
    """Dimension-aware fusion over lineages -> one DiffPlusIOperator (M = I + D)."""

    def __init__(self, backend: Backend | None = None):
        self.backend = backend or Backend.active()

    def build(self, traj_c, rank_tol=1e-8, weight="count", gene_panel=None,
              ridge=0.0, n_steps_hint=38, sigma_cap=None, max_rank=None,
              explained_variance_threshold=None,
              eps_fuse=0.0,
              pair_lineages=False, pair_indices=None, n_pairs=None,
              pair_dropout=0.0, pair_seed=0):
        """
        Fit a fused operator M = I + D over lineages.

        Two kinds of differencing happen here. Get both right or the model is wrong:

        (1) TIME differencing within each (paired or single) trajectory:
              DX_k = X_{k+1} - X_k     ->   fit DX = A X   instead of   X' = M X
            This is for stable low-rank fitting.

        (2) LINEAGE pairing (NEW; off by default for backward compat):
              y^p_k = traj[I[p], k] - traj[J[p], k]
              fit on y^p instead of traj[l]
            By the model x_{k+1} = M x_k + b_k with b_k shared across lineages,
            taking the lineage difference CANCELS b_k.

        Rank control: two mutually exclusive options
          - max_rank=K          : every lineage uses the same rank K
          - explained_variance_threshold=t (in (0,1]) : per-lineage adaptive rank;
            choose smallest r such that cumsum(S^2)/sum(S^2) >= t. Reports the
            distribution of per-lineage ranks. The principled choice -- lineages
            with simpler structure get small r, complex lineages get higher r,
            and the user can see how heterogeneous the data is.

        Pair selection: if pair_indices=(I, J) is passed, use those. Else generate
        n_pairs random pairs (default L). pair_dropout in [0,1) randomly drops a
        fraction of pairs at fit time.
        """
        be = self.backend
        L_orig, K, Gfull = traj_c.shape
        if gene_panel is not None:
            traj_c = traj_c[:, :, gene_panel]
        G = traj_c.shape[2]
        if G > 4000:
            print(f"  [warn] fusion is G x G with G={G}; consider --gene_panel")

        # (2) lineage pairing -- cancels b_k by trajectory difference
        if pair_lineages:
            if pair_indices is None:
                rng = np.random.default_rng(pair_seed)
                budget = n_pairs if n_pairs is not None else L_orig
                I = rng.integers(0, L_orig, size=budget)
                J = rng.integers(0, L_orig, size=budget)
                # avoid self-pairing
                same = (I == J)
                if same.any():
                    J[same] = (J[same] + 1) % L_orig
            else:
                I, J = (np.asarray(pair_indices[0]),
                        np.asarray(pair_indices[1]))
            traj_c = traj_c[I] - traj_c[J]   # (P, K, G); shared b_k cancels
            print(f"  lineage pairing: {len(I)} difference series "
                  f"from L={L_orig} lineages")

        # pair / lineage dropout -- subsample for robustness / regularization
        if pair_dropout > 0:
            rng = np.random.default_rng(pair_seed + 1)
            keep = rng.random(traj_c.shape[0]) >= pair_dropout
            n_kept = int(keep.sum())
            if n_kept < 2:
                print(f"  [warn] pair_dropout={pair_dropout} left {n_kept} items; "
                      f"using all instead")
            else:
                traj_c = traj_c[keep]
                print(f"  pair dropout: keeping {n_kept}/{len(keep)} "
                      f"({100*n_kept/len(keep):.0f}%) at seed {pair_seed}")

        L = traj_c.shape[0]

        if (explained_variance_threshold is not None and max_rank is not None):
            print(f"  [warn] both explained_variance_threshold and max_rank set; "
                  f"using threshold (per-lineage adaptive)")

        W_blocks, U_blocks, ranks = [], [], []
        sigma_pre, sigma_post, fit_r2, comp_rhos = [], [], [], []
        for l in range(L):
            Xl = traj_c[l].T
            Xsnap = Xl[:, :-1]
            DX = Xl[:, 1:] - Xl[:, :-1]                 # (1) time differencing
            U, S, Vt = svd(Xsnap, full_matrices=False)
            r_avail = int(np.sum(S > rank_tol * (S[0] if S.size else 1.0)))
            n_snap = Xsnap.shape[1]
            if explained_variance_threshold is not None:
                # per-lineage adaptive: smallest r covering threshold of S^2 energy
                if r_avail == 0:
                    r = 0
                else:
                    cum = np.cumsum(S[:r_avail] ** 2) / (np.sum(S[:r_avail] ** 2) + 1e-12)
                    r = int(np.searchsorted(cum, explained_variance_threshold) + 1)
                    r = min(r, r_avail)
            else:
                cap = max_rank if max_rank is not None else max(2, n_snap // 2)
                r = min(r_avail, cap)
            if r == 0:
                continue
            Ur, Sr, Vtr = U[:, :r], S[:r], Vt[:r, :]
            filt = Sr / (Sr ** 2 + ridge)
            W_l = DX @ Vtr.T @ np.diag(filt)              # A_l = W_l @ Ur.T

            DX_pred = W_l @ (Ur.T @ Xsnap)
            ss_res = norm(DX - DX_pred) ** 2
            ss_tot = norm(DX) ** 2 + 1e-12
            fit_r2.append(1.0 - ss_res / ss_tot)

            try:
                small = np.eye(Ur.shape[1]) + (Ur.T @ W_l)
                comp_rhos.append(float(np.max(np.abs(np.linalg.eigvals(small)))))
            except Exception:
                pass

            if sigma_cap is not None:
                M_active = np.concatenate([Ur, W_l], axis=1)
                Q, _ = np.linalg.qr(M_active)
                T = Q.T @ Q + (Q.T @ W_l) @ (Ur.T @ Q)
                uu, ss, vv = svd(T)
                sigma_pre.append(float(ss.max()))
                ss_cap = np.minimum(ss, sigma_cap)
                sigma_post.append(float(ss_cap.max()))
                T_cap = (uu * ss_cap) @ vv
                q = Q.shape[1]
                W_l = Q @ (T_cap - np.eye(q))
                Ur = Q

            if weight == "fit":
                A_l_app = W_l @ Ur.T @ Xsnap
                w = max(0.0, 1.0 - norm(DX - A_l_app) ** 2 / (norm(DX) ** 2 + 1e-12))
            else:
                w = 1.0
            sw = np.sqrt(w)
            W_blocks.append(sw * W_l)
            U_blocks.append(sw * Ur)
            ranks.append(Ur.shape[1])

        Wcat = np.concatenate(W_blocks, axis=1) if W_blocks else np.zeros((G, 0))
        Ucat = np.concatenate(U_blocks, axis=1) if U_blocks else np.zeros((G, 0))
        sum_AP = be.matmul(Wcat, Ucat.T)
        sum_P = be.matmul(Ucat, Ucat.T)
        if eps_fuse and eps_fuse > 0:
            denom_inv = np.linalg.inv(sum_P + eps_fuse * np.eye(G))
        else:
            denom_inv = be.pinv(sum_P, rcond=1e-10)
        D_fused = be.matmul(sum_AP, denom_inv)
        fr = np.linalg.matrix_rank(D_fused, tol=1e-6)

        self._report(ranks, ridge, sigma_cap, max_rank,
                     explained_variance_threshold,
                     fr, G, sum_P, eps_fuse,
                     sigma_pre, sigma_post, fit_r2, comp_rhos)
        rho_val = self._report_propagator(D_fused, G, n_steps_hint)

        return (DiffPlusIOperator(D_fused),
                {"fused_rank": fr, "G": G, "panel": gene_panel, "rho": rho_val,
                 "D": D_fused, "ranks": np.array(ranks),
                 "fit_r2": np.array(fit_r2)})

    # -- diagnostics (unchanged prints) ----------------------------------------
    def _report(self, ranks, ridge, sigma_cap, max_rank,
                explained_variance_threshold,
                fr, G, sum_P, eps_fuse,
                sigma_pre, sigma_post, fit_r2, comp_rhos):
        try:
            eig = np.linalg.eigvalsh(sum_P)
            eig = eig[eig > 1e-8]
            if explained_variance_threshold is not None:
                rank_desc = f"adaptive (threshold={explained_variance_threshold})"
            elif max_rank is not None:
                rank_desc = f"max_rank={max_rank}"
            else:
                rank_desc = "max_rank=auto(~snap/2)"
            print(f"  fused over {len(ranks)} lineages (ridge={ridge:g}"
                  f"{', sigma_cap=%g' % sigma_cap if sigma_cap is not None else ''}"
                  f", {rank_desc}); "
                  f"per-lineage rank ~{np.median(ranks) if ranks else 0:.0f}; "
                  f"rank(D_fused)={fr} (G={G})")
            if ranks and (explained_variance_threshold is not None or len(set(ranks)) > 1):
                ra = np.array(ranks)
                qs = np.quantile(ra, [0.05, 0.25, 0.5, 0.75, 0.95])
                print(f"  per-lineage rank distribution: "
                      f"min={ra.min()} 5%={qs[0]:.0f} 25%={qs[1]:.0f} "
                      f"median={qs[2]:.0f} 75%={qs[3]:.0f} 95%={qs[4]:.0f} "
                      f"max={ra.max()} mean={ra.mean():.1f}")
                # ASCII histogram across unique ranks (or binned if too many)
                rmin, rmax = ra.min(), ra.max()
                if rmax - rmin + 1 <= 20:
                    counts = np.bincount(ra, minlength=rmax + 1)
                    total = counts.sum()
                    print(f"  rank histogram:")
                    bar_max = max(counts.max(), 1)
                    for r_val in range(rmin, rmax + 1):
                        c = counts[r_val]
                        bar = "#" * int(40 * c / bar_max)
                        print(f"    r={r_val:>3}: {c:>5} ({100*c/total:>5.1f}%) {bar}")
                else:
                    # bin into 20 bins
                    h, edges = np.histogram(ra, bins=20)
                    bar_max = max(h.max(), 1)
                    print(f"  rank histogram (20 bins):")
                    for i in range(len(h)):
                        bar = "#" * int(40 * h[i] / bar_max)
                        print(f"    [{edges[i]:>5.1f}, {edges[i+1]:>5.1f}): "
                              f"{h[i]:>5} {bar}")
            if sigma_cap is not None and sigma_pre:
                print(f"  per-lineage propagator sigma_max: "
                      f"pre-cap median={np.median(sigma_pre):.2f} max={np.max(sigma_pre):.2f} "
                      f"-> post-cap median={np.median(sigma_post):.2f} max={np.max(sigma_post):.2f}")
            if fit_r2:
                fr2 = np.array(fit_r2)
                print(f"  per-lineage DMD fit R^2: median={np.median(fr2):.3f} "
                      f"mean={fr2.mean():.3f} [10th={np.percentile(fr2,10):.3f}, "
                      f"90th={np.percentile(fr2,90):.3f}]")
            if comp_rhos:
                cr = np.array(comp_rhos)
                n_unstable = int(np.sum(cr > 1.0 + 1e-6))
                print(f"  per-COMPONENT propagator rho(I+A_l): "
                      f"median={np.median(cr):.3f} mean={cr.mean():.3f} "
                      f"max={cr.max():.3f} min={cr.min():.3f}; "
                      f"{n_unstable}/{len(cr)} unstable (rho>1)")
            print(f"  S=sum P_l spectrum: min={eig.min():.2f} max={eig.max():.2f} "
                  f"mean={eig.mean():.2f}"
                  f"{' | eps_fuse=%g' % eps_fuse if eps_fuse else ' | eps_fuse=0 (bare pinv)'}")
            near_int = np.mean(np.abs(eig - np.round(eig)) < 0.1)
            print(f"  S-eig within 0.1 of integer: {near_int:.2f}"
                  f"  ({'near-orthogonal' if near_int>0.7 else 'smeared'})")
        except Exception:
            print(f"  fused over {len(ranks)} lineages; rank(D_fused)={fr} (G={G})")

    def _report_propagator(self, D_fused, G, n_steps_hint):
        try:
            evM = np.linalg.eigvals(np.eye(G) + D_fused)
            mag = np.abs(evM)
            rho = mag.max()
            n_unstable = int(np.sum(mag > 1.0 + 1e-9))
            n_marg = int(np.sum(np.abs(mag - 1.0) <= 1e-9))
            magD = np.abs(np.linalg.eigvals(D_fused))
            print(f"  PROPAGATOR M=I+D spectrum: rho(M)={rho:.4f}  "
                  f"({'STABLE' if rho<=1+1e-6 else 'UNSTABLE -> powers explode'})")
            try:
                smax = float(np.linalg.norm(np.eye(G) + D_fused, 2))
                print(f"    sigma_max(M)={smax:.4f}  (non-normality gap "
                      f"sigma_max-rho = {smax - rho:.4f}; large => transient amplification)")
            except Exception:
                pass
            print(f"    |eig(M)|: max={mag.max():.3f} median={np.median(mag):.3f} "
                  f"min={mag.min():.3f};  #|eig|>1: {n_unstable}/{G}; #|eig|=1: {n_marg}")
            print(f"    over {n_steps_hint}-step horizon, rho^steps ~ {rho**n_steps_hint:.2e}")
            print(f"    |eig(D)| (the increment): max={magD.max():.3f} "
                  f"median={np.median(magD):.3f}")
            qs = np.quantile(mag, [0.5, 0.9, 0.99, 1.0])
            print(f"    |eig(M)| quantiles [50,90,99,100]%: "
                  f"{qs[0]:.3f} {qs[1]:.3f} {qs[2]:.3f} {qs[3]:.3f}")
            return float(rho)
        except Exception as e:
            print(f"  (propagator spectrum unavailable: {e})")
            return None
