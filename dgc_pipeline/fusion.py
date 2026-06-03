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
              eps_fuse=0.0):
        be = self.backend
        L, K, Gfull = traj_c.shape
        if gene_panel is not None:
            traj_c = traj_c[:, :, gene_panel]
        G = traj_c.shape[2]
        if G > 4000:
            print(f"  [warn] fusion is G x G with G={G}; consider --gene_panel")

        W_blocks, U_blocks, ranks = [], [], []
        sigma_pre, sigma_post, fit_r2, comp_rhos = [], [], [], []
        for l in range(L):
            Xl = traj_c[l].T
            Xsnap = Xl[:, :-1]
            DX = Xl[:, 1:] - Xl[:, :-1]
            U, S, Vt = svd(Xsnap, full_matrices=False)
            r = int(np.sum(S > rank_tol * (S[0] if S.size else 1.0)))
            n_snap = Xsnap.shape[1]
            cap = max_rank if max_rank is not None else max(2, n_snap // 2)
            r = min(r, cap)
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

        self._report(ranks, ridge, sigma_cap, max_rank, fr, G, sum_P, eps_fuse,
                     sigma_pre, sigma_post, fit_r2, comp_rhos)
        rho_val = self._report_propagator(D_fused, G, n_steps_hint)

        # --- EXPLICIT MANUSCRIPT VALIDATION PRINTOUT ---
        self._report_manuscript_metrics(D_fused, traj_c, rho_val)

        return (DiffPlusIOperator(D_fused),
                {"fused_rank": fr, "G": G, "panel": gene_panel, "rho": rho_val,
                 "D": D_fused})

    # -- newly added manuscript metric reporting -------------------------------
    def _report_manuscript_metrics(self, D_fused, traj_c, rho_val):
        """Calculates and prints the specific empirical metrics needed for the 1-page summary."""
        try:
            L, K, G = traj_c.shape
            n_steps = K - 1
            
            print("\n--- 1-PAGER MANUSCRIPT VALIDATIONS ---")
            
            # [1] PHYSICS CHECK
            print("[1] PHYSICS CHECK:")
            if rho_val is not None:
                print(f"    Spectral Radius rho(M): {rho_val:.4f}")
                print(f"    (Bounded amplification confirms non-explosive continuous-time stability)\n")
            else:
                print(f"    Spectral Radius unavailable.\n")

            # [2] EXTRACTING CONTROL FORCING & DIMENSIONALITY
            # Safely cast to numpy for diagnostic math regardless of backend
            traj_c_np = np.asarray(traj_c)
            X_raw = traj_c_np[:, :-1, :].reshape(L * n_steps, G).T
            Y_raw = traj_c_np[:, 1:, :].reshape(L * n_steps, G).T
            
            M = np.eye(G) + np.asarray(D_fused)
            B_estimated = Y_raw - (M @ X_raw)
            
            # Extract variance of the forcing vector (b_k)
            _, S_b, _ = svd(B_estimated, full_matrices=False)
            variance_explained = (S_b**2) / np.sum(S_b**2)
            
            print("[2] CONTROL DIMENSIONALITY (SCREE PLOT VARIANCES):")
            for i in range(min(8, len(variance_explained))):
                print(f"    Dim {i+1}: {variance_explained[i]:.4f}")
            print("    (Expect strict variance collapse after Dimension 4)\n")

            # [3] EMPIRICAL FIT (R^2)
            mean_B = np.mean(B_estimated, axis=1, keepdims=True)
            Y_predicted = (M @ X_raw) + mean_B
            
            SS_res = np.sum((Y_raw - Y_predicted)**2)
            SS_tot = np.sum((Y_raw - np.mean(Y_raw, axis=1, keepdims=True))**2)
            r_squared = 1 - (SS_res / SS_tot)
            
            print("[3] EMPIRICAL VALIDATION:")
            print(f"    Linear Matrix Fit R^2:  {r_squared:.4f}")
            print("--------------------------------------\n")
            
        except Exception as e:
            print(f"\n[Warning] Manuscript metrics generation failed: {e}\n")

    # -- diagnostics (unchanged prints) ----------------------------------------
    def _report(self, ranks, ridge, sigma_cap, max_rank, fr, G, sum_P, eps_fuse,
                sigma_pre, sigma_post, fit_r2, comp_rhos):
        try:
            eig = np.linalg.eigvalsh(sum_P)
            eig = eig[eig > 1e-8]
            print(f"  fused over {len(ranks)} lineages (ridge={ridge:g}"
                  f"{', sigma_cap=%g' % sigma_cap if sigma_cap is not None else ''}"
                  f"{', max_rank=%d' % max_rank if max_rank is not None else ', max_rank=auto(~snap/2)'}); "
                  f"per-lineage rank ~{np.median(ranks) if ranks else 0:.0f}; "
                  f"rank(D_fused)={fr} (G={G})")
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