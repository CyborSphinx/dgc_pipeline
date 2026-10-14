"""
Fused operator builder: dimension-aware fusion over lineages.

For each (paired or single) trajectory l, fit DX_l = A_l X_l (low-rank DMD
on time differences); let P_l project onto row(A_l). Then

    D_fused = ( sum_l A_l P_l ) ( sum_l P_l )^+

computed via skinny factors. Each trajectory contributes equal effective mass.

Lineage pairing (off by default): given pairs (I[p], J[p]),
    y^p_k = traj[I[p], k] - traj[J[p], k]
and the fit is on y^p instead of traj[l]. By the model
x_{k+1} = M x_k + b_k with b_k shared across lineages, the lineage
difference CANCELS b_k.
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import norm, svd


class FusedOperatorBuilder:
    """Fit a fused operator D over lineages and return it."""

    def build(self, traj_c, rank_tol=1e-8, gene_panel=None, ridge=0.0,
              max_rank=None, explained_variance_threshold=None,
              eps_fuse=0.0,
              pair_lineages=False, pair_indices=None, n_pairs=None,
              pair_dropout=0.0, pair_seed=0):
        """
        Fit the fused operator D and return it.

        traj_c : (L, K, G) centered trajectories.
        Rank control (mutually exclusive):
            max_rank=K  OR  explained_variance_threshold=t in (0,1].
        """
        if gene_panel is not None:
            traj_c = traj_c[:, :, gene_panel]
        L_orig, K, G = traj_c.shape

        # ---- lineage pairing (cancels shared b_k) ---------------------
        if pair_lineages:
            if pair_indices is None:
                rng = np.random.default_rng(pair_seed)
                budget = n_pairs if n_pairs is not None else L_orig
                I = rng.integers(0, L_orig, size=budget)
                J = rng.integers(0, L_orig, size=budget)
                same = (I == J)
                if same.any():
                    J[same] = (J[same] + 1) % L_orig
            else:
                I, J = (np.asarray(pair_indices[0]),
                        np.asarray(pair_indices[1]))
            traj_c = traj_c[I] - traj_c[J]
            print(f"  lineage pairing: {len(I)} difference series "
                  f"from L={L_orig} lineages")

        if pair_dropout > 0:
            rng = np.random.default_rng(pair_seed + 1)
            keep = rng.random(traj_c.shape[0]) >= pair_dropout
            n_kept = int(keep.sum())
            if n_kept < 2:
                print(f"  [warn] pair_dropout={pair_dropout} left {n_kept} "
                      f"items; using all instead")
            else:
                traj_c = traj_c[keep]
                print(f"  pair dropout: keeping {n_kept}/{len(keep)} "
                      f"({100 * n_kept / len(keep):.0f}%) at seed {pair_seed}")

        L = traj_c.shape[0]

        # ---- fit each trajectory --------------------------------------
        W_blocks, U_blocks, ranks, fit_r2 = [], [], [], []

        for l in range(L):
            Xl = traj_c[l]                                          # (K, G)
            Xsnap = Xl[:-1].T                                       # (G, K-1)
            DX = (Xl[1:] - Xl[:-1]).T                               # (G, K-1)

            U, S, Vt = svd(Xsnap, full_matrices=False)
            r_avail = int(np.sum(S > rank_tol * (S[0] if S.size else 1.0)))
            n_snap = Xsnap.shape[1]

            if explained_variance_threshold is not None:
                if r_avail == 0:
                    r = 0
                else:
                    cum = (np.cumsum(S[:r_avail] ** 2)
                           / (np.sum(S[:r_avail] ** 2) + 1e-12))
                    r = int(np.searchsorted(cum, explained_variance_threshold) + 1)
                    r = min(r, r_avail)
            else:
                cap = max_rank if max_rank is not None else max(2, n_snap // 2)
                r = min(r_avail, cap)
            if r == 0:
                continue

            Ur, Sr, Vtr = U[:, :r], S[:r], Vt[:r, :]
            filt = Sr / (Sr ** 2 + ridge)
            W_l = DX @ Vtr.T @ np.diag(filt)

            DX_pred = W_l @ (Ur.T @ Xsnap)
            fit_r2.append(1.0 - norm(DX - DX_pred) ** 2
                          / (norm(DX) ** 2 + 1e-12))

            W_blocks.append(W_l)
            U_blocks.append(Ur)
            ranks.append(r)

        # ---- fuse -----------------------------------------------------
        Wcat = np.concatenate(W_blocks, axis=1) if W_blocks else np.zeros((G, 0))
        Ucat = np.concatenate(U_blocks, axis=1) if U_blocks else np.zeros((G, 0))
        sum_AP = Wcat @ Ucat.T
        sum_P = Ucat @ Ucat.T
        if eps_fuse and eps_fuse > 0:
            denom_inv = np.linalg.inv(sum_P + eps_fuse * np.eye(G))
        else:
            denom_inv = np.linalg.pinv(sum_P, rcond=1e-10)
        D_fused = sum_AP @ denom_inv

        # ---- statistics ----------------------------------------------
        fr = np.linalg.matrix_rank(D_fused, tol=1e-6)
        ra = np.array(ranks, dtype=int) if ranks else np.array([0])
        fr2 = np.array(fit_r2) if fit_r2 else np.array([0.0])

        print(f"  fused over {L} trajectories; "
              f"rank(D_fused)={fr} (G={G})")

        # rank histogram
        rmin, rmax = int(ra.min()), int(ra.max())
        counts = np.bincount(ra, minlength=rmax + 1)
        total = counts.sum()
        bar_max = max(counts.max(), 1)
        print(f"  per-trajectory rank histogram:")
        for r_val in range(rmin, rmax + 1):
            c = counts[r_val]
            bar = "#" * int(40 * c / bar_max)
            print(f"    r={r_val:>3}: {c:>5} "
                  f"({100 * c / total:>5.1f}%) {bar}")

        print(f"  per-trajectory fit R^2: median={np.median(fr2):.3f} "
              f"mean={fr2.mean():.3f} "
              f"[10th={np.percentile(fr2, 10):.3f}, "
              f"90th={np.percentile(fr2, 90):.3f}]")

        return D_fused