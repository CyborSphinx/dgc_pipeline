"""
Interleaved 24h-step operator fitting (M1/M2 fusion).

The 12h M powered k times to deconvolve a 7-day perturbation gives S_14 =
sum_{j=0}^{13} M^j. With sigma_max(M) ~ 23 and rho ~ 1.003, S_14 is
ill-conditioned (effective rank ~552/1479).

Mitigation: fit a 24h-step operator DIRECTLY from data, by skipping every other
timepoint. Two independent fits use disjoint subsets:
  M1 from transitions {0->1, 1->2, 2->3, ...} (24h jumps, even-indexed timepoints)
  M2 from transitions {0.5->1.5, 1.5->2.5, ...} (24h jumps, odd-indexed timepoints)
Average their D's. The resulting 24h operator gives a 7-step deconvolution
S_7 with much better conditioning (effective rank ~1413/1479 in the user's run).

Bonus sanity check: M1 and M2 estimate the SAME 24h operator under the model's
time-translation-invariance assumption. If they agree (cos(D1,D2) close to 1,
relative Frobenius difference small), that supports the constant-M model. If
they disagree, the dynamics are time-varying and the constant-M assumption is
questionable.
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import norm

from .fusion import FusedOperatorBuilder
from .operators import DiffPlusIOperator


class InterleavedFusion:
    """
    Build a 24h-step operator by fitting two interleaved 12h-stride fits and
    averaging their D matrices.
    """

    def __init__(self, builder: FusedOperatorBuilder | None = None, backend=None):
        if builder is None:
            from .backend import Backend
            builder = FusedOperatorBuilder(backend or Backend.active())
        self.builder = builder

    def build(self, traj_c, gene_panel=None, M_12_op=None, **kwargs):
        """
        Fit M1 on even-index timepoints, M2 on odd, fuse to a 24h operator.

        If M_12_op (DiffPlusIOperator for the 12h step) is passed, the primary
        diagnostic compares the fused 24h operator to M_12^2 -- this is the
        meaningful constant-dynamics check. M1 vs M2 alone can disagree for
        period-2 reasons (e.g. alternating cell-cycle phase, batch structure)
        that DO NOT mean the dynamics are time-varying at the 24h scale;
        comparing the FUSED 24h op to M_12^2 controls for that.

        kwargs passed through to FusedOperatorBuilder.build.
        Returns (M_24h_op, info).
        """
        if traj_c.shape[1] < 4:
            raise ValueError(f"need >= 4 timepoints; got K={traj_c.shape[1]}")

        traj_even = traj_c[:, 0::2, :]
        traj_odd  = traj_c[:, 1::2, :]
        if traj_even.shape[1] < 2 or traj_odd.shape[1] < 2:
            raise ValueError("interleaved subsets too short; need K >= 4")

        print(f"  fitting M1 on even timepoints  (K_even={traj_even.shape[1]})")
        op1, info1 = self.builder.build(traj_even, gene_panel=gene_panel, **kwargs)
        print(f"  fitting M2 on odd timepoints   (K_odd={traj_odd.shape[1]})")
        op2, info2 = self.builder.build(traj_odd, gene_panel=gene_panel, **kwargs)

        D1, D2 = op1.D, op2.D
        D_24h = 0.5 * (D1 + D2)
        op_24h = DiffPlusIOperator(D_24h)

        # M1 vs M2 (informative but NOT the primary check)
        cos_12 = float(np.dot(D1.flatten(), D2.flatten())
                       / (norm(D1) * norm(D2) + 1e-12))
        rel_12 = float(norm(D1 - D2) / (norm(D1) + 1e-12))

        info = dict(
            D=D_24h, op=op_24h,
            cos_M1_M2=cos_12, rel_frob_M1_M2=rel_12,
            rho_M1=info1.get("rho"), rho_M2=info2.get("rho"),
            info_M1=info1, info_M2=info2,
        )

        print(f"\n  M1/M2 split (informative; period-2 effects can split these "
              f"benignly):")
        print(f"    cos(D1, D2) = {cos_12:+.4f}   rel ||D1-D2||/||D1|| = {rel_12:.3f}")
        if info["rho_M1"] is not None and info["rho_M2"] is not None:
            print(f"    rho(M1) = {info['rho_M1']:.4f}    rho(M2) = {info['rho_M2']:.4f}")

        # PRIMARY check: fused 24h vs M_12 squared
        if M_12_op is not None:
            G = D_24h.shape[0]
            M12 = M_12_op.as_matrix(G)
            M24_fused = np.eye(G) + D_24h
            M12_sq = M12 @ M12
            # measure on the INCREMENT (D), not M, to ignore the identity bias
            D12_sq = M12_sq - np.eye(G)
            cos_primary = float(np.dot(D_24h.flatten(), D12_sq.flatten())
                                / (norm(D_24h) * norm(D12_sq) + 1e-12))
            rel_primary = float(norm(D_24h - D12_sq) / (norm(D12_sq) + 1e-12))
            info["cos_fused_vs_M12sq"] = cos_primary
            info["rel_frob_fused_vs_M12sq"] = rel_primary
            print(f"\n  PRIMARY CHECK: fused 24h operator vs (M_12)^2")
            print(f"    cos(D_24_fused, M_12^2 - I) = {cos_primary:+.4f}   "
                  f"(1.0 = perfectly consistent)")
            print(f"    rel ||D_24_fused - (M_12^2-I)|| / ||M_12^2-I|| = "
                  f"{rel_primary:.3f}")
            if cos_primary > 0.9:
                print(f"    OK: the 24h fit is consistent with squaring the 12h fit. "
                      f"Deconvolution at 24h step is supported.")
            elif cos_primary > 0.7:
                print(f"    PARTIAL: the 24h fit agrees with M_12^2 in direction but "
                      f"differs in magnitude. Deconvolution is usable with some bias.")
            else:
                print(f"    WARNING: the 24h fit disagrees with M_12^2. Either the "
                      f"12h fit is wrong, the 24h fit is wrong, or the underlying "
                      f"dynamics aren't time-translation invariant.")

        return op_24h, info


def make_response_sum(D, n_steps):
    """
    S_k = I + M + M^2 + ... + M^{k-1}, where M = I + D.
    Returns S_k as a dense (G, G) matrix.
    """
    G = D.shape[0]
    M = np.eye(G) + D
    S = np.eye(G)
    M_pow = np.eye(G)
    for _ in range(1, n_steps):
        M_pow = M_pow @ M
        S = S + M_pow
    return S


def tikhonov_deconvolve(S, Delta_X, lam=None, lam_frac=1e-2):
    """
    Solve B in S B = Delta_X (Delta_X is G x m, columns = per-TF observed
    response). Uses ridge regression: B = (S^T S + lam I)^{-1} S^T Delta_X.

    lam: explicit Tikhonov parameter; if None, set as lam_frac * sigma_max(S)^2.
    """
    SVt = S.T
    StS = SVt @ S
    if lam is None:
        # scale Tikhonov to the operator. lam_frac * (largest singular value)^2
        sm = float(np.linalg.norm(S, 2))
        lam = lam_frac * (sm ** 2)
    G = S.shape[1]
    return np.linalg.solve(StS + lam * np.eye(G), SVt @ Delta_X), lam
