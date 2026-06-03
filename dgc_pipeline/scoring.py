"""
TF scoring against an iPSC target, sharing one OperatorSequence abstraction.

Scorer hierarchy
----------------
Scorer (ABC)            common interface: .score(...) -> {tf_index: score}
  BOnlyScorer           static B, single-vector non-negative projection (no time)
  DynamicsScorer        per-TF timed NNLS over that TF's own control schedule
  ControlBestDayScorer  JOINT NNLS over all TFs, swept over endpoint K (best day);
                        importance via coefficient mass + leave-one-out
  ComboScorer           joint timed NNLS over a fixed TF subset (cocktail)

All scorers operate in the centered+panel frame; B columns are deltas, so adding
raw-frame binding directions to centered states is consistent (the constant
expression baseline cancels in every state difference the scorers form).

The numerics are unchanged from the original score_* functions; only the
structure (shared sequence object, class interface) differs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.linalg import norm
from scipy.optimize import nnls

from .operators import OperatorSequence


def _scorable(B, tf_filter):
    """TF column indices in tf_filter whose B column is non-zero."""
    return [j for j in tf_filter if norm(B[:, j]) > 1e-12]


class Scorer(ABC):
    @abstractmethod
    def score(self, B, tf_filter, x_init, x_target):
        """Return {tf_index: score} where higher = better driver toward target."""


class BOnlyScorer(Scorer):
    """
    Static B, no dynamics: best non-negative scalar of column b toward
    r = x_target - x_init. score = ||r|| - ||r - u b||, u = max(0, r·b/||b||²).
    Collapses to 0 for any column with r·b <= 0 (correct, but degenerate when it
    happens for every column -- read such a ranking as tie-order, not signal).
    """

    def score(self, B, tf_filter, x_init, x_target):
        r = x_target - x_init
        d0 = norm(r)
        out = {}
        for j in tf_filter:
            b = B[:, j]
            if norm(b) < 1e-12:
                continue
            u = max(0.0, (r @ b) / (b @ b + 1e-12))
            out[j] = d0 - norm(r - u * b)
        return out


class DynamicsScorer(Scorer):
    """
    Per-TF dynamics-propagated score. r = x_target - free_evolution(x_init).
    timed=True scores TF j by the best NON-NEGATIVE schedule over its own per-step
    control columns (joint NNLS within that one TF). timed=False is the legacy
    single summed-vector form kept only for comparison.
    """

    def __init__(self, seq: OperatorSequence, n_steps: int, timed: bool = True):
        self.seq = seq
        self.n_steps = n_steps
        self.timed = timed
        self.cos = {}                      # best-step alignment diagnostic per TF

    def score(self, B, tf_filter, x_init, x_target):
        z = self.seq.free_evolution(x_init, self.n_steps)
        r = x_target - z
        d0 = norm(r)
        out = {}
        self.cos = {}
        for j in tf_filter:
            b = B[:, j]
            if norm(b) < 1e-12:
                continue
            if self.timed:
                C = self.seq.control_columns_timed(b, self.n_steps)
                keep = [k for k in range(C.shape[1]) if norm(C[:, k]) >= 1e-12]
                if not keep:
                    out[j] = 0.0
                    self.cos[j] = 0.0
                    continue
                M_ = C[:, keep]
                u, _ = nnls(M_, r)
                out[j] = d0 - norm(r - M_ @ u)
                self.cos[j] = max((r @ C[:, k]) / (norm(r) * norm(C[:, k]) + 1e-12)
                                  for k in keep)
            else:
                C = self.seq.control_vector(b, self.n_steps)
                if norm(C) < 1e-12:
                    out[j] = 0.0
                    continue
                u = max(0.0, (r @ C) / (C @ C + 1e-12))
                out[j] = d0 - norm(r - u * C)
                ref = self.n_steps * b
                self.cos[j] = (C @ ref) / (norm(C) * norm(ref) + 1e-12)
        return out


class ControlBestDayScorer:
    """
    The actual control problem, solved jointly and swept over the endpoint K.

    For each K, gap r_K = x_target - M^K x_init must be closed by a non-negative
    schedule over the controllability matrix C_K = [M^{K-1}B | ... | M^0 B]. We
    solve min_{u>=0} ||r_K - C_K u|| (column-normalised for conditioning; optional
    Tikhonov eps), pick the K with smallest residual ("best day"), and read per-TF
    importance off that solution: coefficient mass and leave-one-out.

    Built incrementally: growing K propagates existing columns one step and
    appends a fresh B block, so the whole sweep is O(n_steps) propagations.
    """

    def __init__(self, seq: OperatorSequence, n_steps: int,
                 eps: float = 0.0, do_loo: bool = True):
        self.seq = seq
        self.n_steps = n_steps
        self.eps = eps
        self.do_loo = do_loo

    def run(self, B, tf_filter, x_init, x_target):
        cols_idx = _scorable(B, tf_filter)
        if not cols_idx:
            return None
        m = len(cols_idx)
        Bsub = B[:, cols_idx]
        cols = np.zeros((len(x_init), 0))
        col_tf = np.zeros((0,), int)
        xfree = x_init.copy()
        curve, best = [], None
        for K in range(1, self.n_steps + 1):
            op_prev = self.seq.at(K - 1)
            if cols.shape[1] > 0:
                cols = op_prev.apply_cols(cols)
            xfree = op_prev.apply(xfree)
            cols = np.hstack([cols, Bsub])
            col_tf = np.concatenate([col_tf, np.arange(m)])
            rK = x_target - xfree
            gap = float(norm(rK))
            cn = norm(cols, axis=0) + 1e-12
            Cs = cols / cn
            if self.eps > 0:
                A_aug = np.vstack([Cs, self.eps * np.eye(Cs.shape[1])])
                b_aug = np.concatenate([rK, np.zeros(Cs.shape[1])])
                u_s, _ = nnls(A_aug, b_aug)
            else:
                u_s, _ = nnls(Cs, rK)
            resid = float(norm(rK - Cs @ u_s))
            curve.append((K, gap, resid, resid / (gap + 1e-12)))
            if best is None or resid < best["resid"]:
                best = dict(K=K, resid=resid, gap=gap, u=(u_s / cn).copy(),
                            col_tf=col_tf.copy(), Cs=Cs.copy(), rK=rK.copy())

        mass_local = np.zeros(m)
        for c, t in enumerate(best["col_tf"]):
            mass_local[t] += best["u"][c]
        loo_local = np.zeros(m)
        if self.do_loo:
            base, Cs, rK, ctf = best["resid"], best["Cs"], best["rK"], best["col_tf"]
            for t in range(m):
                keep = ctf != t
                if keep.sum() == 0:
                    loo_local[t] = norm(rK) - base
                    continue
                u2, _ = nnls(Cs[:, keep], rK)
                loo_local[t] = float(norm(rK - Cs[:, keep] @ u2) - base)

        return dict(best_K=best["K"], curve=curve,
                    best_resid=best["resid"], best_gap=best["gap"], n_tf=m,
                    mass={cols_idx[t]: mass_local[t] for t in range(m)},
                    loo={cols_idx[t]: loo_local[t] for t in range(m)})


# -- ranking utilities ---------------------------------------------------------
def ranks_of_known(scores, tfs_present, known):
    """Rank (1=best) of each known TF among all scored TFs; descending score."""
    order = sorted(scores, key=lambda j: -scores[j])
    pos = {j: r for r, j in enumerate(order, 1)}
    name_of = {j: t for t, j in ((t, tfs_present.index(t))
                                 for t in tfs_present if t in tfs_present)}
    out = {}
    for t in known:
        if t in tfs_present:
            j = tfs_present.index(t)
            out[t] = pos.get(j)
    return out, order


def precision_at_k(order, known_indices, k):
    """Fraction of the top-k scored TFs that are known drivers."""
    if k == 0:
        return 0.0
    top = order[:k]
    return sum(1 for j in top if j in known_indices) / k
