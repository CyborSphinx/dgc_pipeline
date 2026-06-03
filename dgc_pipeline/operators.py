"""
Linear dynamics operators and the sequence abstraction the scorers run on.

The original code represented operators as tuples dispatched by a string tag in
`apply_op` ("rank1" / "diff_plus_I" / "dense") and re-implemented the per-step
control-column propagation in four different functions. Both are unified here:

  Operator (ABC)
    .apply(vec)               -> M @ vec               (one step)
    .as_matrix(G)             -> dense (I+A) for spectra/powers
  Rank1Operator      DGC A_k = I + u vᵀ   (identity built in)
  DiffPlusIOperator  fused difference op, applied as (I + D)
  DenseOperator      raw matrix, applied as-is

  OperatorSequence
    wraps a list of Operators with the "use ops[k], clamp to last" rule and owns
    the propagation maths every scorer needs:
      .free_evolution(x, K)         -> M_{K-1}…M_0 x
      .propagate(V, k0, K)          -> apply ops[k0..K-1] to columns of V
      .control_column(b, k, K)      -> (prod_{j>k} M_j) b   (inject b at step k)
      .control_columns_timed(b, K)  -> (G,K) all per-step columns
      .control_vector(b, K)         -> sum_k control_column (inject-and-hold)

Convention (verified to machine precision against the original): injecting at
step k and propagating to endpoint K contributes M^{K-1-k} b, i.e. ops j=k+1..K-1
are applied. free_evolution over K steps applies ops 0..K-1.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Operator(ABC):
    """One linear map x -> M x. Subclasses fix what M is and how to apply it."""

    @abstractmethod
    def apply(self, vec: np.ndarray) -> np.ndarray:
        """Return M @ vec for a single (G,) vector."""

    def apply_cols(self, V: np.ndarray) -> np.ndarray:
        """Apply M to every column of a (G, n) matrix. Default loops apply()."""
        if V.shape[1] == 0:
            return V
        return np.column_stack([self.apply(V[:, c]) for c in range(V.shape[1])])

    @abstractmethod
    def as_matrix(self, G: int) -> np.ndarray:
        """Dense (G,G) form of M = I + A, for spectra / matrix powers."""

    @classmethod
    def from_legacy(cls, op_tuple) -> "Operator":
        """Build from the original ('rank1'|'diff_plus_I'|'dense', ...) tuple."""
        tag = op_tuple[0]
        if tag == "rank1":
            _, u, v = op_tuple
            return Rank1Operator(np.asarray(u), np.asarray(v))
        if tag == "diff_plus_I":
            return DiffPlusIOperator(np.asarray(op_tuple[1]))
        if tag == "dense":
            return DenseOperator(np.asarray(op_tuple[1]))
        raise ValueError(f"unknown legacy op tag: {tag!r}")


class Rank1Operator(Operator):
    """DGC operator A_k = I + u vᵀ, applied as x + u (v·x). Identity is built in."""

    def __init__(self, u: np.ndarray, v: np.ndarray):
        self.u = u
        self.v = v

    def apply(self, vec: np.ndarray) -> np.ndarray:
        return vec + self.u * (self.v @ vec)

    def as_matrix(self, G: int) -> np.ndarray:
        return np.eye(G) + np.outer(self.u, self.v)


class DiffPlusIOperator(Operator):
    """Fused difference operator D, applied as (I + D): x -> x + D x."""

    def __init__(self, D: np.ndarray):
        self.D = D

    def apply(self, vec: np.ndarray) -> np.ndarray:
        return vec + self.D @ vec

    def apply_cols(self, V: np.ndarray) -> np.ndarray:
        if V.shape[1] == 0:
            return V
        return V + self.D @ V                      # vectorised; matches per-col apply

    def as_matrix(self, G: int) -> np.ndarray:
        return np.eye(G) + self.D


class DenseOperator(Operator):
    """Raw operator M, applied as-is: x -> M x."""

    def __init__(self, M: np.ndarray):
        self.M = M

    def apply(self, vec: np.ndarray) -> np.ndarray:
        return self.M @ vec

    def apply_cols(self, V: np.ndarray) -> np.ndarray:
        return self.M @ V if V.shape[1] else V

    def as_matrix(self, G: int) -> np.ndarray:
        return self.M


class OperatorSequence:
    """
    A time-indexed list of Operators with the original "clamp to last" rule
    (ops[k] for k < len, else ops[-1]) and all the propagation maths the scorers
    share. Construct from Operators or, for drop-in compatibility, from the
    legacy list of tuples via `from_legacy`.
    """

    def __init__(self, ops):
        self.ops = list(ops)
        if not self.ops:
            raise ValueError("OperatorSequence needs at least one operator")

    @classmethod
    def from_legacy(cls, ops_tuples) -> "OperatorSequence":
        return cls([Operator.from_legacy(o) for o in ops_tuples])

    def __len__(self):
        return len(self.ops)

    def at(self, k: int) -> Operator:
        """Operator for step k, clamped to the last available one."""
        return self.ops[k] if k < len(self.ops) else self.ops[-1]

    # -- propagation primitives -------------------------------------------------
    def free_evolution(self, x_init: np.ndarray, n_steps: int) -> np.ndarray:
        """Apply ops 0..n_steps-1 to x_init (the uncontrolled state at step n)."""
        z = x_init.copy()
        for k in range(n_steps):
            z = self.at(k).apply(z)
        return z

    def control_column(self, b_m: np.ndarray, k: int, K: int) -> np.ndarray:
        """Endpoint effect at step K of injecting b_m at step k: (prod_{j>k} M_j) b."""
        c = b_m.copy()
        for j in range(k + 1, K):
            c = self.at(j).apply(c)
        return c

    def control_columns_timed(self, b_m: np.ndarray, n_steps: int) -> np.ndarray:
        """(G, n_steps) matrix; column k = inject b_m at step k, propagate to n_steps."""
        G = len(b_m)
        cols = np.zeros((G, n_steps))
        for k in range(n_steps):
            cols[:, k] = self.control_column(b_m, k, n_steps)
        return cols

    def control_vector(self, b_m: np.ndarray, n_steps: int) -> np.ndarray:
        """Inject-and-hold control: sum_k (prod_{j>k} M_j) b_m."""
        G = len(b_m)
        C = np.zeros(G)
        for k in range(n_steps):
            C += self.control_column(b_m, k, n_steps)
        return C
