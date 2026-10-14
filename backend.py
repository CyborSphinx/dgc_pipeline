"""
Linear-algebra backend: optionally offloads pinv/matmul to a CUDA GPU.

Replaces the original module-global `_TORCH`/`_DEV` pair and the free functions
`init_gpu`/`gpu_pinv`/`gpu_matmul` with a single `Backend` object. Numerically
identical: GPU path uses float32 (matching the original), CPU path uses numpy.

A process-wide default Backend is provided so call-sites that don't thread an
instance through still work; `Backend.activate()` sets it. The semantics of the
original code (refuse to silently fall back to CPU when --gpu is requested) are
preserved in `Backend.gpu(require=True)`.
"""
from __future__ import annotations

import sys

import numpy as np
from numpy.linalg import pinv


class Backend:
    """Holds the compute device and routes pinv/matmul through it."""

    _active: "Backend | None" = None

    def __init__(self, torch_mod=None, device: str | None = None):
        self._torch = torch_mod
        self._dev = device

    # -- construction -----------------------------------------------------------
    @classmethod
    def cpu(cls) -> "Backend":
        return cls(None, None)

    @classmethod
    def gpu(cls, require: bool = True) -> "Backend":
        """
        Build a GPU backend, smoke-testing an actual kernel launch. On failure
        with require=True the process exits (the original refuse-to-fallback
        behaviour); with require=False returns a CPU backend instead.
        """
        try:
            import torch
        except ImportError:
            if require:
                sys.exit("ERROR: --gpu given but torch not installed.\n"
                         "  pip install --pre torch --index-url "
                         "https://download.pytorch.org/whl/nightly/cu128")
            return cls.cpu()
        if not torch.cuda.is_available():
            if require:
                sys.exit("ERROR: --gpu given but torch.cuda.is_available() is False.\n"
                         "  On RTX 50-series you need a cu128+ NIGHTLY build.")
            return cls.cpu()
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        try:
            _ = (torch.randn(8, 8, device="cuda")
                 @ torch.randn(8, 8, device="cuda")).cpu()
        except Exception as e:
            if require:
                sys.exit(f"ERROR: GPU matmul failed on {name} (cap {cap}).\n"
                         f"  Blackwell sm_120 'no kernel image' problem; install "
                         f"cu128+ nightly. Underlying error: {e}")
            return cls.cpu()
        print(f"  GPU backend: {name} (compute {cap[0]}.{cap[1]}), kernel test OK")
        return cls(torch, "cuda")

    def activate(self) -> "Backend":
        """Make this the process-wide default backend; returns self."""
        Backend._active = self
        return self

    @classmethod
    def active(cls) -> "Backend":
        """The current default backend (CPU if none activated)."""
        if cls._active is None:
            cls._active = cls.cpu()
        return cls._active

    @property
    def on_gpu(self) -> bool:
        return self._torch is not None

    # -- operations -------------------------------------------------------------
    def pinv(self, M: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
        if self._torch is None:
            return pinv(M, rcond=rcond)
        t = self._torch.from_numpy(np.ascontiguousarray(M)).to(
            self._dev, dtype=self._torch.float32)
        out = self._torch.linalg.pinv(t, rtol=rcond).cpu().numpy()
        del t
        return out

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self._torch is None:
            return A @ B
        ta = self._torch.from_numpy(np.ascontiguousarray(A)).to(
            self._dev, dtype=self._torch.float32)
        tb = self._torch.from_numpy(np.ascontiguousarray(B)).to(
            self._dev, dtype=self._torch.float32)
        out = (ta @ tb).cpu().numpy()
        del ta, tb
        return out
