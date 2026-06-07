"""
iPSC target-state construction.

The state TFs should drive cells toward. Two definitions share all of their
cell-selection logic (which the original duplicated across two functions), so
they live as methods on one builder that resolves the iPSC cell set once:

  TargetBuilder(A, genes, cell_sets_path, ...)
    .centroid()      mean expression of all (final-day) iPSC-annotated cells
    .biological()    purity-gated: mean of the iPSC cells that actually express
                     the endogenous pluripotency program highly (a cleaner state,
                     not smeared by partially-reprogrammed cells)

Both return a full-gene-space (G,) vector or None if no iPSC set is found.
Behaviour is identical to the original ipsc_centroid_target /
ipsc_biological_target; only the shared selection is factored out.
"""
from __future__ import annotations

import os

import numpy as np

from .io import load_gmt, ENDOGENOUS_PLURIPOTENCY

_IPSC_SET_KEYS = ["iPSC", "IPS", "iPS", "Pluripotent", "Stem",
                  "ESC", "ES-like", "Epiblast"]


class TargetBuilder:
    def __init__(self, A, genes, cell_sets_path, ipsc_key_substr="iPSC",
                 cell_days=None, final_day=None):
        self.A = A
        self.genes = list(genes)
        self.cell_sets_path = cell_sets_path
        self.ipsc_key_substr = ipsc_key_substr
        self.cell_days = cell_days
        self.final_day = final_day
        self._rows = None          # cached indices of selected iPSC cells
        self._X = None             # cached dense expression of those cells

    # -- shared cell selection --------------------------------------------------
    def _resolve_rows(self):
        if self._rows is not None:
            return self._rows
        if self.cell_sets_path is None or not os.path.exists(self.cell_sets_path):
            self._rows = np.array([], int)
            return self._rows
        sets = load_gmt(self.cell_sets_path)
        keys = [self.ipsc_key_substr] + _IPSC_SET_KEYS
        ipsc_ids = set()
        for name, members in sets.items():
            if any(k.lower() in name.lower() for k in keys):
                ipsc_ids.update(members)
        if not ipsc_ids:
            self._rows = np.array([], int)
            return self._rows
        obs = np.asarray(self.A.obs_names)
        # Strip trailing suffixes before checking
        mask = np.array([c.split('_')[0] in ipsc_ids for c in obs])
        if self.cell_days is not None and self.final_day is not None:
            day_of = {c: self.cell_days.get(c, np.nan) for c in obs}
            dvals = np.array([day_of[c] for c in obs])
            mask = mask & (dvals >= self.final_day - 1e-6)
        self._rows = np.where(mask)[0]
        return self._rows

    def _expr(self):
        if self._X is not None:
            return self._X
        rows = self._resolve_rows()
        if rows.size == 0:
            self._X = None
            return None
        X = self.A.X[rows, :]
        self._X = np.asarray(X.toarray() if hasattr(X, "toarray") else X)
        return self._X

    @property
    def n_ipsc(self) -> int:
        return int(self._resolve_rows().size)

    # -- the two target definitions --------------------------------------------
    def centroid(self):
        X = self._expr()
        if X is None:
            return None
        print(f"  iPSC-centroid target from {self.n_ipsc} iPSC cells")
        return X.mean(axis=0).ravel()

    def biological(self, marker_genes=None, purity_quantile=0.75):
        X = self._expr()
        if X is None:
            n_ipsc = self.n_ipsc
            if n_ipsc == 0:
                print("  iPSC-biological target: returning None (no iPSC cells "
                      f"resolved from cell-sets file '{self.cell_sets_path}' under "
                      f"key '~{self.ipsc_key_substr}'; check the file and final-day filter)")
            else:
                print(f"  iPSC-biological target: returning None ({n_ipsc} iPSC cells "
                      "found but expression slice empty)")
            return None
        gidx = {g: i for i, g in enumerate(self.genes)}
        markers = marker_genes if marker_genes is not None else ENDOGENOUS_PLURIPOTENCY
        midx = [gidx[g] for g in markers if g in gidx]
        if len(midx) < 2:
            print(f"  iPSC-biological target: <2 markers in panel; "
                  f"falling back to full centroid ({self.n_ipsc} cells)")
            return X.mean(axis=0).ravel()
        Xm = X[:, midx]
        mu = Xm.mean(0, keepdims=True)
        sd = Xm.std(0, keepdims=True) + 1e-8
        purity = ((Xm - mu) / sd).mean(axis=1)
        keep = purity >= np.quantile(purity, purity_quantile)
        if keep.sum() < 5:
            keep = purity >= np.quantile(purity, 0.5)
        present = [g for g in markers if g in gidx]
        print(f"  iPSC-biological target: {int(keep.sum())}/{self.n_ipsc} "
              f"final-day iPSC cells passing purity gate "
              f"(top {100*(1-purity_quantile):.0f}% on {len(midx)} endogenous "
              f"markers: {present[:6]}...)")
        return X[keep].mean(axis=0).ravel()
