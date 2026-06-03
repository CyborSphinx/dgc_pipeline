"""
Data loading: expression matrix, transport maps, GMT cell sets, generic tables.

These are thin, stateless readers lifted verbatim (behaviour-preserving) from the
original module-level functions. Grouped here so the rest of the package never
touches file formats directly.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

try:
    import anndata as ad
except ImportError:
    ad = None


# WOT fate-predictive / validated reprogramming TFs (mouse casing).
# OSKM are the experimental input; the rest are pluripotency/fate-predictive TFs.
KNOWN_IPSC_TFS = [
    "Pou5f1", "Sox2", "Klf4", "Myc",          # OSKM (experimental input)
    "Nanog", "Esrrb", "Zfp42", "Obox6",       # fate-predictive pluripotency
    "Sall4", "Lin28a", "Nr5a2", "Prdm14",
    "Tfcp2l1", "Utf1", "Dppa2", "Dppa4", "Dppa5a",
]

# endogenous pluripotency markers (NOT exogenous OSKM) for the biological target
ENDOGENOUS_PLURIPOTENCY = [
    "Nanog", "Esrrb", "Zfp42", "Obox6", "Sall4", "Prdm14",
    "Dppa2", "Dppa4", "Dppa5a", "Utf1", "Tfcp2l1",
]


def read_table_auto(path: str) -> pd.DataFrame:
    """Read a whitespace/comma/tab table, auto-sniffing the separator."""
    return pd.read_csv(path, sep=None, engine="python")


def load_expression(h5ad_path: str):
    """Load an AnnData .h5ad expression object (cells x genes)."""
    if ad is None:
        raise ImportError("anndata not installed; needed to load expression")
    A = ad.read_h5ad(h5ad_path)
    print(f"  {A.shape[0]} cells x {A.shape[1]} genes")
    return A


def load_cell_days(path: str) -> dict:
    """cell_id -> day (float). Two-column table (id, day)."""
    df = read_table_auto(path)
    c0, c1 = df.columns[:2]
    return dict(zip(df[c0].astype(str), df[c1].astype(float)))


def load_id_list(path: str) -> list:
    """One id per line."""
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def load_gmt(path: str) -> dict:
    """GMT cell-set file -> {set_name: [member ids]}."""
    sets = {}
    with open(path) as fh:
        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, _desc, *members = parts
            sets[name] = [m for m in members if m]
    return sets


def discover_tmaps(tmap_dir: str) -> list:
    """
    Find transport-map files and order them by their day-pair. Returns a list of
    (day0, day1, path) sorted by day0. Files named like '*_<d0>_<d1>.*'.
    """
    paths = sorted(glob.glob(os.path.join(tmap_dir, "*")))
    chain = []
    for p in paths:
        base = os.path.basename(p)
        nums = [float(x) for x in _extract_floats(base)]
        if len(nums) >= 2:
            chain.append((nums[-2], nums[-1], p))
    chain.sort(key=lambda t: t[0])
    return chain


def _extract_floats(s: str):
    import re
    return re.findall(r"[-+]?\d*\.?\d+", s)


def load_tmap(path: str):
    """Load a transport map -> (row_ids, col_ids, matrix). Supports npz/h5ad-like."""
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        return list(z["row_ids"]), list(z["col_ids"]), z["matrix"]
    if ad is not None:
        T = ad.read_h5ad(path)
        return (list(np.asarray(T.obs_names)),
                list(np.asarray(T.var_names)),
                T.X.toarray() if hasattr(T.X, "toarray") else np.asarray(T.X))
    raise ValueError(f"unsupported transport-map format: {path}")
