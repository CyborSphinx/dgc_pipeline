"""
Build an effective B matrix from Norman lab CRISPRa Perturb-seq of TFs in Hs27
human fibroblasts (Southard, Ardy, Norman et al. 2024; Zenodo 15200179).

Input file (use the regression-aggregated 'mean' population):
   fibroblast_CRISPRa_mean_pop.h5ad        (1.7 GB)

Pipeline:
  1. Load h5ad. Detect the perturbation/guide column in .obs (defensive).
  2. Identify control rows (non-targeting / NTC).
  3. For each TF perturbation, compute Delta_x_TF = mean(perturbed) - mean(control).
     Result is a (G_human, n_TF) matrix Delta_X in HUMAN gene space.
  4. Orthology map human gene symbols -> mouse gene symbols (HGNC/MGI). For OSKM
     plus the known pluripotency factors this is a small fixed table; for the
     full panel pass an orthology_map dict.
  5. Restrict to the Schiebinger panel gene set (intersect mouse panel with
     orthology-mapped Delta_X rows).
  6. Deconvolve: B_TF = (S_k^T S_k + lambda I)^{-1} S_k^T Delta_x_TF, where
     S_k = I + M + M^2 + ... + M^{k-1} is the response-sum for the 24h-step
     operator built via InterleavedFusion.
  7. Save B (G_panel, n_TF) with column names (mouse symbol for OSKM, raw human
     symbol otherwise).

Honest caveats baked into the docstring:
  - Cell-context mismatch: Norman's Hs27 (human dermal fibroblast) is not
    Schiebinger's mouse embryonic fibroblast. The M operator deconvolving the
    Norman response was fit on MEF->iPSC. Treat the resulting B as a
    CONSTRUCTED CANDIDATE to test against the data-grounded b_OSKM_effect.
  - Single endpoint: Norman is profiled at one harvest time (~7 days). The
    deconvolution assumes the harvest time is N_STEPS_TO_HARVEST 24h-units
    after CRISPRa induction. Adjust based on the protocol.
  - Tikhonov bias: lambda > 0 biases B toward zero. The bias is much smaller
    than the variance from inverting near-zero singular values.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np


# --- minimal orthology map for OSKM + endogenous pluripotency factors ---
# Human symbol -> mouse symbol. Add to this dict as needed; pass `orthology_map`
# to override or extend.
HUMAN_TO_MOUSE_OSKM = {
    # Yamanaka factors (OSKM)
    "POU5F1": "Pou5f1",  "OCT4": "Pou5f1",
    "SOX2":   "Sox2",
    "KLF4":   "Klf4",
    "MYC":    "Myc",     "CMYC": "Myc",  "C-MYC": "Myc",
    # endogenous pluripotency factors
    "NANOG":  "Nanog",
    "ESRRB":  "Esrrb",
    "ZFP42":  "Zfp42",   "REX1": "Zfp42",
    "OBOX6":  "Obox6",
    "SALL4":  "Sall4",
    "LIN28A": "Lin28a",
    "NR5A2":  "Nr5a2",
    "PRDM14": "Prdm14",
    "TFCP2L1": "Tfcp2l1",
    "UTF1":   "Utf1",
    "DPPA2":  "Dppa2",
    "DPPA4":  "Dppa4",
    "DPPA5A": "Dppa5a",
}


class NormanBBuilder:
    def __init__(self, h5ad_path: str, orthology_map: dict | None = None,
                 control_keywords: Iterable[str] = ("nontargeting", "non-targeting",
                                                    "ntc", "control", "scramble",
                                                    "non_targeting")):
        self.h5ad_path = h5ad_path
        self.ortho = dict(HUMAN_TO_MOUSE_OSKM)
        if orthology_map:
            self.ortho.update({k.upper(): v for k, v in orthology_map.items()})
        self.control_keywords = tuple(k.lower() for k in control_keywords)
        self._A = None
        self._pert_col = None
        self._control_mask = None

    # -- step 1: load and inspect --------------------------------------------
    def load(self):
        if self._A is not None:
            return self._A
        import anndata as ad
        print(f"  loading {self.h5ad_path} ...")
        A = ad.read_h5ad(self.h5ad_path)
        print(f"    shape: {A.shape[0]} rows x {A.shape[1]} genes")
        print(f"    .obs columns: {list(A.obs.columns)}")
        self._A = A
        return A

    def detect_perturbation_column(self):
        """
        Find the .obs column that names the perturbed gene per row. Norman h5ad
        files typically use one of: 'gene', 'target', 'guide_target', 'perturbation',
        'sgRNA_target', 'target_gene'. We search defensively.
        """
        if self._pert_col is not None:
            return self._pert_col
        A = self.load()
        candidates = ["gene", "target_gene", "target", "perturbation",
                      "guide_target", "sgRNA_target", "tf", "perturbed_gene",
                      "feature_call", "pert_name", "pert_iname"]
        found = None
        for c in candidates:
            if c in A.obs.columns:
                vals = A.obs[c].astype(str).str.upper()
                # heuristic: column should have many unique values and include
                # at least one control keyword
                if vals.nunique() > 10 and any(
                    any(kw in str(v).lower() for kw in self.control_keywords)
                    for v in vals.unique()
                ):
                    found = c; break
        if found is None:
            # weakest: just take the categorical column with the most unique
            # values that look like gene symbols
            best, best_n = None, 0
            for c in A.obs.columns:
                try:
                    n = A.obs[c].astype(str).nunique()
                    if n > best_n and n < A.n_obs * 0.5:
                        best, best_n = c, n
                except Exception:
                    continue
            found = best
            print(f"    [defensive] could not auto-detect perturbation column "
                  f"by name; picking '{found}' (most unique non-cell-ID column)")
        else:
            print(f"    detected perturbation column: '{found}'")
        self._pert_col = found
        return found

    def control_mask(self):
        if self._control_mask is not None:
            return self._control_mask
        A = self.load(); col = self.detect_perturbation_column()
        vals = A.obs[col].astype(str).str.lower()
        mask = vals.apply(lambda v: any(kw in v for kw in self.control_keywords))
        mask = mask.values
        n_ctrl = int(mask.sum())
        if n_ctrl == 0:
            raise ValueError(
                f"no control rows found in column '{col}' using keywords "
                f"{self.control_keywords}. Inspect unique values and pass "
                f"control_keywords=...")
        print(f"    controls: {n_ctrl} rows ({100*n_ctrl/A.n_obs:.1f}% of total)")
        self._control_mask = mask
        return mask

    # -- step 2: per-TF Delta_x ----------------------------------------------
    def per_tf_delta(self, layer: str | None = None, min_cells_per_tf: int = 5):
        """
        For each non-control perturbation in the data, return
        delta[tf] = mean(perturbed rows) - mean(control rows)  in HUMAN gene space.

        Uses .X by default (Norman's mean_pop is regression-aggregated, so .X is
        already mean log1p-normalized expression per perturbation). Pass
        layer='counts' to use raw counts instead.
        """
        A = self.load(); col = self.detect_perturbation_column()
        ctrl = self.control_mask()
        X = A.X if layer is None else A.layers[layer]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        x_ctrl = X[ctrl].mean(axis=0)             # (G_human,)
        labels = A.obs[col].astype(str).str.upper().values
        # only non-control unique labels
        unique = []
        for v in np.unique(labels):
            if any(kw in v.lower() for kw in self.control_keywords):
                continue
            unique.append(v)
        tf_names = []
        deltas = []
        for tf in unique:
            sel = (labels == tf) & (~ctrl)
            if sel.sum() < min_cells_per_tf:
                continue
            x_tf = X[sel].mean(axis=0)
            deltas.append(x_tf - x_ctrl)
            tf_names.append(tf)
        Delta = np.vstack(deltas).T               # (G_human, n_TF)
        print(f"    per-TF delta: {Delta.shape[1]} TFs with >= {min_cells_per_tf} "
              f"cells, in {Delta.shape[0]} human genes")
        return Delta, tf_names, np.asarray(A.var_names)

    # -- step 3: orthology + panel restriction -------------------------------
    def to_mouse_panel(self, Delta_human, human_gene_names, mouse_panel_names,
                      missing_orthologs: str = "drop"):
        """
        Map (G_human, n_TF) Delta_human onto the mouse_panel_names ordering,
        using self.ortho for symbol->symbol lookups. Returns (G_panel, n_TF),
        and reports how many panel genes were covered.

        missing_orthologs: 'drop' = leave row at zero; 'warn' = same + count.
        """
        # build human->panel-index map: human symbol -> mouse symbol -> panel index
        mouse_idx = {g: i for i, g in enumerate(mouse_panel_names)}
        # also try direct (case-insensitive) match human->mouse if a human gene
        # happens to share a symbol with a mouse gene (capitalization difference)
        mouse_idx_ci = {g.lower(): i for i, g in enumerate(mouse_panel_names)}

        human_to_panel = {}
        for j, h in enumerate(human_gene_names):
            hU = str(h).upper()
            # explicit ortholog?
            m = self.ortho.get(hU)
            if m is not None and m in mouse_idx:
                human_to_panel[j] = mouse_idx[m]; continue
            # casefold match (e.g., "ACTB" <-> "Actb"): only safe for genes
            # whose symbols are conserved (most of them).
            if hU.lower() in mouse_idx_ci:
                human_to_panel[j] = mouse_idx_ci[hU.lower()]

        G_panel = len(mouse_panel_names)
        n_TF = Delta_human.shape[1]
        out = np.zeros((G_panel, n_TF))
        covered = 0
        for h_idx, p_idx in human_to_panel.items():
            out[p_idx] += Delta_human[h_idx]
            covered += 1
        print(f"    orthology: {covered}/{Delta_human.shape[0]} human rows "
              f"mapped to {len(set(human_to_panel.values()))}/{G_panel} mouse "
              f"panel rows ({100*len(set(human_to_panel.values()))/G_panel:.1f}%)")
        if missing_orthologs == "warn":
            unmapped = Delta_human.shape[0] - covered
            print(f"    [warn] {unmapped} human rows had no panel ortholog "
                  f"(dropped). pass orthology_map to extend coverage.")
        return out

    # -- step 4: deconvolution + B build -------------------------------------
    def build(self, M_24h_D, mouse_panel_names, n_steps_to_harvest=7,
              tikhonov_frac=1e-2, layer=None, min_cells_per_tf=5):
        """
        End-to-end: load, per-TF delta, orthology-map to mouse panel, deconvolve.

        Args
        ----
        M_24h_D            : (G_panel, G_panel) D matrix of the 24h operator
                             (from InterleavedFusion.build).
        mouse_panel_names  : list/array of mouse gene symbols, length G_panel.
        n_steps_to_harvest : number of 24h steps from CRISPRa induction to
                             harvest. Norman's protocol harvests ~7 days = 7.
        tikhonov_frac      : ridge lambda set to (tikhonov_frac * sigma_max(S))^2.
        layer              : h5ad layer to use ('counts' for raw, None for .X).

        Returns
        -------
        B           : (G_panel, n_TF) effective-B matrix (deconvolved).
        tf_names    : list of TF names in column order (uppercased human, with
                      mouse OSKM names where mapped).
        Delta       : (G_panel, n_TF) raw observed delta (no deconvolution), for
                      comparison.
        info        : dict with lambda used, S_k condition number, etc.
        """
        from .interleaved import make_response_sum, tikhonov_deconvolve
        Delta_human, tf_names_h, gene_names_h = self.per_tf_delta(
            layer=layer, min_cells_per_tf=min_cells_per_tf)
        Delta_panel = self.to_mouse_panel(Delta_human, gene_names_h,
                                          mouse_panel_names,
                                          missing_orthologs="warn")
        # build S_k and deconvolve
        S = make_response_sum(M_24h_D, n_steps_to_harvest)
        sigma = np.linalg.svd(S, compute_uv=False)
        cond = float(sigma[0] / (sigma[-1] + 1e-30))
        eff_rank = int(np.sum(sigma > 0.01 * sigma[0]))
        print(f"\n  S_{n_steps_to_harvest} spectrum: sigma_max={sigma[0]:.3f}  "
              f"sigma_min={sigma[-1]:.3e}  cond={cond:.3e}  "
              f"effective_rank={eff_rank}/{S.shape[0]}")
        B, lam = tikhonov_deconvolve(S, Delta_panel, lam=None,
                                     lam_frac=tikhonov_frac)
        print(f"  Tikhonov lambda = {lam:.3e} (={tikhonov_frac} x sigma_max^2)")

        # rename OSKM and friends with mouse symbols (the panel uses mouse)
        tf_names_final = []
        for t in tf_names_h:
            tf_names_final.append(self.ortho.get(t, t))
        return B, tf_names_final, Delta_panel, dict(
            lambda_used=lam, S_cond=cond, S_eff_rank=eff_rank,
            n_steps_to_harvest=n_steps_to_harvest,
            n_TF=len(tf_names_final), G_panel=len(mouse_panel_names),
            n_panel_covered=int(np.sum(np.abs(Delta_panel).sum(axis=1) > 0)),
        )

    @staticmethod
    def save(path, B, tf_names, mouse_panel_names, Delta=None, info=None):
        """Save as .npz compatible with the existing B-loading code in the notebook."""
        kw = dict(B=B, tf_names=np.asarray(tf_names, dtype=object),
                  gene_names=np.asarray(mouse_panel_names, dtype=object))
        if Delta is not None:
            kw["Delta_raw"] = Delta
        if info is not None:
            kw["info"] = np.asarray([info], dtype=object)
        np.savez(path, **kw)
        print(f"  saved B -> {path}  (shape {B.shape}, {len(tf_names)} TFs)")
