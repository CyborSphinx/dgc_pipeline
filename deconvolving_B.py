import os
import numpy as np
import pandas as pd
import anndata as ad
import sys

# Import your pipeline backend
sys.path.insert(0, os.path.abspath('.'))
from dgc_pipeline import FusedOperatorBuilder

def build_norman_b_matrix(traj_c, panel, be, norman_h5ad_path, output_b_matrix, k_steps=7, rcond=0.01):
    """
    Coarse-grains trajectories, builds the S7 operator, deconvolves the Norman dataset,
    and saves the inferred B matrix to cache.
    """
    # 1. Coarse-Grain (2-Time-Step M)
    print("Coarse-graining trajectories to 24-hour steps...")
    L, K, G = traj_c.shape
    min_len = min(len(np.arange(0, K, 2)), len(np.arange(1, K, 2)))
    traj_even = traj_c[:, 0:2*min_len:2, :]
    traj_odd  = traj_c[:, 1:1+2*min_len:2, :]
    traj_coarse = np.concatenate([traj_even, traj_odd], axis=0)

    print("Building 24-hour Fused Operator (M_coarse)...")
    fused_op_coarse, _ = FusedOperatorBuilder(be).build(
        traj_coarse, ridge=0.0, eps_fuse=0.05, gene_panel=panel, n_steps_hint=traj_coarse.shape[1]-1
    )
    M_coarse = fused_op_coarse.as_matrix(len(panel))

    # 2. Construct the S7 Deconvolution Operator
    print(f"Constructing S_{k_steps} operator for deconvolution...")
    S7 = np.zeros_like(M_coarse)
    M_power = np.eye(len(panel))

    for _ in range(k_steps):
        S7 += M_power
        M_power = M_power @ M_coarse

    print("Calculating Truncated Pseudo-Inverse (S7_pinv)...")
    S7_pinv = np.linalg.pinv(S7, rcond=rcond)

    # 3. Load Norman Data & Extract Baseline
    print(f"Loading Norman dataset from {norman_h5ad_path}...")
    adata_norman = ad.read_h5ad(norman_h5ad_path)

    # Use the exact Norman metadata labels
    PERT_COLUMN = 'guide_target'  
    CTRL_LABEL = 'non' 

    control_mask = adata_norman.obs[PERT_COLUMN] == CTRL_LABEL
    baseline_expr = np.array(adata_norman[control_mask].X.mean(axis=0)).flatten()

    perturbed_tfs = [tf for tf in adata_norman.obs[PERT_COLUMN].unique() if tf != CTRL_LABEL]
    print(f"Found {len(perturbed_tfs)} unique TF perturbations.")

    # 4. Human-to-Mouse Ortholog Mapping (FIXED)
    print("Mapping Human HGNC symbols to Mouse panel...")
    panel_human = [str(g).upper() for g in panel]
    
    # Extract the HGNC symbols from the 'gene_name' column instead of the Ensembl index
    norman_gene_symbols = [str(g).upper() for g in adata_norman.var['gene_name']]

    # Find the overlap
    human_to_panel_idx = {gene: idx for idx, gene in enumerate(panel_human) if gene in norman_gene_symbols}
    print(f"Ortholog overlap: {len(human_to_panel_idx)} out of {len(panel)} panel genes found in Norman data.")

    # Map the Norman array indices to Panel indices for fast projection
    projection_indices = []
    for h_gene, p_idx in human_to_panel_idx.items():
        n_idx = norman_gene_symbols.index(h_gene)
        projection_indices.append((n_idx, p_idx))

    # 5. Construct B Matrix
    print("Deconvolving Delta x vectors into B matrix columns...")
    B_matrix = np.zeros((len(panel), len(perturbed_tfs)))

    for j, tf in enumerate(perturbed_tfs):
        tf_mask = adata_norman.obs[PERT_COLUMN] == tf
        tf_expr = np.array(adata_norman[tf_mask].X.mean(axis=0)).flatten()
        delta_x_raw = tf_expr - baseline_expr
        
        delta_x_panel = np.zeros(len(panel))
        for n_idx, p_idx in projection_indices:
            delta_x_panel[p_idx] = delta_x_raw[n_idx]
            
        B_matrix[:, j] = S7_pinv @ delta_x_panel

    # 6. Save
    os.makedirs(os.path.dirname(output_b_matrix), exist_ok=True)
    np.savez_compressed(output_b_matrix, B=B_matrix, tf_names=np.array(perturbed_tfs))

    print(f"\nSuccess! B matrix shape {B_matrix.shape} saved to {output_b_matrix}")
    return output_b_matrix