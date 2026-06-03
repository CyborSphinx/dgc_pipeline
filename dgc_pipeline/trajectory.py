"""
Trajectory construction: expected-path trajectories from transport maps,
global centering, per-seed iPSC fate, and a cache save/load (the "Option B" that
the notebook uses so it depends only on this package).

TrajectoryBuilder(tmap_chain, A, genes)
  .build(cell_filter=None)        -> traj (L,K,G) raw expression, seed_ids
  .center(traj)                   -> (traj_c, xbar)
  .seed_fate(seed_ids, ipsc_ids)  -> (L,) iPSC fate probability per seed
  .panel(traj, n_top)             -> indices of the n_top most-variable genes
  .save(path, traj_c, xbar, panel, seed_ids=None, fate=None)
  TrajectoryBuilder.load(path)    -> dict with traj_c, xbar, panel, ...

The maths is identical to the monolith's expected_path_trajectories /
center_global / compute_seed_fate; only the packaging differs.
"""
from __future__ import annotations

import numpy as np

from .io import load_tmap


class TrajectoryBuilder:
    def __init__(self, tmap_chain, A, genes):
        self.tmap_chain = tmap_chain          # list of (day0, day1, path)
        self.A = A
        self.genes = np.asarray(genes)

    # -- expected-path trajectories --------------------------------------------
    def build(self, cell_filter=None):
        A, genes = self.A, self.genes
        gidx = {c: i for i, c in enumerate(np.asarray(A.obs_names))}
        Xall = A.X
        G = len(genes)
        maps = [load_tmap(p) for (_, _, p) in self.tmap_chain]
        r0 = maps[0][0]

        seed_idx = list(range(len(r0)))
        if cell_filter is not None:
            keep = set(cell_filter)
            seed_idx = [i for i in seed_idx if r0[i] in keep]
        if not seed_idx:
            seed_idx = list(range(len(r0)))
        seed_idx = np.asarray(seed_idx)

        K = len(self.tmap_chain) + 1
        L = len(seed_idx)
        seed_ids = [r0[i] for i in seed_idx]

        def expr_rows(ids):
            idxs = [gidx.get(c, -1) for c in ids]
            E = np.zeros((len(ids), G))
            valid = [(r, ii) for r, ii in enumerate(idxs) if ii >= 0]
            if valid:
                rows = [ii for _, ii in valid]
                sub = Xall[rows, :]
                sub = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
                for (r, _), e in zip(valid, sub):
                    E[r] = e
            return E

        traj = np.zeros((L, K, G))
        traj[:, 0, :] = expr_rows(seed_ids)

        rids0, cids0, M0 = maps[0]
        P = M0[seed_idx, :].astype(float)
        P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
        E0 = expr_rows(cids0)
        traj[:, 1, :] = P @ E0

        for k in range(1, len(maps)):
            rids_k, cids_k, Mk = maps[k]
            prev_cids = maps[k - 1][1]
            if not np.array_equal(prev_cids, rids_k):
                pos = {cid: idx for idx, cid in enumerate(prev_cids)}
                sel = np.array([pos.get(cid, -1) for cid in rids_k])
                P_new = np.zeros((L, len(rids_k)))
                good = sel >= 0
                P_new[:, good] = P[:, sel[good]]
                P = P_new
            P = P @ Mk
            P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
            Ek = expr_rows(cids_k)
            traj[:, k + 1, :] = P @ Ek

        print(f"  expected-path trajectories: L={L} seeds, K={K} days, G={G}")
        return traj, seed_ids

    # -- centering / panel -----------------------------------------------------
    @staticmethod
    def center(traj):
        xbar = traj.reshape(-1, traj.shape[2]).mean(axis=0)
        return traj - xbar[None, None, :], xbar

    @staticmethod
    def panel(traj, n_top=None):
        """Indices of the n_top most temporally-variable genes (all if None)."""
        G = traj.shape[2]
        if n_top is None or n_top >= G:
            return np.arange(G)
        var = traj.reshape(-1, G).var(axis=0)
        return np.sort(np.argsort(-var)[:n_top])

    # -- per-seed iPSC fate (survivorship weight) ------------------------------
    def seed_fate(self, seed_ids, ipsc_ids):
        """
        Propagate each seed's descendant distribution to the final day and return
        the fraction of terminal mass on iPSC-annotated cells. This is the
        survivorship weight: fate[l] ~ P(seed l reaches iPSC). Use it to fate-weight
        the operator / input recovery so the model caters to the SUCCESSFUL path
        (failed lineages are a lost cause for control).
        """
        maps = [load_tmap(p) for (_, _, p) in self.tmap_chain]
        r0 = maps[0][0]
        pos0 = {c: i for i, c in enumerate(r0)}
        seed_rows = np.array([pos0.get(c, -1) for c in seed_ids])
        L = len(seed_ids)
        rids0, cids0, M0 = maps[0]
        P = np.zeros((L, len(cids0)))
        good = seed_rows >= 0
        P[good] = M0[seed_rows[good], :]
        P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
        for k in range(1, len(maps)):
            rids_k, cids_k, Mk = maps[k]
            prev_cids = maps[k - 1][1]
            if not np.array_equal(prev_cids, rids_k):
                p = {cid: idx for idx, cid in enumerate(prev_cids)}
                sel = np.array([p.get(cid, -1) for cid in rids_k])
                Pn = np.zeros((L, len(rids_k)))
                g = sel >= 0
                Pn[:, g] = P[:, sel[g]]
                P = Pn
            P = P @ Mk
            P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
        final_cids = maps[-1][1]
        is_ipsc = np.array([1.0 if c in set(ipsc_ids) else 0.0 for c in final_cids])
        fate = P @ is_ipsc
        print(f"  seed fate probabilities: min={fate.min():.3f} "
              f"max={fate.max():.3f} mean={fate.mean():.3f}")
        return fate

    # -- cache (Option B) ------------------------------------------------------
    @staticmethod
    def save(path, traj_c, xbar, panel, seed_ids=None, fate=None):
        """Save everything the analysis/notebook needs for the next run."""
        kw = dict(traj_c=traj_c, xbar=xbar, panel=np.asarray(panel))
        if seed_ids is not None:
            kw["seed_ids"] = np.asarray(seed_ids, dtype=object)
        if fate is not None:
            kw["fate"] = np.asarray(fate)
        np.savez(path, **kw)
        print(f"  trajectory cache saved -> {path}")

    @staticmethod
    def load(path):
        Z = np.load(path, allow_pickle=True)
        out = {k: Z[k] for k in Z.files}
        return out
