"""
dgc_pipeline — Waddington-OT multi-trajectory DMD vs static-B for TF ranking
in cellular reprogramming.

Refactor of the original monolithic wot_dmd_test_v2.py into a modular,
object-oriented package. Public surface mirrors the original behaviour exactly
(same CLI, same printed output, numerically identical results); the internals
are reorganised for readability and testability.

Module map
----------
backend    : GPU/CPU linear-algebra backend (Backend), config constants
io         : data loading (expression, transport maps, GMT, tables)
operators  : Operator ABC + Rank1Operator / DiffPlusIOperator / DenseOperator,
             and the OperatorSequence used by every scorer
trajectory : TrajectoryBuilder (expected-path), centering, seed fate
fusion     : FusedOperatorBuilder (dimension-aware lineage fusion, eps_fuse),
             DGCOperatorBuilder (rank-one mean-path)
diffdmd    : DiffDMD (per-pair difference-DMD then fuse; input recovery)
bmatrix    : BMatrix builders (binding / coexpression) + panel restriction
targets    : iPSC target construction (centroid, biological purity-gated)
scoring    : Scorer hierarchy (BOnly, Dynamics, ControlBestDay, Combo) + ranking
pipeline   : Pipeline orchestrator (wires everything; owns the run)
cli        : argument parsing + entry point (main)
"""
from .backend import Backend
from .operators import (Operator, OperatorSequence, Rank1Operator,
                        DiffPlusIOperator, DenseOperator)
from .fusion import FusedOperatorBuilder, DGCOperatorBuilder
from .targets import TargetBuilder
from .trajectory import TrajectoryBuilder
from .diagnostics import fusion_vs_averaging, forcing_svd, forcing_geometry
from .interleaved import InterleavedFusion, make_response_sum, tikhonov_deconvolve
from .bmatrix import NormanBBuilder, HUMAN_TO_MOUSE_OSKM
from .scoring import (Scorer, BOnlyScorer, DynamicsScorer,
                      ControlBestDayScorer, ranks_of_known, precision_at_k)

__all__ = [
    "Backend",
    "Operator", "OperatorSequence",
    "Rank1Operator", "DiffPlusIOperator", "DenseOperator",
    "FusedOperatorBuilder", "DGCOperatorBuilder",
    "TargetBuilder", "TrajectoryBuilder",
    "Scorer", "BOnlyScorer", "DynamicsScorer", "ControlBestDayScorer",
    "ranks_of_known", "precision_at_k",
    "fusion_vs_averaging", "forcing_svd", "forcing_geometry",
    "InterleavedFusion", "make_response_sum", "tikhonov_deconvolve",
    "NormanBBuilder", "HUMAN_TO_MOUSE_OSKM",
]
