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
"""
# from .operators import (Operator, OperatorSequence, Rank1Operator,
#                         DiffPlusIOperator, DenseOperator)
from .fusion import FusedOperatorBuilder
from .trajectory import TrajectoryBuilder

__all__ = [
    "FusedOperatorBuilder", "TrajectoryBuilder",
]
# "Operator", "OperatorSequence",
#     "Rank1Operator", "DiffPlusIOperator", "DenseOperator",