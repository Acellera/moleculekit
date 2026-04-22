"""KDTree port from SciPy.

The implementation was taken from the SciPy project (commit
f82b3f9d3c30cc6e7f906b0d00b8b43bfea02e5c) and modified to drop the
``scipy.sparse`` and ``scipy._lib`` dependencies so that ``moleculekit``
does not need to pull in SciPy.  See :mod:`moleculekit.kdtree._ckdtree`
for details on the changes.

The public API is ``KDTree`` and the lower-level ``cKDTree``.
"""
from moleculekit.kdtree._ckdtree import cKDTree, cKDTreeNode
from moleculekit.kdtree._kdtree import KDTree, minkowski_distance, minkowski_distance_p

__all__ = [
    "KDTree",
    "cKDTree",
    "cKDTreeNode",
    "minkowski_distance",
    "minkowski_distance_p",
]
