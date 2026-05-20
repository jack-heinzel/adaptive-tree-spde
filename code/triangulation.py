"""
Adaptive triangulation built on a recursive equal-count kd-tree partition.

Each leaf cell of the kd-tree contributes one node (the centroid of its data
points, or the cell midpoint if empty) to a Delaunay triangulation. This gives
a mesh that is automatically denser where data is dense.

Refinement splits a leaf cell and adds a node; coarsening merges sibling leaf
pairs and removes a node. The Delaunay triangulation is rebuilt after each
adaptation step. The domain boundary corners are always included as fixed nodes
so the triangulation covers the entire domain.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from math import factorial
from typing import Optional

import numpy as np
from scipy.sparse import coo_array
from scipy.spatial import Delaunay


@dataclass
class Cell:
    lo: np.ndarray
    hi: np.ndarray
    data_idx: np.ndarray          # indices into the global points array
    parent: Optional[Cell] = field(default=None, repr=False)
    left: Optional[Cell] = field(default=None, repr=False)
    right: Optional[Cell] = field(default=None, repr=False)
    split_axis: Optional[int] = None
    split_val: Optional[float] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    @property
    def count(self) -> int:
        return len(self.data_idx)

    @property
    def sibling(self) -> Optional[Cell]:
        if self.parent is None:
            return None
        p = self.parent
        return p.right if (p.left is self) else p.left

    def node(self, points: np.ndarray) -> np.ndarray:
        """Data centroid if the cell contains points, otherwise cell midpoint."""
        if self.count > 0:
            return points[self.data_idx].mean(axis=0)
        return 0.5 * (self.lo + self.hi)


class AdaptiveKDMesh:
    """
    Adaptive Delaunay mesh driven by a kd-tree partition of the data.

    Parameters
    ----------
    points : array-like, shape (n, d)
        Observed point pattern that drives the partition.
    bounds : list of (float, float)
        Domain per dimension: [(x0_min, x0_max), ...].

    Usage
    -----
    mesh = AdaptiveKDMesh(points, bounds)
    mesh.refine(max_count=20)       # split all leaves with > 20 points
    mesh.coarsen(min_count=5)       # merge sibling pairs with < 5 combined
    tri  = mesh.triangulation       # scipy.spatial.Delaunay object
    nodes = mesh.nodes              # (m, d) node coordinates
    """

    def __init__(self, points: np.ndarray, bounds: list[tuple[float, float]]):
        self.points = np.asarray(points, dtype=float)
        self.bounds = bounds
        self.lo = np.array([b[0] for b in bounds])
        self.hi = np.array([b[1] for b in bounds])

        self.root = Cell(
            lo=self.lo.copy(),
            hi=self.hi.copy(),
            data_idx=np.arange(len(self.points)),
        )
        self._rebuild_triangulation()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def triangulation(self) -> Delaunay:
        return self._tri

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes

    def leaves(self) -> list[Cell]:
        """All leaf cells in depth-first order."""
        result: list[Cell] = []
        stack = [self.root]
        while stack:
            cell = stack.pop()
            if cell.is_leaf:
                result.append(cell)
            else:
                stack.append(cell.right)
                stack.append(cell.left)
        return result

    def refine(
        self,
        cells: Optional[list[Cell]] = None,
        max_count: Optional[int] = None,
    ) -> AdaptiveKDMesh:
        """
        Split leaf cells and rebuild the triangulation.

        Pass ``cells`` to split specific leaves, or ``max_count`` to split all
        leaves whose point count exceeds the threshold.
        """
        if cells is None and max_count is not None:
            cells = [c for c in self.leaves() if c.count > max_count]
        elif cells is None:
            raise ValueError("Provide either cells or max_count")

        for cell in cells:
            self._split_cell(cell)

        self._rebuild_triangulation()
        return self

    def coarsen(
        self,
        cells: Optional[list[Cell]] = None,
        min_count: Optional[int] = None,
    ) -> AdaptiveKDMesh:
        """
        Merge sibling leaf pairs and rebuild the triangulation.

        Pass ``cells`` to merge the parents of those leaves (both siblings must
        be leaves), or ``min_count`` to merge all sibling pairs whose combined
        count falls below the threshold.
        """
        if cells is None and min_count is not None:
            cells = self._find_mergeable_parents(min_count)
        elif cells is None:
            raise ValueError("Provide either cells or min_count")

        for cell in cells:
            self._merge_cell(cell)

        self._rebuild_triangulation()
        return self

    def triangle_counts(self) -> np.ndarray:
        """Number of data points assigned to each Delaunay simplex."""
        simplex_idx = self._tri.find_simplex(self.points)
        valid = simplex_idx >= 0
        return np.bincount(simplex_idx[valid], minlength=len(self._tri.simplices))

    def leaf_counts(self) -> np.ndarray:
        return np.array([c.count for c in self.leaves()])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_cell(self, cell: Cell) -> None:
        if not cell.is_leaf:
            raise ValueError("Can only split leaf cells")
        if cell.count <= 1:
            return  # cannot split a cell with 0 or 1 points

        # Split along the longest axis; use data median when points exist
        axis = int(np.argmax(cell.hi - cell.lo))
        if cell.count >= 2:
            val = float(np.median(self.points[cell.data_idx, axis]))
            # Clamp away from the boundary so neither child is degenerate
            eps = (cell.hi[axis] - cell.lo[axis]) * 1e-8
            val = float(np.clip(val, cell.lo[axis] + eps, cell.hi[axis] - eps))
        else:
            val = float(0.5 * (cell.lo[axis] + cell.hi[axis]))

        left_hi = cell.hi.copy(); left_hi[axis] = val
        right_lo = cell.lo.copy(); right_lo[axis] = val

        left_mask = self.points[cell.data_idx, axis] <= val
        cell.left = Cell(cell.lo.copy(), left_hi, cell.data_idx[left_mask],  parent=cell)
        cell.right = Cell(right_lo, cell.hi.copy(), cell.data_idx[~left_mask], parent=cell)
        cell.split_axis = axis
        cell.split_val = val
        cell.data_idx = np.empty(0, dtype=int)  # interior nodes hold no data

    def _merge_cell(self, cell: Cell) -> None:
        """Merge a non-leaf cell whose both children are leaves."""
        if cell.is_leaf:
            raise ValueError("Cell is already a leaf")
        if not (cell.left.is_leaf and cell.right.is_leaf):
            raise ValueError("Both children must be leaves to merge")

        cell.data_idx = np.concatenate([cell.left.data_idx, cell.right.data_idx])
        cell.left = cell.right = None
        cell.split_axis = cell.split_val = None

    def _find_mergeable_parents(self, min_count: int) -> list[Cell]:
        """Parent cells whose two leaf children have combined count < min_count."""
        seen: set[int] = set()
        candidates: list[Cell] = []
        for cell in self.leaves():
            sib = cell.sibling
            if (
                sib is not None
                and sib.is_leaf
                and id(cell.parent) not in seen
                and cell.count + sib.count < min_count
            ):
                seen.add(id(cell.parent))
                candidates.append(cell.parent)
        return candidates

    def _boundary_nodes(self) -> np.ndarray:
        """All 2^d corners of the bounding box."""
        d = len(self.bounds)
        corners = np.array([
            [self.lo[i] if bit == 0 else self.hi[i] for i, bit in enumerate(bits)]
            for bits in product(range(2), repeat=d)
        ])
        return corners

    def _rebuild_triangulation(self) -> None:
        leaf_nodes = np.array([c.node(self.points) for c in self.leaves()])
        all_nodes = np.vstack([self._boundary_nodes(), leaf_nodes])
        self._nodes = all_nodes
        self._tri = Delaunay(all_nodes)

    def fem_matrices(self, kappa: float) -> tuple:
        """
        Assemble and return (C, G, K) for this mesh.
        Convenience wrapper around :func:`assemble_fem_matrices`.
        K = kappa^2 * C + G.
        """
        C, G = assemble_fem_matrices(self.nodes, self.triangulation.simplices)
        K = kappa ** 2 * C + G
        return C, G, K


# ---------------------------------------------------------------------------
# FEM matrix assembly (Lindgren, Rue & Lindström 2011 notation)
# ---------------------------------------------------------------------------

def assemble_fem_matrices(
    nodes: np.ndarray,
    simplices: np.ndarray,
    H=None,
    kappa=None,
) -> tuple:
    """
    Assemble FEM matrices for the generalized Matérn SPDE on a simplicial mesh.

    Standard SPDE (Lindgren et al. 2011):
        (kappa^2 - Delta)^(alpha/2) x = W

    Generalized SPDE (Section 4):
        (kappa(s)^2 - div H(s) grad)^(alpha/2) x = W

    H is the diffusion tensor controlling anisotropy and non-stationarity.
    Only the stiffness matrix changes; H is evaluated at each simplex centroid.

    Parameters
    ----------
    nodes : array (n_nodes, d)
    simplices : array (n_simplices, d+1)
    H : diffusion tensor, one of:
        None              — isotropic (H = identity, recovers standard G)
        array (d, d)      — stationary anisotropy (constant H across domain)
        callable          — non-stationary; H(centroids) receives (n_sim, d)
                            and must return (n_sim, d, d)
    kappa : range parameter, one of:
        None              — do not assemble K; return only C, G
        scalar            — stationary range; K = kappa^2 * C + G
        callable          — non-stationary; kappa(centroids) receives (n_sim, d)
                            and must return (n_sim,)

    Returns
    -------
    C, G        if kappa is None
    C, G, K     if kappa is provided
    """
    nodes = np.asarray(nodes, dtype=float)
    simplices = np.asarray(simplices, dtype=int)

    n_nodes = len(nodes)
    n_sim = len(simplices)
    d = nodes.shape[1]
    n_loc = d + 1

    ref_grads = np.zeros((n_loc, d))
    ref_grads[0] = -1.0
    ref_grads[1:] = np.eye(d)

    pts = nodes[simplices]                                      # (n_sim, n_loc, d)
    B = (pts[:, 1:, :] - pts[:, :1, :]).transpose(0, 2, 1)    # (n_sim, d, d)
    vols = np.abs(np.linalg.det(B)) / factorial(d)             # (n_sim,)
    centroids = pts.mean(axis=1)                                # (n_sim, d)

    B_inv = np.linalg.inv(B)
    phys_grads = np.einsum("ik,tkj->tij", ref_grads, B_inv)   # (n_sim, n_loc, d)

    # Generalized stiffness G̃_ij = integral grad(phi_i)^T H(s) grad(phi_j)
    if H is None:
        G_local = np.einsum("t,tik,tjk->tij", vols, phys_grads, phys_grads)
    elif callable(H):
        H_vals = H(centroids)                                   # (n_sim, d, d)
        G_local = np.einsum("t,tik,tkl,tjl->tij", vols, phys_grads, H_vals, phys_grads)
    else:
        H_arr = np.asarray(H, dtype=float)                     # (d, d)
        G_local = np.einsum("t,tik,kl,tjl->tij", vols, phys_grads, H_arr, phys_grads)

    # Mass matrix C_ij = integral phi_i phi_j
    C_template = (np.ones((n_loc, n_loc)) + np.eye(n_loc)) / ((d + 1) * (d + 2))
    C_local = vols[:, None, None] * C_template                 # (n_sim, n_loc, n_loc)

    loc_i, loc_j = np.meshgrid(np.arange(n_loc), np.arange(n_loc), indexing="ij")
    rows = simplices[:, loc_i].ravel()
    cols = simplices[:, loc_j].ravel()

    C = coo_array((C_local.ravel(), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    G = coo_array((G_local.ravel(), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    if kappa is None:
        return C, G

    # Operator K = kappa(s)^2 * C + G, with kappa evaluated at centroids
    if callable(kappa):
        kappa_vals = kappa(centroids)                           # (n_sim,)
        C_kappa_local = kappa_vals[:, None, None] ** 2 * C_local
        C_kappa = coo_array(
            (C_kappa_local.ravel(), (rows, cols)), shape=(n_nodes, n_nodes)
        ).tocsr()
        K = C_kappa + G
    else:
        K = float(kappa) ** 2 * C + G

    return C, G, K


def precision_matrix(C, G, kappa=None, alpha: int = 2):
    """
    Build the SPDE precision matrix Q for the Matérn field.

    For the SPDE (kappa^2 - div H grad)^(alpha/2) x = W:
      alpha=1 : Q = K
      alpha=2 : Q = K C_lumped^{-1} K
      alpha=k : Q = K (C_lumped^{-1} K)^{k-1}

    The lumped mass matrix diag(C @ 1) keeps Q sparse.

    Parameters
    ----------
    C : sparse mass matrix
    G : sparse stiffness matrix, OR a pre-assembled K (when kappa=None)
    kappa : scalar range parameter.
        If None, G is treated as the pre-assembled operator K — use this when
        K was built by assemble_fem_matrices with a callable kappa.
    alpha : smoothness order (alpha = nu + d/2)

    Returns
    -------
    K : sparse — operator matrix kappa^2*C + G  (or G itself if kappa=None)
    Q : sparse — precision matrix
    """
    from scipy.sparse import diags

    K = float(kappa) ** 2 * C + G if kappa is not None else G

    c_lumped = np.asarray(C.sum(axis=1)).ravel()
    C_inv = diags(1.0 / c_lumped)

    Q = K
    for _ in range(alpha - 1):
        Q = Q @ C_inv @ K

    return K, Q
