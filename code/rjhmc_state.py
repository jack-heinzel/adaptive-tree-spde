"""
Fixed-shape BFS tree state for reversible-jump MCMC over AdaptiveKDMesh.

Tree serialisation
------------------
Nodes are stored in **breadth-first (BFS) heap order**: the root is at index 0;
node i has its left child at 2i+1 and right child at 2i+2.  For a tree of at
most max_depth levels, the backing arrays have length

    max_nodes = 2^(max_depth+1) − 1.

Three JAX arrays, all of shape (max_nodes,), form a TreeState:

    split_axis  int32   split axis  for internal nodes,  -1   for leaves / absent
    split_val   float64 split value for internal nodes,  NaN  for leaves / absent
    rates       float64 log-intensity at leaf nodes,     NaN  for internals / absent

Node classification at BFS index i:
    occupied  ←  (split_axis[i] >= 0) | isfinite(rates[i])
    internal  ←  split_axis[i] >= 0
    leaf      ←  (split_axis[i] < 0)  & isfinite(rates[i])
    absent    ←  (split_axis[i] < 0)  & isnan(rates[i])

Because every sample has exactly the same (max_nodes,) shape, TreeState can be
used as the position in a jax.lax.scan loop.

RJMCMC view
-----------
Within-model moves (Hilbert HMC):
    sample `rates` with split_axis / split_val fixed.
    HMC position = rates (NaN at non-leaf slots, masked in the potential).

Transdimensional moves (birth / death):
    birth_update  — split a leaf: activates two child slots
    death_update  — merge two leaf children into their parent leaf

Wrap / unwrap
-------------
    wrap_to_tree_state   : AdaptiveKDMesh + per-leaf rates  →  TreeState
    unwrap_to_mesh       : TreeState + data + bounds        →  (AdaptiveKDMesh, leaf_rates)

The `rates` vector in wrap/unwrap is ordered to match mesh.leaves(), which uses
DFS (left-first pre-order) — the same order as FEM matrix assembly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import jax.numpy as jnp

from triangulation import AdaptiveKDMesh, Cell


# ── BFS heap index arithmetic ──────────────────────────────────────────────────

def bfs_left(i: int) -> int:
    return 2 * i + 1

def bfs_right(i: int) -> int:
    return 2 * i + 2

def bfs_parent(i: int) -> int | None:
    return (i - 1) // 2 if i > 0 else None

def bfs_depth(i: int) -> int:
    """Depth of BFS index i (root is depth 0)."""
    return int(np.floor(np.log2(i + 1))) if i > 0 else 0

def max_nodes_for_depth(max_depth: int) -> int:
    return 2 ** (max_depth + 1) - 1


# ── Primary data structure ────────────────────────────────────────────────────

class TreeState:
    """
    Fixed-shape BFS tree state.  All three arrays have shape (max_nodes,).

    This is intentionally a plain class (not a JAX PyTree or NamedTuple) so
    that split_axis (int) and rates (float) can use different dtypes.  Pass
    rates directly to HMC as the position; use split_axis / split_val as
    static structural state between RJ moves.

    For use in jax.lax.scan, construct a dict or NamedTuple from the three
    arrays:  {'split_axis': ..., 'split_val': ..., 'rates': ...}.
    """

    __slots__ = ("split_axis", "split_val", "rates")

    def __init__(
        self,
        split_axis: jnp.ndarray,
        split_val:  jnp.ndarray,
        rates:      jnp.ndarray,
    ):
        self.split_axis = split_axis  # (max_nodes,) int32
        self.split_val  = split_val   # (max_nodes,) float64
        self.rates      = rates       # (max_nodes,) float64

    # ── shape properties ──────────────────────────────────────────────────────

    @property
    def max_nodes(self) -> int:
        return int(self.split_axis.shape[0])

    @property
    def max_depth(self) -> int:
        return int(np.floor(np.log2(self.max_nodes + 1))) - 1

    # ── node classification (JAX arrays) ─────────────────────────────────────

    def occupied_mask(self) -> jnp.ndarray:
        """bool (max_nodes,): True for every node present in the current tree."""
        return (self.split_axis >= 0) | jnp.isfinite(self.rates)

    def leaf_mask(self) -> jnp.ndarray:
        """bool (max_nodes,): True for leaf nodes."""
        return (self.split_axis < 0) & jnp.isfinite(self.rates)

    def internal_mask(self) -> jnp.ndarray:
        """bool (max_nodes,): True for internal (split) nodes."""
        return self.split_axis >= 0

    # ── numpy helpers (for Python-level RJ logic) ─────────────────────────────

    def leaf_bfs_indices(self) -> np.ndarray:
        """BFS indices of all leaf nodes, in BFS order (left-to-right per level)."""
        return np.where(np.array(self.leaf_mask()))[0]

    def n_leaves(self) -> int:
        return int(self.leaf_mask().sum())

    def n_nodes(self) -> int:
        return int(self.occupied_mask().sum())

    # ── functional updates (return new TreeState, no mutation) ───────────────

    def birth_update(
        self,
        leaf_i: int,
        split_axis: int,
        split_val: float,
        rate_left: float,
        rate_right: float,
    ) -> "TreeState":
        """
        Split leaf at BFS index `leaf_i` into two children.

        The parent slot (leaf_i) becomes an internal node; child slots
        bfs_left(leaf_i) and bfs_right(leaf_i) become new leaves.

        Parameters
        ----------
        leaf_i     : BFS index of the leaf to split.
        split_axis : axis along which to split (0 to d-1).
        split_val  : coordinate value of the split.
        rate_left  : log-intensity proposed for the left child leaf.
        rate_right : log-intensity proposed for the right child leaf.
        """
        li = bfs_left(leaf_i)
        ri = bfs_right(leaf_i)
        if ri >= self.max_nodes:
            raise ValueError(
                f"Birth at BFS index {leaf_i} would create children at {ri} "
                f">= max_nodes={self.max_nodes}.  Increase max_depth."
            )
        sa  = self.split_axis.at[leaf_i].set(split_axis)
        sv  = self.split_val.at[leaf_i].set(split_val)
        r   = (
            self.rates
            .at[leaf_i].set(jnp.nan)
            .at[li].set(rate_left)
            .at[ri].set(rate_right)
        )
        return TreeState(sa, sv, r)

    def death_update(
        self,
        internal_i: int,
        merged_rate: float,
    ) -> "TreeState":
        """
        Merge the two leaf children of `internal_i` back into a single leaf.

        Both children (bfs_left(internal_i) and bfs_right(internal_i)) must
        currently be leaves.  After the merge, `internal_i` becomes a leaf
        with `merged_rate`; the child slots are zeroed out.
        """
        li = bfs_left(internal_i)
        ri = bfs_right(internal_i)
        sa = self.split_axis.at[internal_i].set(-1)
        sv = self.split_val.at[internal_i].set(jnp.nan)
        r  = (
            self.rates
            .at[internal_i].set(merged_rate)
            .at[li].set(jnp.nan)
            .at[ri].set(jnp.nan)
        )
        return TreeState(sa, sv, r)

    def update_rates(self, new_rates: jnp.ndarray) -> "TreeState":
        """Return a new TreeState with rates replaced (used after HMC step)."""
        return TreeState(self.split_axis, self.split_val, new_rates)

    def __repr__(self) -> str:
        return (
            f"TreeState(max_depth={self.max_depth}, "
            f"n_nodes={self.n_nodes()}, n_leaves={self.n_leaves()})"
        )


# ── Constructors ──────────────────────────────────────────────────────────────

def empty_tree_state(max_depth: int, dtype=jnp.float64) -> TreeState:
    """All-absent tree state (no nodes allocated)."""
    mn = max_nodes_for_depth(max_depth)
    return TreeState(
        split_axis=jnp.full(mn, -1,       dtype=jnp.int32),
        split_val =jnp.full(mn, jnp.nan,  dtype=dtype),
        rates     =jnp.full(mn, jnp.nan,  dtype=dtype),
    )

def root_only_state(rate: float, max_depth: int, dtype=jnp.float64) -> TreeState:
    """Tree with a single root leaf at the given log-intensity."""
    s = empty_tree_state(max_depth, dtype)
    return TreeState(s.split_axis, s.split_val, s.rates.at[0].set(rate))


# ── Wrap: AdaptiveKDMesh → TreeState ─────────────────────────────────────────

def wrap_to_tree_state(
    mesh: AdaptiveKDMesh,
    rates: np.ndarray,
    max_depth: int,
    dtype=jnp.float64,
) -> TreeState:
    """
    Serialise an AdaptiveKDMesh together with per-leaf log-intensities into a
    fixed-shape TreeState.

    Parameters
    ----------
    mesh      : AdaptiveKDMesh whose tree structure to encode.
    rates     : (n_leaves,) log-intensities in DFS order (matching mesh.leaves()).
    max_depth : depth of the backing BFS array; must be >= tree depth.

    Returns
    -------
    TreeState with split_axis, split_val, and rates populated.
    """
    mn = max_nodes_for_depth(max_depth)
    sa_arr = -np.ones(mn, dtype=np.int32)
    sv_arr = np.full(mn, np.nan)
    rt_arr = np.full(mn, np.nan)

    rates = np.asarray(rates, dtype=float)

    # BFS pass: assign every Cell a BFS index; record internal node splits.
    cell_to_bfs: dict[int, int] = {}
    queue: deque[tuple[Cell, int]] = deque([(mesh.root, 0)])
    while queue:
        cell, bfs_i = queue.popleft()
        if bfs_i >= mn:
            raise ValueError(
                f"Tree depth exceeds max_depth={max_depth}.  "
                f"Node at BFS index {bfs_i} >= max_nodes={mn}."
            )
        cell_to_bfs[id(cell)] = bfs_i
        if not cell.is_leaf:
            sa_arr[bfs_i] = cell.split_axis
            sv_arr[bfs_i] = cell.split_val
            queue.append((cell.left,  bfs_left(bfs_i)))
            queue.append((cell.right, bfs_right(bfs_i)))

    # DFS pass: assign rates using mesh.leaves() order so the rate ordering
    # matches FEM matrix assembly (which also iterates leaves in DFS order).
    leaves_dfs = mesh.leaves()
    if len(leaves_dfs) != len(rates):
        raise ValueError(
            f"rates has length {len(rates)} but mesh has {len(leaves_dfs)} leaves."
        )
    for leaf, rate in zip(leaves_dfs, rates):
        rt_arr[cell_to_bfs[id(leaf)]] = rate

    return TreeState(
        split_axis=jnp.array(sa_arr),
        split_val =jnp.array(sv_arr,  dtype=dtype),
        rates     =jnp.array(rt_arr,  dtype=dtype),
    )


# ── Unwrap: TreeState → AdaptiveKDMesh ───────────────────────────────────────

def unwrap_to_mesh(
    tree_state: TreeState,
    data: np.ndarray,
    bounds: list[tuple[float, float]],
) -> tuple[AdaptiveKDMesh, np.ndarray]:
    """
    Reconstruct an AdaptiveKDMesh from a TreeState.

    Data point assignments (Cell.data_idx) are recomputed by filtering the
    provided ``data`` array against each cell's bounding box, so the original
    observed point process must be provided.  The resulting node positions
    (centroids) adapt to whichever data points fall in each leaf.

    Parameters
    ----------
    tree_state : TreeState to decode.
    data       : (n, d) array of observed point process locations.
    bounds     : [(lo_0, hi_0), ..., (lo_{d-1}, hi_{d-1})].

    Returns
    -------
    mesh       : Fully initialised AdaptiveKDMesh (triangulation rebuilt).
    leaf_rates : (n_leaves,) float array of leaf log-intensities in DFS order
                 (matching the order of mesh.leaves() and FEM assembly).
    """
    sa   = np.array(tree_state.split_axis)
    sv   = np.array(tree_state.split_val)
    rt   = np.array(tree_state.rates)
    data = np.asarray(data, dtype=float)
    mn   = tree_state.max_nodes

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    all_idx = np.arange(len(data))

    # Build Cell tree via BFS, partitioning data at each stored split.
    root = Cell(lo=lo.copy(), hi=hi.copy(), data_idx=all_idx)
    cell_to_bfs: dict[int, int] = {}
    queue: deque[tuple[Cell, int]] = deque([(root, 0)])
    while queue:
        cell, bfs_i = queue.popleft()
        cell_to_bfs[id(cell)] = bfs_i

        if bfs_i >= mn:
            continue

        axis = int(sa[bfs_i])
        val  = float(sv[bfs_i])

        if axis >= 0 and not np.isnan(val):
            # Internal node — split and recurse.
            left_hi          = cell.hi.copy();  left_hi[axis]  = val
            right_lo         = cell.lo.copy();  right_lo[axis] = val

            if len(cell.data_idx) > 0:
                left_mask  = data[cell.data_idx, axis] <= val
                left_idx   = cell.data_idx[left_mask]
                right_idx  = cell.data_idx[~left_mask]
            else:
                left_idx = right_idx = np.empty(0, dtype=int)

            cell.split_axis  = axis
            cell.split_val   = val
            cell.data_idx    = np.empty(0, dtype=int)  # internal nodes hold no data

            cell.left  = Cell(lo=cell.lo.copy(), hi=left_hi,         data_idx=left_idx,  parent=cell)
            cell.right = Cell(lo=right_lo,        hi=cell.hi.copy(), data_idx=right_idx, parent=cell)

            queue.append((cell.left,  bfs_left(bfs_i)))
            queue.append((cell.right, bfs_right(bfs_i)))
        # else: leaf — nothing to do; cell.is_leaf is True by default.

    # Build the mesh object without re-running __init__.
    mesh = object.__new__(AdaptiveKDMesh)
    mesh.points = data
    mesh.bounds = bounds
    mesh.lo     = lo
    mesh.hi     = hi
    mesh.root   = root
    mesh._rebuild_triangulation()

    # Collect leaf rates in DFS order (matching mesh.leaves() / FEM assembly).
    leaves_dfs = mesh.leaves()
    leaf_rates = np.array([rt[cell_to_bfs[id(lf)]] for lf in leaves_dfs])

    return mesh, leaf_rates


# ── RJ move helpers (Python-level, for the outer RJMCMC loop) ─────────────────

def splittable_leaves(tree_state: TreeState) -> np.ndarray:
    """
    BFS indices of leaf nodes whose children would fit within max_nodes.
    These are the candidates for a birth move.
    """
    mn  = tree_state.max_nodes
    idx = tree_state.leaf_bfs_indices()
    return idx[2 * idx + 2 < mn]   # bfs_right(i) = 2i+2, vectorised over array

def mergeable_internals(tree_state: TreeState) -> np.ndarray:
    """
    BFS indices of internal nodes whose both children are leaves.
    These are the candidates for a death move.
    """
    sa   = np.array(tree_state.split_axis)
    rt   = np.array(tree_state.rates)
    mn   = tree_state.max_nodes
    candidates = []
    for i in np.where(sa >= 0)[0]:         # all internal nodes
        li, ri = bfs_left(int(i)), bfs_right(int(i))
        if ri < mn and sa[li] < 0 and np.isfinite(rt[li]) \
                    and sa[ri] < 0 and np.isfinite(rt[ri]):
            candidates.append(int(i))
    return np.array(candidates, dtype=int)


def proposed_split(
    tree_state: TreeState,
    leaf_i: int,
    data: np.ndarray,
    bounds: list[tuple[float, float]],
) -> tuple[int, float]:
    """
    Compute the data-driven split (longest axis, data median) for a given leaf.
    Returns (split_axis, split_val) — the same rule as AdaptiveKDMesh._split_cell.

    Call this to determine the split proposal for a birth move; the RJMCMC
    Jacobian is then 1 (deterministic split → no Jacobian correction needed for
    the split geometry, only for the rate dimension change).
    """
    # Reconstruct the leaf's bounding box by walking from the root.
    sa = np.array(tree_state.split_axis)
    sv = np.array(tree_state.split_val)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # Walk from root to leaf_i via parent chain.
    path = []
    i = leaf_i
    while i > 0:
        p = bfs_parent(i)
        path.append((p, i))
        i = p
    for parent_i, child_i in reversed(path):
        axis = int(sa[parent_i])
        val  = float(sv[parent_i])
        if child_i == bfs_left(parent_i):  # we went left
            hi = hi.copy(); hi[axis] = val
        else:                               # we went right
            lo = lo.copy(); lo[axis] = val

    # Points in this cell
    in_cell = np.ones(len(data), dtype=bool)
    for dim in range(data.shape[1]):
        in_cell &= (data[:, dim] >= lo[dim]) & (data[:, dim] <= hi[dim])
    pts_in = data[in_cell]

    # Longest-axis median split (same rule as AdaptiveKDMesh._split_cell)
    axis = int(np.argmax(hi - lo))
    if len(pts_in) >= 2:
        val = float(np.median(pts_in[:, axis]))
        eps = (hi[axis] - lo[axis]) * 1e-8
        val = float(np.clip(val, lo[axis] + eps, hi[axis] - eps))
    else:
        val = float(0.5 * (lo[axis] + hi[axis]))

    return axis, val
