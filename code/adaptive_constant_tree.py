"""
Adaptive piecewise-constant tree basis on [0, 1]^D.

This is the leaf-only sibling of the hat-tree basis in ``adaptive_grid_tree.py``
(draft: ``sec:const-tree``).  The domain is refined by the same binary kd-tree,
but the basis functions are *indicators of the leaf cells* rather than products
of constant/linear/hat functions.

Tree and basis
--------------
The domain is recursively split: a leaf cell (an axis-aligned box) is cut at the
midpoint of one chosen axis into two children.  At any stage the leaves
partition the domain into disjoint dyadic boxes

    C_i = ∏_d [lo_d^(i), hi_d^(i)],    ⋃_i C_i = domain,    C_i ∩ C_j = ∅.

The active basis is the *leaves only*:

    φ_i(x) = 1[x ∈ C_i],    i a leaf.

Because the leaves tile the domain (partition of unity, ∑_i φ_i ≡ 1) this is a
universal function approximator without needing any ancestor functions — unlike
leaf hats, which vanish on cell boundaries.  Every birth is a binary midpoint
split along one dimension (no CONST/LINEAR/HAT distinction), so — in contrast to
``adaptive_grid_tree.TreeState`` — no ``split_type`` field is needed.

FEM matrices
------------
* Mass matrix is **diagonal**: C_ij = δ_ij |C_i|, the cell volumes (lumped ==
  consistent).  See :meth:`AdaptiveConstantTree.mass_diagonal`.
* The conforming H^1 stiffness ∫∇φ_i·∇φ_j does **not exist** (indicators are not
  in H^1).  We instead use the **finite-volume / GMRF** operator: a weighted
  graph Laplacian over face-adjacent leaves,

      G^FV_ij = -w_ij  (i~j),   w_ij = |F_ij| / ‖x̄_i - x̄_j‖,   G^FV_ii = ∑_k w_ik,

  with |F_ij| the shared-face area and x̄ the cell centroid.  This is sparse
  (only face-adjacent leaves couple) and has no dense ancestor rows.  See
  :meth:`AdaptiveConstantTree.fv_stiffness`.
* Precision Q = κ² C + G^FV is a Besag-type GMRF
  (:meth:`AdaptiveConstantTree.precision`).

BFS serialisation (ConstTreeState)
-----------------------------------
Fixed-shape arrays of length max_nodes = 2^(max_depth+1) - 1, for JAX:

    split_dim  (max_nodes,) int32  split dim for internal nodes; -1 elsewhere
    rates      (max_nodes,) float  coefficient at each leaf; NaN for internal/absent

Node classification (note: only *leaves* carry coefficients here):
    internal : split_dim[i] >= 0
    leaf     : split_dim[i] <  0  AND  isfinite(rates[i])
    absent   : split_dim[i] <  0  AND  isnan(rates[i])

JAX evaluation
--------------
``compute_node_boxes`` builds (max_nodes, D) lo/hi arrays; ``eval_basis_jax`` /
``design_matrix_jax`` then return the one-hot leaf indicator(s) at a point.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp


# ── BFS index arithmetic ──────────────────────────────────────────────────────

def bfs_left(i: int) -> int:
    return 2 * i + 1

def bfs_right(i: int) -> int:
    return 2 * i + 2

def bfs_parent(i: int) -> int | None:
    return (i - 1) // 2 if i > 0 else None

def bfs_depth(i: int) -> int:
    return int(np.floor(np.log2(i + 1))) if i > 0 else 0

def max_nodes_for_depth(max_depth: int) -> int:
    return 2 ** (max_depth + 1) - 1


# ── CellNode ──────────────────────────────────────────────────────────────────

@dataclass
class CellNode:
    """
    One node of the kd-tree: an axis-aligned box ∏_d [lo_d, hi_d].

    A leaf's basis function is the indicator of its box.  An internal node is
    split at the midpoint of ``split_dim`` into ``left`` ([lo, mid]) and
    ``right`` ((mid, hi]).

    lo, hi      (D,) float box corners
    parent      parent node (None for root)
    split_dim   dimension this node was split in (None if leaf)
    left, right children after a split (None if leaf)
    """
    lo:        np.ndarray
    hi:        np.ndarray
    parent:    Optional["CellNode"] = field(default=None, repr=False)
    split_dim: Optional[int]        = None
    left:      Optional["CellNode"] = field(default=None, repr=False)
    right:     Optional["CellNode"] = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    @property
    def D(self) -> int:
        return len(self.lo)

    @property
    def volume(self) -> float:
        return float(np.prod(self.hi - self.lo))

    @property
    def centroid(self) -> np.ndarray:
        return 0.5 * (self.lo + self.hi)

    def contains(self, x: np.ndarray) -> bool:
        """True if x lies in this box (closed)."""
        return bool(np.all(x >= self.lo) and np.all(x <= self.hi))

    def eval_batch(self, X: np.ndarray) -> np.ndarray:
        """Indicator 1[X[i] ∈ box] for each row (half-open per dim; see module
        docstring).  Used only for visualisation — descent is exact."""
        ge = X >= self.lo
        lt = X <  self.hi
        return np.all(ge & lt, axis=1).astype(float)

    def __repr__(self) -> str:
        box = ", ".join(f"[{l:.3g},{h:.3g}]" for l, h in zip(self.lo, self.hi))
        return f"CellNode({box}, leaf={self.is_leaf})"


# ── AdaptiveConstantTree ──────────────────────────────────────────────────────

class AdaptiveConstantTree:
    """
    Adaptive piecewise-constant (leaf-indicator) basis on a box domain.

    The active basis is the set of LEAF cells.  Each birth is a binary midpoint
    split of a leaf along one axis.

    Parameters
    ----------
    D         : spatial dimension
    bounds    : list of (lo, hi) per dimension; defaults to [(0, 1)] * D
    max_depth : maximum tree depth (root = depth 0)
    """

    def __init__(
        self,
        D: int,
        bounds: list[tuple[float, float]] | None = None,
        max_depth: int = 16,
    ):
        self.D = D
        self.max_depth = max_depth
        self.bounds = bounds if bounds is not None else [(0.0, 1.0)] * D
        lo = np.array([b[0] for b in self.bounds], dtype=float)
        hi = np.array([b[1] for b in self.bounds], dtype=float)
        self.root = CellNode(lo=lo, hi=hi)

    # ── traversal ──────────────────────────────────────────────────────────────

    def all_nodes(self) -> list[CellNode]:
        """All nodes (root + internal + leaves) in DFS left-first pre-order."""
        out: list[CellNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            out.append(node)
            if not node.is_leaf:
                stack.append(node.right)
                stack.append(node.left)
        return out

    def leaves(self) -> list[CellNode]:
        """Leaf cells (the active basis) in DFS left-first pre-order."""
        return [n for n in self.all_nodes() if n.is_leaf]

    def n_leaves(self) -> int:
        return len(self.leaves())

    def depth(self, node: CellNode) -> int:
        d, n = 0, node
        while n.parent is not None:
            d += 1
            n = n.parent
        return d

    # ── refinement ─────────────────────────────────────────────────────────────

    def refine(self, node: CellNode, dim: int) -> tuple[CellNode, CellNode]:
        """Split leaf ``node`` at the midpoint of axis ``dim`` into two children.

        Returns ``(left, right)`` with left = [lo, mid], right = [mid, hi]."""
        if not node.is_leaf:
            raise ValueError(f"Node {node!r} is not a leaf.")
        if dim < 0 or dim >= self.D:
            raise ValueError(f"dim={dim} out of range [0, {self.D}).")
        if self.depth(node) >= self.max_depth:
            raise ValueError(f"Cannot refine: already at max_depth={self.max_depth}.")

        mid = 0.5 * (node.lo[dim] + node.hi[dim])

        llo, lhi = node.lo.copy(), node.hi.copy(); lhi[dim] = mid
        rlo, rhi = node.lo.copy(), node.hi.copy(); rlo[dim] = mid
        left  = CellNode(lo=llo, hi=lhi, parent=node)
        right = CellNode(lo=rlo, hi=rhi, parent=node)
        node.left, node.right, node.split_dim = left, right, dim
        return left, right

    def coarsen(self, node: CellNode) -> None:
        """Merge an internal node's two children back into it (both must be
        leaves)."""
        if node.is_leaf:
            raise ValueError("Node has no children to merge.")
        if not (node.left.is_leaf and node.right.is_leaf):
            raise ValueError("Both children must be leaves to coarsen.")
        node.left = node.right = None
        node.split_dim = None

    # ── RJMCMC helpers ───────────────────────────────────────────────────────

    def splittable_leaves(self) -> list[CellNode]:
        """Leaves that can still be refined (depth < max_depth)."""
        return [n for n in self.leaves() if self.depth(n) < self.max_depth]

    def mergeable_nodes(self) -> list[CellNode]:
        """Internal nodes whose two children are both leaves."""
        return [
            n for n in self.all_nodes()
            if not n.is_leaf and n.left.is_leaf and n.right.is_leaf
        ]

    def canonical_split_dims(self, node: CellNode) -> list[int]:
        """Axes along which splitting ``node`` keeps the tree in *canonical*
        (sorted-split) form — the uniqueness fix of draft ``sec:uniqueness``.

        Along the root→node path the split dimensions must be non-decreasing, so
        the admissible axes are ``d >= last_split_dim_on_path``.  Using only
        these in an RJMCMC proposal makes the tree→partition map injective and
        removes the cross-dimensional double-counting.

        Caveat: restricting to canonical splits also shrinks the family of
        reachable partitions (a branch can never re-split a lower axis after a
        higher one), so this is opt-in rather than the default refinement rule.
        """
        last = -1
        n = node
        while n.parent is not None:
            last = max(last, n.parent.split_dim)
            n = n.parent
        return [d for d in range(self.D) if d >= last]

    # ── evaluation ─────────────────────────────────────────────────────────────

    def leaf_of(self, x: np.ndarray) -> CellNode:
        """Descend to the unique leaf whose cell contains ``x`` (exact; ties at
        a split midpoint go left)."""
        node = self.root
        while not node.is_leaf:
            mid = node.left.hi[node.split_dim]
            node = node.left if x[node.split_dim] <= mid else node.right
        return node

    def design_matrix(self, X: np.ndarray) -> np.ndarray:
        """One-hot design matrix Phi[i, j] = 1[X[i] ∈ leaf_j], columns in
        ``leaves()`` order.  Exact (computed by tree descent)."""
        leaves = self.leaves()
        col_of = {id(leaf): j for j, leaf in enumerate(leaves)}
        Phi = np.zeros((len(X), len(leaves)))
        for i, x in enumerate(np.asarray(X, dtype=float)):
            Phi[i, col_of[id(self.leaf_of(x))]] = 1.0
        return Phi

    # ── FEM matrices ───────────────────────────────────────────────────────────

    def mass_diagonal(self) -> np.ndarray:
        """Diagonal of the mass matrix: leaf-cell volumes, in ``leaves()`` order.

        C = diag(mass) is exact (lumped == consistent for disjoint indicators)."""
        return np.array([leaf.volume for leaf in self.leaves()])

    def face_adjacency(self) -> list[tuple[int, int, float, float]]:
        """Face-adjacency list of the leaf cells.

        Returns tuples ``(i, j, area, dist)`` with i < j (indices into
        ``leaves()``), ``area`` the shared (D-1)-face measure, and ``dist`` the
        centroid distance.  Two leaves are face-adjacent if their boxes touch on
        a hyperplane in exactly one axis and their projections overlap with
        positive measure in every other axis.
        """
        leaves = self.leaves()
        L = len(leaves)
        out: list[tuple[int, int, float, float]] = []
        for i in range(L):
            ci = leaves[i]
            for j in range(i + 1, L):
                cj = leaves[j]
                # overlap interval per axis
                ov_lo = np.maximum(ci.lo, cj.lo)
                ov_hi = np.minimum(ci.hi, cj.hi)
                ov_len = ov_hi - ov_lo
                # touching axis: boxes meet on a plane (ov_len == 0) in exactly
                # one dim, with strictly positive overlap in all others.
                touch = np.isclose(ov_len, 0.0)
                if touch.sum() != 1:
                    continue
                if np.any(ov_len[~touch] <= 0.0):
                    continue
                area = float(np.prod(ov_len[~touch])) if self.D > 1 else 1.0
                dist = float(np.linalg.norm(ci.centroid - cj.centroid))
                out.append((i, j, area, dist))
        return out

    def fv_stiffness(self) -> np.ndarray:
        """Finite-volume / GMRF stiffness G^FV (dense, ``leaves()`` order).

        Weighted graph Laplacian over face-adjacent leaves:
        off-diagonal -w_ij, diagonal +∑_k w_ik, with w = area / centroid-dist.
        Symmetric positive semi-definite (rows sum to zero)."""
        L = self.n_leaves()
        G = np.zeros((L, L))
        for i, j, area, dist in self.face_adjacency():
            w = area / dist
            G[i, j] -= w
            G[j, i] -= w
            G[i, i] += w
            G[j, j] += w
        return G

    def precision(self, kappa: float) -> np.ndarray:
        """SPDE precision Q = κ² C + G^FV (dense, ``leaves()`` order)."""
        Q = self.fv_stiffness()
        Q[np.diag_indices_from(Q)] += float(kappa) ** 2 * self.mass_diagonal()
        return Q


# ── ConstTreeState (BFS serialisation) ────────────────────────────────────────

class ConstTreeState:
    """
    Fixed-shape BFS serialisation of an :class:`AdaptiveConstantTree` for JAX.

    split_dim  (max_nodes,) int32  split dim for internal nodes; -1 elsewhere
    rates      (max_nodes,) float  coefficient at each LEAF; NaN for internal/absent

    Only leaves are active (carry coefficients).
    """

    __slots__ = ("split_dim", "rates")

    def __init__(self, split_dim, rates):
        self.split_dim = split_dim   # (max_nodes,) int32
        self.rates     = rates       # (max_nodes,) float

    @property
    def max_nodes(self) -> int:
        return int(self.split_dim.shape[0])

    @property
    def max_depth(self) -> int:
        return int(np.floor(np.log2(self.max_nodes + 1))) - 1

    # ── masks ────────────────────────────────────────────────────────────────

    def leaf_mask(self) -> jnp.ndarray:
        """(max_nodes,) bool: leaf nodes (active and not split)."""
        return (self.split_dim < 0) & jnp.isfinite(self.rates)

    def internal_mask(self) -> jnp.ndarray:
        return self.split_dim >= 0

    # ── counts / indices ───────────────────────────────────────────────────────

    def n_leaves(self) -> int:
        return int(self.leaf_mask().sum())

    def leaf_bfs_indices(self) -> np.ndarray:
        return np.where(np.array(self.leaf_mask()))[0]

    # ── functional updates ─────────────────────────────────────────────────────

    def birth_update(
        self,
        leaf_i:     int,
        dim:        int,
        rate_left:  float,
        rate_right: float,
    ) -> "ConstTreeState":
        """Split leaf ``leaf_i`` along ``dim``; the parent stops being a leaf
        (its rate is cleared) and the two children become leaves."""
        li, ri = bfs_left(leaf_i), bfs_right(leaf_i)
        if ri >= self.max_nodes:
            raise ValueError(
                f"Birth at BFS {leaf_i} would exceed max_nodes={self.max_nodes}."
            )
        sd = self.split_dim.at[leaf_i].set(dim)
        r  = self.rates.at[leaf_i].set(jnp.nan)   # parent no longer a leaf
        r  = r.at[li].set(rate_left)
        r  = r.at[ri].set(rate_right)
        return ConstTreeState(sd, r)

    def death_update(self, internal_i: int, parent_rate: float) -> "ConstTreeState":
        """Merge the two (leaf) children of ``internal_i`` back into it; the node
        becomes a leaf again with coefficient ``parent_rate``."""
        li, ri = bfs_left(internal_i), bfs_right(internal_i)
        sd = self.split_dim.at[internal_i].set(-1)
        r  = self.rates.at[internal_i].set(parent_rate)
        r  = r.at[li].set(jnp.nan)
        r  = r.at[ri].set(jnp.nan)
        return ConstTreeState(sd, r)

    def update_rates(self, new_rates: jnp.ndarray) -> "ConstTreeState":
        return ConstTreeState(self.split_dim, new_rates)

    def __repr__(self) -> str:
        return (
            f"ConstTreeState(max_depth={self.max_depth}, "
            f"n_leaves={self.n_leaves()})"
        )


# ── Constructors ──────────────────────────────────────────────────────────────

def empty_const_state(max_depth: int, dtype=jnp.float64) -> ConstTreeState:
    mn = max_nodes_for_depth(max_depth)
    return ConstTreeState(
        split_dim = jnp.full(mn, -1,      dtype=jnp.int32),
        rates     = jnp.full(mn, jnp.nan, dtype=dtype),
    )

def root_only_const_state(rate: float, max_depth: int, dtype=jnp.float64) -> ConstTreeState:
    """Single root leaf (indicator of the whole domain) with coefficient ``rate``."""
    s = empty_const_state(max_depth, dtype)
    return ConstTreeState(s.split_dim, s.rates.at[0].set(rate))


# ── Wrap: AdaptiveConstantTree → ConstTreeState ──────────────────────────────

def wrap_to_const_state(
    tree:      AdaptiveConstantTree,
    rates:     np.ndarray,
    max_depth: int,
    dtype=jnp.float64,
) -> ConstTreeState:
    """Serialise a tree plus per-leaf coefficients into a ConstTreeState.

    ``rates`` are in ``tree.leaves()`` (DFS) order.
    """
    mn = max_nodes_for_depth(max_depth)
    sd = -np.ones(mn, dtype=np.int32)
    rt = np.full(mn, np.nan)
    rates = np.asarray(rates, dtype=float)

    leaf_to_bfs: dict[int, int] = {}
    queue: deque[tuple[CellNode, int]] = deque([(tree.root, 0)])
    while queue:
        node, bfs_i = queue.popleft()
        if bfs_i >= mn:
            raise ValueError(f"Tree exceeds max_depth={max_depth}.")
        if node.is_leaf:
            leaf_to_bfs[id(node)] = bfs_i
        else:
            sd[bfs_i] = node.split_dim
            queue.append((node.left,  bfs_left(bfs_i)))
            queue.append((node.right, bfs_right(bfs_i)))

    leaves = tree.leaves()
    if len(leaves) != len(rates):
        raise ValueError(f"rates length {len(rates)} != n_leaves {len(leaves)}.")
    for leaf, rate in zip(leaves, rates):
        rt[leaf_to_bfs[id(leaf)]] = rate

    return ConstTreeState(jnp.array(sd), jnp.array(rt, dtype=dtype))


# ── Unwrap: ConstTreeState → AdaptiveConstantTree ────────────────────────────

def unwrap_to_const_tree(
    tree_state: ConstTreeState,
    D:          int,
    bounds:     list[tuple[float, float]] | None = None,
    max_depth:  int | None = None,
) -> tuple[AdaptiveConstantTree, np.ndarray]:
    """Reconstruct a tree from a ConstTreeState.

    Returns ``(tree, leaf_rates)`` with leaf_rates in ``tree.leaves()`` order.
    """
    if max_depth is None:
        max_depth = tree_state.max_depth

    sd = np.array(tree_state.split_dim)
    rt = np.array(tree_state.rates)
    mn = tree_state.max_nodes

    tree = AdaptiveConstantTree(D, bounds, max_depth)
    leaf_to_bfs: dict[int, int] = {}

    queue: deque[tuple[CellNode, int]] = deque([(tree.root, 0)])
    while queue:
        node, bfs_i = queue.popleft()
        if bfs_i >= mn:
            continue
        dim = int(sd[bfs_i])
        if dim < 0:
            leaf_to_bfs[id(node)] = bfs_i
            continue
        left, right = tree.refine(node, dim)
        queue.append((left,  bfs_left(bfs_i)))
        queue.append((right, bfs_right(bfs_i)))

    leaves = tree.leaves()
    leaf_rates = np.array([rt[leaf_to_bfs[id(n)]] for n in leaves])
    return tree, leaf_rates


# ── Node box lookup table ──────────────────────────────────────────────────────

def compute_node_boxes(
    tree_state: ConstTreeState,
    D:          int,
    bounds:     list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-node cell boxes for the current tree.

    Returns
    -------
    node_lo : (max_nodes, D) float  — box lower corners
    node_hi : (max_nodes, D) float  — box upper corners

    Reachable nodes get their real box; absent BFS slots are left as an empty
    box (lo = hi = 0) so they never match a point in ``eval_basis_jax``.
    """
    bounds = bounds if bounds is not None else [(0.0, 1.0)] * D
    sd = np.array(tree_state.split_dim)
    mn = tree_state.max_nodes

    node_lo = np.zeros((mn, D))
    node_hi = np.zeros((mn, D))

    root_lo = np.array([b[0] for b in bounds], dtype=float)
    root_hi = np.array([b[1] for b in bounds], dtype=float)

    queue: deque[tuple[int, np.ndarray, np.ndarray]] = deque([(0, root_lo, root_hi)])
    while queue:
        bfs_i, lo, hi = queue.popleft()
        if bfs_i >= mn:
            continue
        node_lo[bfs_i] = lo
        node_hi[bfs_i] = hi

        dim = int(sd[bfs_i])
        if dim < 0:
            continue
        mid = 0.5 * (lo[dim] + hi[dim])
        llo, lhi = lo.copy(), hi.copy(); lhi[dim] = mid
        rlo, rhi = lo.copy(), hi.copy(); rlo[dim] = mid
        queue.append((bfs_left(bfs_i),  llo, lhi))
        queue.append((bfs_right(bfs_i), rlo, rhi))

    return node_lo, node_hi


# ── JAX evaluation ──────────────────────────────────────────────────────────────

@jax.jit
def eval_basis_jax(
    x:          jnp.ndarray,
    node_lo:    jnp.ndarray,
    node_hi:    jnp.ndarray,
    leaf_mask:  jnp.ndarray,
    domain_lo:  jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the leaf-indicator basis at point ``x``.

    Returns (max_nodes,) one-hot: 1.0 at the unique active leaf whose cell
    contains ``x``, 0 elsewhere.  Membership is half-open ``(lo, hi]`` per axis
    (ties at a split midpoint go to the left child, matching ``leaf_of``), with
    the global lower boundary included.

    Parameters
    ----------
    x          : (D,) evaluation point
    node_lo    : (max_nodes, D) float
    node_hi    : (max_nodes, D) float
    leaf_mask  : (max_nodes,) bool
    domain_lo  : (D,) float — global lower corner (for the inclusive lower edge)
    """
    above_lo = (x > node_lo) | (node_lo <= domain_lo[None, :])
    below_hi = x <= node_hi
    member   = jnp.all(above_lo & below_hi, axis=1)
    return jnp.where(leaf_mask & member, 1.0, 0.0)


@jax.jit
def design_matrix_jax(
    X:          jnp.ndarray,
    node_lo:    jnp.ndarray,
    node_hi:    jnp.ndarray,
    leaf_mask:  jnp.ndarray,
    domain_lo:  jnp.ndarray,
) -> jnp.ndarray:
    """One-hot design matrix Phi[i, j] = 1[X[i] ∈ leaf_j].  (n_pts, max_nodes)."""
    return jax.vmap(eval_basis_jax, in_axes=(0, None, None, None, None))(
        X, node_lo, node_hi, leaf_mask, domain_lo
    )


# ── RJMCMC helpers (BFS level) ─────────────────────────────────────────────────

def splittable_leaves(tree_state: ConstTreeState) -> np.ndarray:
    """BFS indices of leaves whose children would fit within max_nodes."""
    mn  = tree_state.max_nodes
    idx = tree_state.leaf_bfs_indices()
    return idx[2 * idx + 2 < mn]


def mergeable_internals(tree_state: ConstTreeState) -> np.ndarray:
    """BFS indices of internal nodes whose two children are both leaves."""
    sd = np.array(tree_state.split_dim)
    rt = np.array(tree_state.rates)
    mn = tree_state.max_nodes
    out = []
    for i in np.where(sd >= 0)[0]:
        li, ri = bfs_left(int(i)), bfs_right(int(i))
        if ri >= mn:
            continue
        child_leaf = lambda c: sd[c] < 0 and np.isfinite(rt[c])
        if child_leaf(li) and child_leaf(ri):
            out.append(int(i))
    return np.array(out, dtype=int)
