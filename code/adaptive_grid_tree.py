"""
Adaptive hierarchical hat-function basis on [0, 1]^D (or any rectangular domain).

Basis functions
---------------
A D-dimensional basis function is the product of D one-dimensional hat functions:

    phi(x) = prod_{d=0}^{D-1}  hat(x_d ; lo_d, hi_d)

where the 1-D hat on [lo, hi] (width w = hi - lo, peak at midpoint m = lo + w/2) is:

    hat(x ; lo, hi) = { 0                   if  x  < lo  or  x > hi
                      { 2 (x  - lo) / w      if  lo <= x <= m
                      { 2 (hi -  x) / w      if  m  <= x <= hi

The function is piecewise linear, equals 1 at the midpoint, and is 0 at lo and hi.

Tree structure
--------------
Nodes are arranged in a rooted binary tree.

Root: covers the full domain, intervals = bounds.

A node with intervals [(lo_0,hi_0), ..., (lo_{D-1},hi_{D-1})] split in dimension d
produces two children sharing all intervals except dimension d:
    left  child: [lo_d,  mid_d] in dim d   (lower half, split_side = 0)
    right child: [mid_d, hi_d ] in dim d   (upper half, split_side = 1)

where mid_d = (lo_d + hi_d) / 2.  Splits are always at midpoints — no split_val
needs to be stored.

Active basis = leaf nodes.  Refine (birth) a leaf by splitting it in a chosen
dimension; coarsen (death) an internal node whose two children are both leaves.

BFS serialisation (TreeState)
------------------------------
For use in jax.lax.scan and with JAX-based HMC, the tree is serialised in BFS
heap order (node i → children 2i+1 and 2i+2) into two fixed-shape arrays:

    split_dim  (max_nodes,) int32   split dimension for internal nodes;  -1 elsewhere
    rates      (max_nodes,) float64 field coefficient at leaf nodes;     NaN elsewhere

    max_nodes = 2^(max_depth+1) - 1

Because splits are always at midpoints, the intervals at any BFS node can be
recomputed from the root bounds and the split_dim sequence along its path.
compute_node_intervals() builds a (max_nodes, D, 2) lookup table for the current
tree, which can then be passed to the JAX evaluation functions below.

JAX evaluation
--------------
eval_basis_jax(x, node_intervals, leaf_mask)
    -> (max_nodes,) float — active hat values at point x (zero for non-leaves)

design_matrix_jax(X, node_intervals, leaf_mask)
    -> (n_pts, max_nodes) float — design matrix Phi

To evaluate log λ(x) = rates · phi(x):

    phi = eval_basis_jax(x, node_intervals, leaf_mask)
    log_lambda = jnp.dot(jnp.where(leaf_mask, rates, 0.0), phi)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp


# ── BFS index arithmetic ───────────────────────────────────────────────────────

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


# ── 1D hat function ────────────────────────────────────────────────────────────

def hat1d(x: float, lo: float, hi: float) -> float:
    """1D hat function at a single point."""
    w = hi - lo
    m = lo + w / 2.0
    if x < lo or x > hi:
        return 0.0
    return 2.0 * (x - lo) / w if x <= m else 2.0 * (hi - x) / w


def hat1d_np(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Vectorised 1D hat over a NumPy array."""
    w = hi - lo
    m = lo + w / 2.0
    rising  = 2.0 * (x - lo) / w
    falling = 2.0 * (hi - x) / w
    val = np.where(x <= m, rising, falling)
    return np.where((x < lo) | (x > hi), 0.0, val)


# ── BasisNode ─────────────────────────────────────────────────────────────────

@dataclass
class BasisNode:
    """
    A single multilinear hat basis function.

    intervals   (D, 2) float array — [lo, hi] per dimension
    parent      BasisNode or None (None for root)
    split_dim   which dimension of the PARENT was split to create this node
    split_side  0 = left child (lower half), 1 = right child (upper half)
    left        lower-half child after splitting this node; None if leaf
    right       upper-half child after splitting this node; None if leaf
    _own_dim    dimension in which THIS node was split (set by refine)
    """

    intervals:  np.ndarray
    parent:     Optional["BasisNode"] = field(default=None,  repr=False)
    split_dim:  Optional[int]         = None   # dim of parent split → this node
    split_side: Optional[int]         = None   # 0=left, 1=right
    left:       Optional["BasisNode"] = field(default=None,  repr=False)
    right:      Optional["BasisNode"] = field(default=None,  repr=False)
    _own_dim:   Optional[int]         = None   # dim this node was split in

    # ── intrinsic properties ──────────────────────────────────────────────────

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    @property
    def D(self) -> int:
        return self.intervals.shape[0]

    @property
    def center(self) -> np.ndarray:
        """Midpoint per dimension — the location where phi = 1."""
        return (self.intervals[:, 0] + self.intervals[:, 1]) / 2.0

    @property
    def widths(self) -> np.ndarray:
        return self.intervals[:, 1] - self.intervals[:, 0]

    # ── evaluation ───────────────────────────────────────────────────────────

    def eval(self, x: np.ndarray) -> float:
        """Evaluate the multilinear hat at a single point x (D,)."""
        v = 1.0
        for d in range(self.D):
            v *= hat1d(float(x[d]),
                       float(self.intervals[d, 0]),
                       float(self.intervals[d, 1]))
        return v

    def eval_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate at multiple points X (n, D) → (n,) array."""
        v = np.ones(len(X))
        for d in range(self.D):
            v *= hat1d_np(X[:, d],
                          float(self.intervals[d, 0]),
                          float(self.intervals[d, 1]))
        return v

    def __repr__(self) -> str:
        ivs = ", ".join(f"[{lo:.3g},{hi:.3g}]" for lo, hi in self.intervals)
        return f"BasisNode(intervals=[{ivs}], leaf={self.is_leaf})"


# ── AdaptiveGridTree ──────────────────────────────────────────────────────────

class AdaptiveGridTree:
    """
    Adaptive hierarchical basis of multilinear hat functions.

    The active basis is the set of leaf nodes.  Each leaf holds a D-dimensional
    hat function; its support is the product of D intervals, and it peaks at 1
    at the product of D midpoints.

    Parameters
    ----------
    D         : spatial dimension.
    bounds    : list of (lo, hi) per dimension; defaults to [(0, 1)] * D.
    max_depth : maximum tree depth (root = depth 0).  A node at max_depth
                cannot be refined further.
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
        root_intervals = np.array([[lo, hi] for lo, hi in self.bounds], dtype=float)
        self.root = BasisNode(intervals=root_intervals)

    # ── tree traversal ────────────────────────────────────────────────────────

    def leaves(self) -> list[BasisNode]:
        """All leaf nodes in DFS (left-first pre-order) order."""
        out: list[BasisNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                out.append(node)
            else:
                stack.append(node.right)
                stack.append(node.left)
        return out

    def all_nodes(self) -> list[BasisNode]:
        """All nodes (internal + leaves) in DFS order."""
        out: list[BasisNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            out.append(node)
            if not node.is_leaf:
                stack.append(node.right)
                stack.append(node.left)
        return out

    def n_leaves(self) -> int:
        return len(self.leaves())

    def depth(self, node: BasisNode) -> int:
        d, n = 0, node
        while n.parent is not None:
            d += 1
            n = n.parent
        return d

    # ── evaluation (NumPy) ────────────────────────────────────────────────────

    def eval_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate all active basis functions at a single point x (D,).
        Returns (n_leaves,) in DFS leaf order.
        """
        return np.array([node.eval(x) for node in self.leaves()])

    def design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Design matrix Phi where Phi[i, j] = phi_j(X[i]).
        Returns (n_pts, n_leaves).
        """
        lv = self.leaves()
        out = np.zeros((len(X), len(lv)))
        for j, node in enumerate(lv):
            out[:, j] = node.eval_batch(X)
        return out

    # ── refinement ───────────────────────────────────────────────────────────

    def refine(
        self,
        node: BasisNode,
        dim: int,
    ) -> tuple[BasisNode, BasisNode]:
        """
        Split a leaf node in dimension `dim` at the midpoint of its `dim`-th
        interval.  Returns (left_child, right_child).

        Left child  takes the lower half: [..., (lo_d, mid_d), ...].
        Right child takes the upper half: [..., (mid_d, hi_d), ...].
        All other D-1 intervals are inherited unchanged.
        """
        if not node.is_leaf:
            raise ValueError(f"Node {node!r} is not a leaf.")
        if dim < 0 or dim >= self.D:
            raise ValueError(f"dim={dim} out of range [0, {self.D}).")
        if self.depth(node) >= self.max_depth:
            raise ValueError(
                f"Cannot refine: node is already at max_depth={self.max_depth}."
            )

        lo_d = float(node.intervals[dim, 0])
        hi_d = float(node.intervals[dim, 1])
        mid  = (lo_d + hi_d) / 2.0

        left_iv         = node.intervals.copy();  left_iv[dim, 1]  = mid
        right_iv        = node.intervals.copy();  right_iv[dim, 0] = mid

        left  = BasisNode(intervals=left_iv,  parent=node, split_dim=dim, split_side=0)
        right = BasisNode(intervals=right_iv, parent=node, split_dim=dim, split_side=1)

        node.left     = left
        node.right    = right
        node._own_dim = dim

        return left, right

    def coarsen(self, node: BasisNode) -> None:
        """
        Merge a node's two leaf children back into a single leaf.
        Both node.left and node.right must be leaves.
        """
        if node.is_leaf:
            raise ValueError("Node has no children to merge.")
        if not (node.left.is_leaf and node.right.is_leaf):
            raise ValueError("Both children must be leaves to merge.")

        node.left = node.right = None
        node._own_dim = None

    # ── RJMCMC helpers ────────────────────────────────────────────────────────

    def splittable_leaves(self) -> list[BasisNode]:
        """Leaf nodes that can be split (depth < max_depth)."""
        return [n for n in self.leaves() if self.depth(n) < self.max_depth]

    def mergeable_nodes(self) -> list[BasisNode]:
        """Internal nodes whose two children are both leaves."""
        return [
            n for n in self.all_nodes()
            if not n.is_leaf and n.left.is_leaf and n.right.is_leaf
        ]


# ── TreeState (BFS serialisation) ─────────────────────────────────────────────

class TreeState:
    """
    Fixed-shape BFS tree state for use in jax.lax.scan.

    split_dim  (max_nodes,) int32   — split dim for internal nodes;  -1 elsewhere
    rates      (max_nodes,) float   — field coefficient at leaves;   NaN elsewhere

    Both arrays have length  max_nodes = 2^(max_depth+1) - 1.

    Node classification at BFS index i:
        internal  :  split_dim[i] >= 0
        leaf      :  split_dim[i] <  0  AND  isfinite(rates[i])
        absent    :  split_dim[i] <  0  AND  isnan(rates[i])

    Because splits are always at midpoints, the intervals at any BFS node are
    fully determined by the root bounds and the split_dim values of its ancestors
    — no split coordinate needs to be stored.
    """

    __slots__ = ("split_dim", "rates")

    def __init__(self, split_dim: jnp.ndarray, rates: jnp.ndarray):
        self.split_dim = split_dim  # (max_nodes,) int32
        self.rates     = rates      # (max_nodes,) float

    @property
    def max_nodes(self) -> int:
        return int(self.split_dim.shape[0])

    @property
    def max_depth(self) -> int:
        return int(np.floor(np.log2(self.max_nodes + 1))) - 1

    # ── masks ─────────────────────────────────────────────────────────────────

    def leaf_mask(self) -> jnp.ndarray:
        """(max_nodes,) bool: True for leaf nodes."""
        return (self.split_dim < 0) & jnp.isfinite(self.rates)

    def internal_mask(self) -> jnp.ndarray:
        return self.split_dim >= 0

    def occupied_mask(self) -> jnp.ndarray:
        return (self.split_dim >= 0) | jnp.isfinite(self.rates)

    # ── Python-level counts ───────────────────────────────────────────────────

    def n_leaves(self) -> int:
        return int(self.leaf_mask().sum())

    def n_nodes(self) -> int:
        return int(self.occupied_mask().sum())

    def leaf_bfs_indices(self) -> np.ndarray:
        """BFS indices of all leaves, in BFS (breadth-first) order."""
        return np.where(np.array(self.leaf_mask()))[0]

    # ── functional updates (return new TreeState) ─────────────────────────────

    def birth_update(
        self,
        leaf_i: int,
        split_dim: int,
        rate_left: float,
        rate_right: float,
    ) -> "TreeState":
        """
        Split leaf at BFS index `leaf_i` in dimension `split_dim`.
        Left child (lower half) gets rate_left; right child gets rate_right.
        """
        li = bfs_left(leaf_i)
        ri = bfs_right(leaf_i)
        if ri >= self.max_nodes:
            raise ValueError(
                f"Birth at BFS index {leaf_i} would place children at {ri} "
                f">= max_nodes={self.max_nodes}.  Increase max_depth."
            )
        sd = self.split_dim.at[leaf_i].set(split_dim)
        r  = (
            self.rates
            .at[leaf_i].set(jnp.nan)
            .at[li].set(rate_left)
            .at[ri].set(rate_right)
        )
        return TreeState(sd, r)

    def death_update(
        self,
        internal_i: int,
        merged_rate: float,
    ) -> "TreeState":
        """
        Merge the two leaf children of `internal_i` back into a single leaf.
        Both bfs_left(internal_i) and bfs_right(internal_i) must be leaves.
        """
        li = bfs_left(internal_i)
        ri = bfs_right(internal_i)
        sd = self.split_dim.at[internal_i].set(-1)
        r  = (
            self.rates
            .at[internal_i].set(merged_rate)
            .at[li].set(jnp.nan)
            .at[ri].set(jnp.nan)
        )
        return TreeState(sd, r)

    def update_rates(self, new_rates: jnp.ndarray) -> "TreeState":
        """Replace the rates array (used after an HMC within-model step)."""
        return TreeState(self.split_dim, new_rates)

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
        split_dim=jnp.full(mn, -1,      dtype=jnp.int32),
        rates    =jnp.full(mn, jnp.nan, dtype=dtype),
    )

def root_only_state(rate: float, max_depth: int, dtype=jnp.float64) -> TreeState:
    """Single root leaf with the given coefficient."""
    s = empty_tree_state(max_depth, dtype)
    return TreeState(s.split_dim, s.rates.at[0].set(rate))


# ── Wrap: AdaptiveGridTree → TreeState ───────────────────────────────────────

def wrap_to_tree_state(
    tree: AdaptiveGridTree,
    rates: np.ndarray,
    max_depth: int,
    dtype=jnp.float64,
) -> TreeState:
    """
    Serialise an AdaptiveGridTree together with per-leaf coefficients into a
    fixed-shape TreeState.

    Parameters
    ----------
    tree      : AdaptiveGridTree whose structure to encode.
    rates     : (n_leaves,) array of field coefficients in DFS leaf order
                (matching tree.leaves()).
    max_depth : depth of the backing arrays; must be >= tree depth.

    Returns
    -------
    TreeState with split_dim and rates populated.
    """
    mn   = max_nodes_for_depth(max_depth)
    sd   = -np.ones(mn, dtype=np.int32)
    rt   = np.full(mn, np.nan)
    rates = np.asarray(rates, dtype=float)

    # BFS pass: assign every node a BFS index; record split_dim for internals.
    cell_to_bfs: dict[int, int] = {}
    queue: deque[tuple[BasisNode, int]] = deque([(tree.root, 0)])
    while queue:
        node, bfs_i = queue.popleft()
        if bfs_i >= mn:
            raise ValueError(
                f"Tree exceeds max_depth={max_depth}; node at BFS {bfs_i} >= {mn}."
            )
        cell_to_bfs[id(node)] = bfs_i
        if not node.is_leaf:
            sd[bfs_i] = node._own_dim
            queue.append((node.left,  bfs_left(bfs_i)))
            queue.append((node.right, bfs_right(bfs_i)))

    # DFS pass: assign rates in leaves() order (matches FEM / likelihood assembly).
    leaves_dfs = tree.leaves()
    if len(leaves_dfs) != len(rates):
        raise ValueError(
            f"rates length {len(rates)} != tree leaves {len(leaves_dfs)}."
        )
    for leaf, rate in zip(leaves_dfs, rates):
        rt[cell_to_bfs[id(leaf)]] = rate

    return TreeState(
        split_dim=jnp.array(sd),
        rates    =jnp.array(rt, dtype=dtype),
    )


# ── Unwrap: TreeState → AdaptiveGridTree ─────────────────────────────────────

def unwrap_to_tree(
    tree_state: TreeState,
    D: int,
    bounds: list[tuple[float, float]] | None = None,
    max_depth: int | None = None,
) -> tuple[AdaptiveGridTree, np.ndarray]:
    """
    Reconstruct an AdaptiveGridTree from a TreeState.

    Since splits are at midpoints, no extra data is needed — the tree geometry
    is fully determined by split_dim and the root bounds.

    Parameters
    ----------
    tree_state : TreeState to decode.
    D          : spatial dimension.
    bounds     : [(lo_d, hi_d), ...]; defaults to [(0, 1)] * D.
    max_depth  : passed to AdaptiveGridTree; defaults to tree_state.max_depth.

    Returns
    -------
    tree       : AdaptiveGridTree with the same structure as was serialised.
    leaf_rates : (n_leaves,) float array in DFS leaf order (matching tree.leaves()).
    """
    if bounds is None:
        bounds = [(0.0, 1.0)] * D
    if max_depth is None:
        max_depth = tree_state.max_depth

    sd   = np.array(tree_state.split_dim)
    rt   = np.array(tree_state.rates)
    mn   = tree_state.max_nodes
    root_iv = np.array([[lo, hi] for lo, hi in bounds], dtype=float)

    tree = AdaptiveGridTree(D, bounds, max_depth)
    tree.root = BasisNode(intervals=root_iv)

    cell_to_bfs: dict[int, int] = {}
    queue: deque[tuple[BasisNode, int, np.ndarray]] = deque(
        [(tree.root, 0, root_iv)])

    while queue:
        node, bfs_i, iv = queue.popleft()
        cell_to_bfs[id(node)] = bfs_i
        node.intervals = iv

        if bfs_i >= mn:
            continue

        dim = int(sd[bfs_i])
        if dim >= 0:
            mid = (iv[dim, 0] + iv[dim, 1]) / 2.0
            left_iv         = iv.copy();  left_iv[dim, 1]  = mid
            right_iv        = iv.copy();  right_iv[dim, 0] = mid

            left  = BasisNode(intervals=left_iv,  parent=node, split_dim=dim, split_side=0)
            right = BasisNode(intervals=right_iv, parent=node, split_dim=dim, split_side=1)

            node.left     = left
            node.right    = right
            node._own_dim = dim

            queue.append((left,  bfs_left(bfs_i),  left_iv))
            queue.append((right, bfs_right(bfs_i), right_iv))

    # Collect leaf rates in DFS order (matches tree.leaves()).
    leaves_dfs = tree.leaves()
    leaf_rates = np.array([rt[cell_to_bfs[id(lf)]] for lf in leaves_dfs])

    return tree, leaf_rates


# ── Interval lookup table ────────────────────────────────────────────────────

def compute_node_intervals(
    tree_state: TreeState,
    D: int,
    bounds: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Build a (max_nodes, D, 2) NumPy array of node intervals for the current tree.

    Absent BFS slots have NaN intervals.  This table is precomputed once after
    each RJ move and passed to the JAX evaluation functions below so that their
    shape is static across HMC steps.

    Parameters
    ----------
    tree_state : current tree state
    D          : spatial dimension
    bounds     : root intervals; defaults to [(0, 1)] * D

    Returns
    -------
    node_intervals : (max_nodes, D, 2) float array
    """
    if bounds is None:
        bounds = [(0.0, 1.0)] * D

    sd   = np.array(tree_state.split_dim)
    mn   = tree_state.max_nodes
    ivs  = np.full((mn, D, 2), np.nan)

    root_iv = np.array([[lo, hi] for lo, hi in bounds], dtype=float)
    queue: deque[tuple[int, np.ndarray]] = deque([(0, root_iv)])

    while queue:
        bfs_i, iv = queue.popleft()
        if bfs_i >= mn:
            continue
        ivs[bfs_i] = iv
        dim = int(sd[bfs_i])
        if dim >= 0:
            mid = (iv[dim, 0] + iv[dim, 1]) / 2.0
            l_iv = iv.copy();  l_iv[dim, 1] = mid
            r_iv = iv.copy();  r_iv[dim, 0] = mid
            queue.append((bfs_left(bfs_i),  l_iv))
            queue.append((bfs_right(bfs_i), r_iv))

    return ivs


# ── JAX evaluation ────────────────────────────────────────────────────────────

@jax.jit
def eval_basis_jax(
    x: jnp.ndarray,
    node_intervals: jnp.ndarray,
    leaf_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Evaluate all active (leaf) basis functions at point x.

    Parameters
    ----------
    x              : (D,) evaluation point
    node_intervals : (max_nodes, D, 2) precomputed interval table
    leaf_mask      : (max_nodes,) bool — True for leaf nodes

    Returns
    -------
    phi : (max_nodes,) float — phi_i(x) for active leaves, 0 for non-leaves.
          Use  jnp.dot(jnp.where(leaf_mask, rates, 0.0), phi)  to get
          the log-intensity at x.
    """
    def eval_one_node(iv):
        lo    = iv[:, 0]          # (D,)
        hi    = iv[:, 1]
        w     = hi - lo
        mid   = lo + w / 2.0
        rising  = 2.0 * (x - lo) / w
        falling = 2.0 * (hi - x) / w
        val   = jnp.where(x <= mid, rising, falling)
        val   = jnp.where((x < lo) | (x > hi), 0.0, val)
        return jnp.prod(val)

    all_evals = jax.vmap(eval_one_node)(node_intervals)   # (max_nodes,)
    return jnp.where(leaf_mask, all_evals, 0.0)


@jax.jit
def design_matrix_jax(
    X: jnp.ndarray,
    node_intervals: jnp.ndarray,
    leaf_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Design matrix Phi[i, j] = phi_j(X[i]).

    Parameters
    ----------
    X              : (n_pts, D)
    node_intervals : (max_nodes, D, 2)
    leaf_mask      : (max_nodes,) bool

    Returns
    -------
    Phi : (n_pts, max_nodes) float
    """
    return jax.vmap(eval_basis_jax, in_axes=(0, None, None))(
        X, node_intervals, leaf_mask
    )


# ── RJMCMC helpers (Python-level) ────────────────────────────────────────────

def splittable_leaves(tree_state: TreeState) -> np.ndarray:
    """
    BFS indices of leaf nodes whose children would fit within max_nodes.
    These are the birth-move candidates.
    """
    mn  = tree_state.max_nodes
    idx = tree_state.leaf_bfs_indices()
    return idx[2 * idx + 2 < mn]        # bfs_right(i) = 2i+2


def mergeable_internals(tree_state: TreeState) -> np.ndarray:
    """
    BFS indices of internal nodes whose both children are leaves.
    These are the death-move candidates.
    """
    sd  = np.array(tree_state.split_dim)
    rt  = np.array(tree_state.rates)
    mn  = tree_state.max_nodes
    out = []
    for i in np.where(sd >= 0)[0]:
        li, ri = bfs_left(int(i)), bfs_right(int(i))
        if (ri < mn
                and sd[li] < 0 and np.isfinite(rt[li])
                and sd[ri] < 0 and np.isfinite(rt[ri])):
            out.append(int(i))
    return np.array(out, dtype=int)
