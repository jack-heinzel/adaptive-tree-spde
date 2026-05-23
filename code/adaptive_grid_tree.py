"""
Adaptive hierarchical basis on [0, 1]^D.

1D hierarchy per dimension
--------------------------
Each dimension follows a fixed hierarchy:

    level 0 — CONST   f(x) = 1
    level 1 — LINEAR  f(x) = x
    level 2 — HAT     f(x) = hat(x; 0, 1)  (unit hat, peaks at 0.5)
    level 3+ — dyadic sub-hats:
               hat(x; lo, hi) has two children at midpoint m = (lo+hi)/2:
                   hat(x; lo, m)   left child
                   hat(x; m,  hi)  right child

The collection {1, x, hat(·;0,1), hat(·;0,½), hat(·;½,1), ...} is the
Schauder basis for C([0,1]).

D-dimensional basis
-------------------
Each tree node k holds a D-tuple of 1D states s_d^(k).  Its basis function is

    φ_k(x) = ∏_d  f_{s_d^(k)}(x_d)

The root has all dims at CONST: φ_root(x) = 1.
The active basis = ALL nodes (root + internal + leaves).

Tree structure
--------------
A birth move advances exactly one dimension by one step:
    CONST  → LINEAR  : 1 new child
    LINEAR → HAT     : 1 new child
    HAT    → sub-hats: 2 new children (left and right)

Each node can be split at most once (in one chosen dimension).
BFS heap: node i → left child 2i+1, right child 2i+2.
For CONST→LINEAR and LINEAR→HAT births, only the left child slot is used.

BFS serialisation (TreeState)
------------------------------
Fixed-shape arrays of length max_nodes = 2^(max_depth+1) - 1:

    split_dim  (max_nodes,) int32   split dim for internal nodes;    -1 elsewhere
    split_type (max_nodes,) int32   0=CONST→LINEAR, 1=LINEAR→HAT,
                                    2=HAT split;                      -1 elsewhere
    rates      (max_nodes,) float   coefficient for each active node; NaN if absent

Node classification:
    internal : split_dim[i] >= 0
    leaf     : split_dim[i] <  0  AND  isfinite(rates[i])
    absent   : split_dim[i] <  0  AND  isnan(rates[i])

Both internal and leaf nodes are active (all have finite rates).

Precomputed state arrays (built by compute_node_states after each RJ move):

    node_kind  (max_nodes, D) int32   per-dim kind: CONST=0, LINEAR=1, HAT=2
    node_lo    (max_nodes, D) float   per-dim lo for HAT; 0 otherwise
    node_hi    (max_nodes, D) float   per-dim hi for HAT; 1 otherwise

JAX evaluation
--------------
eval_basis_jax(x, node_kind, node_lo, node_hi, active_mask)
    -> (max_nodes,) float  — phi_i(x) for active nodes, 0 for absent

design_matrix_jax(X, node_kind, node_lo, node_hi, active_mask)
    -> (n_pts, max_nodes) float

log λ(x) = jnp.dot(jnp.where(active_mask, rates, 0.0), phi)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp


# ── 1D state kinds ────────────────────────────────────────────────────────────

CONST  = 0   # f(x) = 1
LINEAR = 1   # f(x) = x
HAT    = 2   # f(x) = hat(x; lo, hi)


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


# ── 1D hat function (NumPy) ───────────────────────────────────────────────────

def hat1d(x: float, lo: float, hi: float) -> float:
    w = hi - lo
    m = lo + w / 2.0
    if x < lo or x > hi:
        return 0.0
    return 2.0 * (x - lo) / w if x <= m else 2.0 * (hi - x) / w


def hat1d_np(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    w = hi - lo
    m = lo + w / 2.0
    rising  = 2.0 * (x - lo) / w
    falling = 2.0 * (hi - x) / w
    val = np.where(x <= m, rising, falling)
    return np.where((x < lo) | (x > hi), 0.0, val)


# ── State1D ───────────────────────────────────────────────────────────────────

@dataclass
class State1D:
    """One-dimensional component state for a single dimension of a basis node."""
    kind: int          # CONST, LINEAR, or HAT
    lo:   float = 0.0  # support lo (meaningful for HAT only)
    hi:   float = 1.0  # support hi (meaningful for HAT only)

    def eval(self, x: float) -> float:
        if self.kind == CONST:
            return 1.0
        elif self.kind == LINEAR:
            return float(x)
        else:
            return hat1d(x, self.lo, self.hi)

    def eval_np(self, x: np.ndarray) -> np.ndarray:
        if self.kind == CONST:
            return np.ones_like(x, dtype=float)
        elif self.kind == LINEAR:
            return np.asarray(x, dtype=float)
        else:
            return hat1d_np(x, self.lo, self.hi)

    # ── child rules ──────────────────────────────────────────────────────────

    def child_linear(self) -> "State1D":
        assert self.kind == CONST
        return State1D(LINEAR)

    def child_hat(self) -> "State1D":
        assert self.kind == LINEAR
        return State1D(HAT, 0.0, 1.0)

    def child_left(self) -> "State1D":
        assert self.kind == HAT
        m = (self.lo + self.hi) / 2.0
        return State1D(HAT, self.lo, m)

    def child_right(self) -> "State1D":
        assert self.kind == HAT
        m = (self.lo + self.hi) / 2.0
        return State1D(HAT, m, self.hi)

    def __repr__(self) -> str:
        if self.kind == CONST:
            return "CONST"
        elif self.kind == LINEAR:
            return "LINEAR"
        else:
            return f"HAT({self.lo:.3g},{self.hi:.3g})"


# ── BasisNode ─────────────────────────────────────────────────────────────────

@dataclass
class BasisNode:
    """
    A single D-dimensional basis function: product of D 1D states.

    states      D State1D objects, one per dimension
    parent      parent node (None for root)
    split_dim   dimension of the parent birth move that created this node
    split_side  None for CONST→LINEAR / LINEAR→HAT; 0=left, 1=right for HAT splits
    left        left child after this node is refined; None if leaf
    right       right child after HAT split; None for 1-child splits or leaf
    _own_dim    dimension in which THIS node was refined
    _own_type   0=CONST→LINEAR, 1=LINEAR→HAT, 2=HAT split
    """
    states:     list[State1D]
    parent:     Optional["BasisNode"] = field(default=None, repr=False)
    split_dim:  Optional[int]         = None
    split_side: Optional[int]         = None
    left:       Optional["BasisNode"] = field(default=None, repr=False)
    right:      Optional["BasisNode"] = field(default=None, repr=False)
    _own_dim:   Optional[int]         = None
    _own_type:  Optional[int]         = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    @property
    def D(self) -> int:
        return len(self.states)

    def eval(self, x: np.ndarray) -> float:
        v = 1.0
        for d, s in enumerate(self.states):
            v *= s.eval(float(x[d]))
        return v

    def eval_batch(self, X: np.ndarray) -> np.ndarray:
        v = np.ones(len(X))
        for d, s in enumerate(self.states):
            v *= s.eval_np(X[:, d])
        return v

    def __repr__(self) -> str:
        return f"BasisNode(states={self.states}, leaf={self.is_leaf})"


# ── AdaptiveGridTree ──────────────────────────────────────────────────────────

class AdaptiveGridTree:
    """
    Adaptive hierarchical basis of D-dimensional product functions.

    The active basis is ALL nodes (root + internal + leaves).  Each node's
    basis function is a product of D 1D components following the
    CONST → LINEAR → HAT → sub-hat hierarchy.

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
        self.root = BasisNode(states=[State1D(CONST) for _ in range(D)])

    # ── tree traversal ────────────────────────────────────────────────────────

    def all_nodes(self) -> list[BasisNode]:
        """All nodes (root + internal + leaves) in DFS left-first pre-order."""
        out: list[BasisNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            out.append(node)
            if not node.is_leaf:
                if node.right is not None:
                    stack.append(node.right)
                stack.append(node.left)
        return out

    def leaves(self) -> list[BasisNode]:
        """All leaf nodes in DFS left-first pre-order."""
        return [n for n in self.all_nodes() if n.is_leaf]

    def n_nodes(self) -> int:
        return len(self.all_nodes())

    def n_leaves(self) -> int:
        return len(self.leaves())

    def depth(self, node: BasisNode) -> int:
        d, n = 0, node
        while n.parent is not None:
            d += 1
            n = n.parent
        return d

    # ── evaluation ────────────────────────────────────────────────────────────

    def eval_basis(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all active basis functions at x (D,). Returns (n_nodes,)."""
        return np.array([node.eval(x) for node in self.all_nodes()])

    def design_matrix(self, X: np.ndarray) -> np.ndarray:
        """Design matrix Phi[i,j] = phi_j(X[i]). Returns (n_pts, n_nodes)."""
        nodes = self.all_nodes()
        out = np.zeros((len(X), len(nodes)))
        for j, node in enumerate(nodes):
            out[:, j] = node.eval_batch(X)
        return out

    # ── refinement ───────────────────────────────────────────────────────────

    def refine(self, node: BasisNode, dim: int) -> tuple[BasisNode, ...]:
        """
        Advance dimension `dim` by one step.

        Returns (child,) for CONST→LINEAR or LINEAR→HAT;
        returns (left, right) for a HAT split.
        """
        if not node.is_leaf:
            raise ValueError(f"Node {node!r} is not a leaf.")
        if dim < 0 or dim >= self.D:
            raise ValueError(f"dim={dim} out of range [0, {self.D}).")
        if self.depth(node) >= self.max_depth:
            raise ValueError(f"Cannot refine: already at max_depth={self.max_depth}.")

        s = node.states[dim]

        if s.kind == CONST:
            cs = list(node.states); cs[dim] = s.child_linear()
            child = BasisNode(states=cs, parent=node, split_dim=dim, split_side=None)
            node.left = child
            node._own_dim = dim;  node._own_type = 0
            return (child,)

        elif s.kind == LINEAR:
            cs = list(node.states); cs[dim] = s.child_hat()
            child = BasisNode(states=cs, parent=node, split_dim=dim, split_side=None)
            node.left = child
            node._own_dim = dim;  node._own_type = 1
            return (child,)

        else:  # HAT
            ls = list(node.states); ls[dim] = s.child_left()
            rs = list(node.states); rs[dim] = s.child_right()
            left  = BasisNode(states=ls, parent=node, split_dim=dim, split_side=0)
            right = BasisNode(states=rs, parent=node, split_dim=dim, split_side=1)
            node.left = left;  node.right = right
            node._own_dim = dim;  node._own_type = 2
            return (left, right)

    def coarsen(self, node: BasisNode) -> None:
        """Remove all children of node (all children must be leaves)."""
        if node.is_leaf:
            raise ValueError("Node has no children to remove.")
        if not node.left.is_leaf:
            raise ValueError("Left child must be a leaf to coarsen.")
        if node.right is not None and not node.right.is_leaf:
            raise ValueError("Right child must be a leaf to coarsen.")
        node.left = node.right = None
        node._own_dim = node._own_type = None

    # ── RJMCMC helpers ───────────────────────────────────────────────────────

    def splittable_leaves(self) -> list[BasisNode]:
        """Leaves that can be refined (depth < max_depth)."""
        return [n for n in self.leaves() if self.depth(n) < self.max_depth]

    def mergeable_nodes(self) -> list[BasisNode]:
        """Internal nodes whose children are all leaves."""
        result = []
        for n in self.all_nodes():
            if not n.is_leaf:
                left_ok  = n.left is not None and n.left.is_leaf
                right_ok = (n.right is None) or n.right.is_leaf
                if left_ok and right_ok:
                    result.append(n)
        return result


# ── TreeState (BFS serialisation) ────────────────────────────────────────────

class TreeState:
    """
    Fixed-shape BFS serialisation for use with JAX.

    split_dim  (max_nodes,) int32  split dim for internal nodes;    -1 elsewhere
    split_type (max_nodes,) int32  0=CONST→LINEAR, 1=LINEAR→HAT,
                                   2=HAT split;                      -1 elsewhere
    rates      (max_nodes,) float  coefficient for each active node; NaN if absent

    Both internal and leaf nodes have finite rates.
    """

    __slots__ = ("split_dim", "split_type", "rates")

    def __init__(self, split_dim, split_type, rates):
        self.split_dim  = split_dim   # (max_nodes,) int32
        self.split_type = split_type  # (max_nodes,) int32
        self.rates      = rates       # (max_nodes,) float

    @property
    def max_nodes(self) -> int:
        return int(self.split_dim.shape[0])

    @property
    def max_depth(self) -> int:
        return int(np.floor(np.log2(self.max_nodes + 1))) - 1

    # ── masks ─────────────────────────────────────────────────────────────────

    def active_mask(self) -> jnp.ndarray:
        """(max_nodes,) bool: True for all active nodes (leaf or internal)."""
        return jnp.isfinite(self.rates)

    def leaf_mask(self) -> jnp.ndarray:
        """(max_nodes,) bool: True for leaf nodes (active and not split)."""
        return (self.split_dim < 0) & jnp.isfinite(self.rates)

    def internal_mask(self) -> jnp.ndarray:
        return self.split_dim >= 0

    # ── counts ────────────────────────────────────────────────────────────────

    def n_nodes(self) -> int:
        return int(self.active_mask().sum())

    def n_leaves(self) -> int:
        return int(self.leaf_mask().sum())

    def leaf_bfs_indices(self) -> np.ndarray:
        return np.where(np.array(self.leaf_mask()))[0]

    def active_bfs_indices(self) -> np.ndarray:
        return np.where(np.array(self.active_mask()))[0]

    # ── functional updates ────────────────────────────────────────────────────

    def birth_update(
        self,
        leaf_i:     int,
        dim:        int,
        split_type: int,        # 0=CONST→LINEAR, 1=LINEAR→HAT, 2=HAT split
        rate_left:  float,
        rate_right: float = float("nan"),  # only used for split_type == 2
    ) -> "TreeState":
        li = bfs_left(leaf_i)
        ri = bfs_right(leaf_i)
        if ri >= self.max_nodes:
            raise ValueError(
                f"Birth at BFS {leaf_i} would exceed max_nodes={self.max_nodes}."
            )
        sd = self.split_dim.at[leaf_i].set(dim)
        st = self.split_type.at[leaf_i].set(split_type)
        r  = self.rates.at[li].set(rate_left)
        if split_type == 2:
            r = r.at[ri].set(rate_right)
        return TreeState(sd, st, r)

    def death_update(self, internal_i: int) -> "TreeState":
        """Remove children of internal_i (children must be leaves).
        The parent's rate is unchanged."""
        li    = bfs_left(internal_i)
        ri    = bfs_right(internal_i)
        stype = int(self.split_type[internal_i])
        sd = self.split_dim.at[internal_i].set(-1)
        st = self.split_type.at[internal_i].set(-1)
        r  = self.rates.at[li].set(jnp.nan)
        if stype == 2 and ri < self.max_nodes:
            r = r.at[ri].set(jnp.nan)
        return TreeState(sd, st, r)

    def update_rates(self, new_rates: jnp.ndarray) -> "TreeState":
        return TreeState(self.split_dim, self.split_type, new_rates)

    def __repr__(self) -> str:
        return (
            f"TreeState(max_depth={self.max_depth}, "
            f"n_nodes={self.n_nodes()}, n_leaves={self.n_leaves()})"
        )


# ── Constructors ──────────────────────────────────────────────────────────────

def empty_tree_state(max_depth: int, dtype=jnp.float64) -> TreeState:
    mn = max_nodes_for_depth(max_depth)
    return TreeState(
        split_dim  = jnp.full(mn, -1,       dtype=jnp.int32),
        split_type = jnp.full(mn, -1,       dtype=jnp.int32),
        rates      = jnp.full(mn, jnp.nan,  dtype=dtype),
    )

def root_only_state(rate: float, max_depth: int, dtype=jnp.float64) -> TreeState:
    """Single root node (constant function) with the given coefficient."""
    s = empty_tree_state(max_depth, dtype)
    return TreeState(s.split_dim, s.split_type, s.rates.at[0].set(rate))


# ── Wrap: AdaptiveGridTree → TreeState ───────────────────────────────────────

def wrap_to_tree_state(
    tree:      AdaptiveGridTree,
    rates:     np.ndarray,
    max_depth: int,
    dtype=jnp.float64,
) -> TreeState:
    """
    Serialise an AdaptiveGridTree with per-node coefficients into a TreeState.

    Parameters
    ----------
    tree      : tree whose structure to encode
    rates     : (n_nodes,) coefficients in DFS all-nodes order (tree.all_nodes())
    max_depth : depth of the backing arrays; must be >= tree depth

    Returns
    -------
    TreeState with split_dim, split_type, and rates populated.
    """
    mn    = max_nodes_for_depth(max_depth)
    sd    = -np.ones(mn, dtype=np.int32)
    st    = -np.ones(mn, dtype=np.int32)
    rt    = np.full(mn, np.nan)
    rates = np.asarray(rates, dtype=float)

    node_to_bfs: dict[int, int] = {}

    queue: deque[tuple[BasisNode, int]] = deque([(tree.root, 0)])
    while queue:
        node, bfs_i = queue.popleft()
        if bfs_i >= mn:
            raise ValueError(f"Tree exceeds max_depth={max_depth}.")
        node_to_bfs[id(node)] = bfs_i
        if not node.is_leaf:
            sd[bfs_i] = node._own_dim
            st[bfs_i] = node._own_type
            queue.append((node.left, bfs_left(bfs_i)))
            if node.right is not None:
                queue.append((node.right, bfs_right(bfs_i)))

    nodes_dfs = tree.all_nodes()
    if len(nodes_dfs) != len(rates):
        raise ValueError(f"rates length {len(rates)} != n_nodes {len(nodes_dfs)}.")
    for node, rate in zip(nodes_dfs, rates):
        rt[node_to_bfs[id(node)]] = rate

    return TreeState(
        split_dim  = jnp.array(sd),
        split_type = jnp.array(st),
        rates      = jnp.array(rt, dtype=dtype),
    )


# ── Unwrap: TreeState → AdaptiveGridTree ─────────────────────────────────────

def unwrap_to_tree(
    tree_state: TreeState,
    D:          int,
    bounds:     list[tuple[float, float]] | None = None,
    max_depth:  int | None = None,
) -> tuple[AdaptiveGridTree, np.ndarray]:
    """
    Reconstruct an AdaptiveGridTree from a TreeState.

    Returns
    -------
    tree       : AdaptiveGridTree with the encoded structure
    node_rates : (n_nodes,) float array in DFS all-nodes order
    """
    if max_depth is None:
        max_depth = tree_state.max_depth

    sd  = np.array(tree_state.split_dim)
    st  = np.array(tree_state.split_type)
    rt  = np.array(tree_state.rates)
    mn  = tree_state.max_nodes

    tree = AdaptiveGridTree(D, bounds, max_depth)
    tree.root = BasisNode(states=[State1D(CONST) for _ in range(D)])

    node_to_bfs: dict[int, int] = {}

    queue: deque[tuple[BasisNode, int]] = deque([(tree.root, 0)])
    while queue:
        node, bfs_i = queue.popleft()
        if bfs_i >= mn:
            continue
        node_to_bfs[id(node)] = bfs_i

        dim   = int(sd[bfs_i])
        stype = int(st[bfs_i])
        if dim < 0:
            continue

        s = node.states[dim]
        node._own_dim = dim;  node._own_type = stype

        if stype == 0:   # CONST → LINEAR
            cs = list(node.states); cs[dim] = s.child_linear()
            child = BasisNode(states=cs, parent=node, split_dim=dim)
            node.left = child
            queue.append((child, bfs_left(bfs_i)))

        elif stype == 1:  # LINEAR → HAT
            cs = list(node.states); cs[dim] = s.child_hat()
            child = BasisNode(states=cs, parent=node, split_dim=dim)
            node.left = child
            queue.append((child, bfs_left(bfs_i)))

        else:             # HAT split
            ls = list(node.states); ls[dim] = s.child_left()
            rs = list(node.states); rs[dim] = s.child_right()
            left  = BasisNode(states=ls, parent=node, split_dim=dim, split_side=0)
            right = BasisNode(states=rs, parent=node, split_dim=dim, split_side=1)
            node.left = left;  node.right = right
            queue.append((left,  bfs_left(bfs_i)))
            queue.append((right, bfs_right(bfs_i)))

    nodes_dfs  = tree.all_nodes()
    node_rates = np.array([rt[node_to_bfs[id(n)]] for n in nodes_dfs])
    return tree, node_rates


# ── Node state lookup table ───────────────────────────────────────────────────

def compute_node_states(
    tree_state: TreeState,
    D: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-node 1D state arrays for the current tree.

    Traverses from the root, propagating the effect of each split.
    Absent BFS slots keep their default (CONST, lo=0, hi=1).

    Returns
    -------
    node_kind : (max_nodes, D) int32  — CONST=0, LINEAR=1, HAT=2
    node_lo   : (max_nodes, D) float  — lo for HAT; 0 for CONST/LINEAR
    node_hi   : (max_nodes, D) float  — hi for HAT; 1 for CONST/LINEAR
    """
    sd  = np.array(tree_state.split_dim)
    st  = np.array(tree_state.split_type)
    mn  = tree_state.max_nodes

    node_kind = np.zeros((mn, D), dtype=np.int32)   # default: CONST
    node_lo   = np.zeros((mn, D))
    node_hi   = np.ones((mn, D))

    root_kind = np.zeros(D, dtype=np.int32)
    root_lo   = np.zeros(D)
    root_hi   = np.ones(D)

    queue: deque[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = deque(
        [(0, root_kind, root_lo, root_hi)]
    )

    while queue:
        bfs_i, k, lo, hi = queue.popleft()
        if bfs_i >= mn:
            continue
        node_kind[bfs_i] = k
        node_lo[bfs_i]   = lo
        node_hi[bfs_i]   = hi

        dim   = int(sd[bfs_i])
        stype = int(st[bfs_i])
        if dim < 0:
            continue

        if stype == 0:   # CONST → LINEAR: child gets LINEAR in dim
            ck = k.copy(); ck[dim] = LINEAR
            cl = lo.copy(); ch = hi.copy()
            queue.append((bfs_left(bfs_i), ck, cl, ch))

        elif stype == 1:  # LINEAR → HAT(0, 1)
            ck = k.copy(); ck[dim] = HAT
            cl = lo.copy(); cl[dim] = 0.0
            ch = hi.copy(); ch[dim] = 1.0
            queue.append((bfs_left(bfs_i), ck, cl, ch))

        else:             # HAT split at midpoint
            m = (lo[dim] + hi[dim]) / 2.0
            lk = k.copy(); ll = lo.copy(); lh = hi.copy(); lh[dim] = m
            rk = k.copy(); rl = lo.copy(); rh = hi.copy(); rl[dim] = m
            queue.append((bfs_left(bfs_i),  lk, ll, lh))
            queue.append((bfs_right(bfs_i), rk, rl, rh))

    return node_kind, node_lo, node_hi


# ── JAX evaluation ────────────────────────────────────────────────────────────

@jax.jit
def eval_basis_jax(
    x:           jnp.ndarray,
    node_kind:   jnp.ndarray,
    node_lo:     jnp.ndarray,
    node_hi:     jnp.ndarray,
    active_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Evaluate all active basis functions at point x.

    Parameters
    ----------
    x           : (D,) evaluation point
    node_kind   : (max_nodes, D) int32 — 0=CONST, 1=LINEAR, 2=HAT
    node_lo     : (max_nodes, D) float
    node_hi     : (max_nodes, D) float
    active_mask : (max_nodes,) bool

    Returns
    -------
    phi : (max_nodes,) float — phi_i(x); 0 for absent nodes
    """
    def eval1d(kind_d, lo_d, hi_d, x_d):
        w       = hi_d - lo_d
        mid     = lo_d + w / 2.0
        rising  = 2.0 * (x_d - lo_d) / w
        falling = 2.0 * (hi_d - x_d) / w
        hat_val = jnp.where(x_d <= mid, rising, falling)
        hat_val = jnp.where((x_d < lo_d) | (x_d > hi_d), 0.0, hat_val)
        return jnp.where(kind_d == CONST, 1.0,
               jnp.where(kind_d == LINEAR, x_d, hat_val))

    def eval_node(kinds, los, his):
        per_dim = jax.vmap(eval1d)(kinds, los, his, x)
        return jnp.prod(per_dim)

    all_evals = jax.vmap(eval_node)(node_kind, node_lo, node_hi)
    return jnp.where(active_mask, all_evals, 0.0)


@jax.jit
def design_matrix_jax(
    X:           jnp.ndarray,
    node_kind:   jnp.ndarray,
    node_lo:     jnp.ndarray,
    node_hi:     jnp.ndarray,
    active_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Design matrix Phi[i, j] = phi_j(X[i]).

    Parameters
    ----------
    X           : (n_pts, D)
    node_kind   : (max_nodes, D) int32
    node_lo     : (max_nodes, D) float
    node_hi     : (max_nodes, D) float
    active_mask : (max_nodes,) bool

    Returns
    -------
    Phi : (n_pts, max_nodes) float
    """
    return jax.vmap(eval_basis_jax, in_axes=(0, None, None, None, None))(
        X, node_kind, node_lo, node_hi, active_mask
    )


# ── RJMCMC helpers (Python-level) ────────────────────────────────────────────

def splittable_leaves(tree_state: TreeState) -> np.ndarray:
    """BFS indices of leaf nodes whose children would fit within max_nodes."""
    mn  = tree_state.max_nodes
    idx = tree_state.leaf_bfs_indices()
    return idx[2 * idx + 2 < mn]


def mergeable_internals(tree_state: TreeState) -> np.ndarray:
    """BFS indices of internal nodes whose all children are leaves."""
    sd  = np.array(tree_state.split_dim)
    st  = np.array(tree_state.split_type)
    rt  = np.array(tree_state.rates)
    mn  = tree_state.max_nodes
    out = []
    for i in np.where(sd >= 0)[0]:
        li = bfs_left(int(i))
        ri = bfs_right(int(i))
        if li >= mn or sd[li] >= 0 or not np.isfinite(rt[li]):
            continue
        if int(st[i]) == 2:
            if ri >= mn or sd[ri] >= 0 or not np.isfinite(rt[ri]):
                continue
        out.append(int(i))
    return np.array(out, dtype=int)
