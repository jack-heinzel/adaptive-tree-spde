# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This codebase experiments with spatial statistics methods for inferring the intensity function of an inhomogeneous Poisson process. The underlying space may be 2, 3, or 4 dimensional. Methods are organized in a hierarchy of increasing sophistication across multiple notebooks.

## Architecture

### Hierarchy of Methods

**Level 1 — `code/spde_triangulation.ipynb`**: Baseline approach. Construct a fixed triangulation (mesh) of the domain, then place a stationary Gaussian random field over it using the SPDE representation of Matérn covariance kernels (Lindgren, Rue & Lindström 2011). Inference uses finite element methods (FEM) to discretize the SPDE operator.

**Level 2 — (separate notebook, TBD)**: Extensions for non-Gaussian/discontinuous fields:
- Heavy-tailed SPDE fields driven by a Cauchy (α-stable) process instead of Gaussian white noise.
- Jump-diffusion SPDEs: add a Poisson-type term to the SPDE representing finite jumps over infinitesimal regions.

### Helper Modules

Reusable functions are factored into `.py` files in `code/` and imported by the notebooks.

| Module | Purpose |
|---|---|
| `triangulation.py` | `AdaptiveKDMesh` (kd-tree adaptive Delaunay mesh), `assemble_fem_matrices`, `precision_matrix` |
| `inference.py` | Log-posterior construction, NUTS runner, posterior prediction |
| `pcn.py` | Preconditioned Crank-Nicolson sampler (BlackJAX-compatible) |
| `hilbert_hmc.py` | `HilbertHMC` and `HilbertNUTS` classes (velocity-parameterised, CG-based); factory functions `hilbert_hmc`, `hilbert_nuts` |
| `rjhmc_state.py` | `TreeState` — fixed-shape BFS encoding of the kd-tree for RJMCMC; `wrap_to_tree_state`, `unwrap_to_mesh`, `birth_update`, `death_update`, `splittable_leaves`, `mergeable_internals` |

**`TreeState` encoding** (`rjhmc_state.py`): The kd-tree is serialised in BFS heap order — node i has children at 2i+1 and 2i+2. Three `(max_nodes,)` JAX arrays carry the state:
- `split_axis` (int32): split axis for internal nodes, -1 for leaves/absent
- `split_val` (float64): split coordinate for internal nodes, NaN for leaves/absent
- `rates` (float64): log-intensity at leaf nodes, NaN for internals/absent

All arrays are padded to `max_nodes = 2^(max_depth+1) - 1`, giving every sample the same fixed shape for `jax.lax.scan`. The split axis is stored (not inferred from depth) because `AdaptiveKDMesh` splits along the longest bounding-box axis, not cyclically. `wrap_to_tree_state` / `unwrap_to_mesh` preserve DFS leaf order to stay consistent with FEM matrix assembly.

## Development Environment

This project uses Jupyter notebooks with Python helper modules. To work with it:

```
jupyter notebook
# or
jupyter lab
```

## Lab Notebook (`draft/`)

The LaTeX documents in `draft/` function as a **lab notebook**: they document
experiments, derive the mathematics, and record design decisions as they are
made.  They are not a polished paper.

The notebook is split into **three standalone documents** that share a common
`preamble.tex` (packages, macros, theorem environments) and `refs.bib`, each
independently compilable on Overleaf:

| File | Topic |
|---|---|
| `foundations.tex` | Problem setting, Mat\'ern SPDE prior, basis-agnostic FEM/Galerkin weak form and precision matrix, likelihood discretisation, hyperparameter model, NUTS. The shared substrate. |
| `approximators.tex` | Universal function approximators on $[0,1]^D$: the adaptive **hat tree basis** (`\label{sec:hat-tree}`, primary; includes the uniqueness/`sec:uniqueness` analysis), the adaptive **piecewise-constant tree basis** (`\label{sec:const-tree}`), the **Delaunay-triangulation baseline** (`\label{sec:baseline}`, demoted Level 1), the stationary anisotropic extension, and the **Lessons Learned** appendix (`\label{sec:lessons}`). |
| `samplers.tex` | Reversible-jump MCMC over the refinement tree (`\label{sec:rjmcmc}`) and dimension-robust infinite-dimensional samplers (pCN, Hilbert-space HMC) for BlackJAX. |

Because the documents compile separately, **cross-document references are written
as plain text** (e.g. "the companion *Foundations* note"), not `\ref` — only
within-document targets use `\ref`/`\eqref`/`\label`.

`main.tex` is the **superseded monolith** that predates the split; it is kept
only until the split is confirmed on Overleaf, then should be removed.  Do not
edit it — make changes in the three split documents.

**Compilation happens on Overleaf**, not locally — the draft is attached to an
Overleaf project.  Do not attempt to `pdflatex`/`latexmk` it locally to "verify"
(the local TeX install is missing packages such as `algpseudocode`, so it will
fail for environmental reasons unrelated to the edit).  Edit the split documents
directly and let the user compile on Overleaf.  When changing structure,
sanity-check `\ref`/`\label` consistency by inspection rather than by building.

**Appendix: Lessons Learned and Paths Avoided** (`\label{sec:lessons}`) is a
running record of approaches that were explored and ruled out, together with the
reason why.  Whenever a promising direction turns out to be a dead end, add an
entry here — this prevents the same dead ends from being rediscovered.  Current
entries: Delaunay triangulation (combinatorial explosion in $D\ge 3$,
non-static under refinement) and why the hat-function tree basis was chosen
instead.
