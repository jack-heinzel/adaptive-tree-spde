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

Reusable functions are factored into `.py` files in `code/` and imported by the notebooks. As modules are added, document them here.

## Development Environment

This project uses Jupyter notebooks with Python helper modules. To work with it:

```
jupyter notebook
# or
jupyter lab
```
