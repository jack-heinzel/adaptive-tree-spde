"""
Utilities for generating synthetic point patterns from inhomogeneous Poisson
processes with truncated Gaussian mixture intensity functions.
"""

import numpy as np
from scipy.stats import multivariate_normal


def mixture_density(points, components):
    """
    Evaluate the (unnormalized) mixture density at given points.

    This is proportional to the Poisson intensity function — useful for
    plotting the true intensity on a grid alongside sampled points.

    Parameters
    ----------
    points : array-like, shape (n, d)
    components : list of dict, each with:
        'mean'   : array-like, shape (d,)
        'cov'    : array-like, shape (d, d)
        'weight' : float

    Returns
    -------
    density : np.ndarray, shape (n,)
    """
    points = np.atleast_2d(points)
    weights = np.array([c["weight"] for c in components], dtype=float)
    weights /= weights.sum()

    density = np.zeros(len(points))
    for c, w in zip(components, weights):
        density += w * multivariate_normal.pdf(points, mean=c["mean"], cov=c["cov"])

    return density


def sample_poisson_process(components, total_intensity, bounds, seed=None):
    """
    Sample from an inhomogeneous Poisson process whose intensity is a mixture
    of Gaussians truncated to the given domain.

    The total number of points N ~ Poisson(total_intensity). Each point is
    drawn from the mixture density restricted to the domain via rejection
    sampling: candidate points are drawn from the untruncated mixture and
    discarded if they fall outside the domain. Weights are not adjusted for
    truncation, so the correct truncated mixture distribution is recovered
    automatically (the per-component acceptance probability is proportional
    to the Gaussian mass inside the domain, yielding the right mixture
    proportions after rejection).

    Parameters
    ----------
    components : list of dict, each with:
        'mean'   : array-like, shape (d,)
        'cov'    : array-like, shape (d, d)
        'weight' : float  (need not sum to 1)
    total_intensity : float
        Expected total number of points (Poisson mean).
    bounds : list of (float, float)
        Domain per dimension: [(x0_min, x0_max), (x1_min, x1_max), ...]
    seed : int or np.random.Generator, optional

    Returns
    -------
    points : np.ndarray, shape (N, d)
        Sampled point pattern. N is a Poisson(total_intensity) random variable.
    """
    rng = np.random.default_rng(seed)
    d = len(bounds)

    weights = np.array([c["weight"] for c in components], dtype=float)
    weights /= weights.sum()

    n_points = rng.poisson(total_intensity)
    if n_points == 0:
        return np.empty((0, d))

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    accepted = []
    oversample_factor = 4

    while len(accepted) < n_points:
        needed = n_points - len(accepted)
        batch_size = max(needed * oversample_factor, 64)

        comp_idx = rng.choice(len(components), size=batch_size, p=weights)
        candidates = np.empty((batch_size, d))

        for k, comp in enumerate(components):
            mask = comp_idx == k
            if mask.any():
                candidates[mask] = rng.multivariate_normal(
                    comp["mean"], comp["cov"], size=int(mask.sum())
                )

        in_domain = np.all((candidates >= lo) & (candidates <= hi), axis=1)
        accepted.extend(candidates[in_domain].tolist())

        accept_rate = float(in_domain.mean())
        if accept_rate > 0:
            oversample_factor = max(4, int(2.0 / accept_rate))

    return np.array(accepted[:n_points])
