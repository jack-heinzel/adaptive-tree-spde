"""
Poisson process likelihood and MCMC inference for a log-Gaussian Cox process
(LGCP) with Matern SPDE prior, using BlackJAX (NUTS sampler).

Model
-----
    log lambda(s) = mu + eta(s),   eta(s) = sum_i x_i phi_i(s)
    x   ~ N(0, Q^{-1})             Matern SPDE prior
    mu  ~ flat                     intercept (absorbs mean log-intensity)

The integral int lambda ds is approximated by one-point (centroid) quadrature:
    int_T exp(mu + eta) ds  ~  |T| * exp(mu + mean(x at vertices of T))
"""

import warnings
from math import factorial

import numpy as np
import jax
import jax.numpy as jnp
import blackjax
from jax_tqdm import scan_tqdm
from scipy.linalg import eigh
from scipy.sparse import coo_array


# ---------------------------------------------------------------------------
# Observation matrix  (numpy / scipy — assembled once before sampling)
# ---------------------------------------------------------------------------

def build_observation_matrix(nodes, tri, data_points):
    """
    Sparse matrix A of shape (n_obs, n_nodes) where A[j, i] = phi_i(data_points[j]).

    Uses barycentric coordinates from the Delaunay triangulation.
    Points outside the triangulation are ignored (with a warning).
    """
    data_points = np.asarray(data_points, dtype=float)
    n_obs   = len(data_points)
    n_nodes = len(nodes)
    d       = nodes.shape[1]

    si    = tri.find_simplex(data_points)
    valid = si >= 0
    if not valid.all():
        warnings.warn(
            f"{(~valid).sum()} / {n_obs} data points lie outside the "
            "triangulation and will be ignored."
        )

    idx = si[valid]
    pts = data_points[valid]

    T  = tri.transform[idx, :d, :]   # (n_valid, d, d)
    v0 = tri.transform[idx,  d, :]   # (n_valid, d)
    bary_rest = np.einsum("nij,nj->ni", T, pts - v0)        # (n_valid, d)
    bary_0    = 1.0 - bary_rest.sum(axis=1, keepdims=True)  # (n_valid, 1)
    bary      = np.hstack([bary_0, bary_rest])               # (n_valid, d+1)

    valid_rows = np.where(valid)[0]
    rows = np.repeat(valid_rows, d + 1)
    cols = tri.simplices[idx].ravel()
    vals = bary.ravel()

    return coo_array((vals, (rows, cols)), shape=(n_obs, n_nodes)).tocsr()


def _simplex_areas(nodes, simplices):
    d   = nodes.shape[1]
    pts = nodes[simplices]
    B   = (pts[:, 1:, :] - pts[:, :1, :]).transpose(0, 2, 1)
    return np.abs(np.linalg.det(B)) / factorial(d)


# ---------------------------------------------------------------------------
# JAX log-posterior
# ---------------------------------------------------------------------------

def make_log_posterior(Q, nodes, simplices, tri, data_points, obs_bounds=None):
    """
    Build a JAX log-posterior with Q fixed (kappa already chosen).

    Position: {'mu': scalar, 'x': array(n_nodes,)}.

    Parameters
    ----------
    obs_bounds : list of (lo, hi) per dimension, optional
        The observation domain, e.g. [(0,1),(0,1)].  Only nodes inside
        obs_bounds contribute to the likelihood integral ∫ λ ds.  Buffer /
        boundary nodes still regularise the field via Q but do not consume
        any of the 'n_obs budget', so the Gamma posterior E[N_obs] = n_obs
        holds over the observation domain rather than the larger mesh domain.
        Defaults to the full mesh (all nodes), which over-normalises when the
        mesh extends beyond the observation domain.
    """
    areas     = _simplex_areas(nodes, simplices)
    A         = build_observation_matrix(nodes, tri, data_points)
    data_term = np.asarray(A.sum(axis=0)).ravel()
    n_obs     = len(data_points)

    # Lumped mass: c_i = sum_{T containing i} |T| / (d+1)
    # Approximates ∫ λ ds ≈ Σ_i c_i exp(μ + x_i), consistent with FEM prior.
    d        = nodes.shape[1]
    n_nodes  = len(nodes)
    c_lumped = np.bincount(
        simplices.ravel(),
        weights=np.repeat(areas, d + 1) / (d + 1),
        minlength=n_nodes,
    )

    # Mask to observation domain: buffer nodes get weight 0 in the likelihood
    # integral so they don't consume the n_obs normalisation budget.
    if obs_bounds is not None:
        lo = np.array([b[0] for b in obs_bounds])
        hi = np.array([b[1] for b in obs_bounds])
        in_obs = np.all((nodes >= lo) & (nodes <= hi), axis=1)
        c_lumped_obs = c_lumped * in_obs
    else:
        c_lumped_obs = c_lumped

    Q_jax            = jnp.array(Q.toarray())
    c_lumped_obs_jax = jnp.array(c_lumped_obs)
    data_term_jax    = jnp.array(data_term)

    def log_posterior(position):
        mu = position["mu"]
        x  = position["x"]
        integral = jnp.dot(c_lumped_obs_jax, jnp.exp(mu + x))
        ll = n_obs * mu + jnp.dot(data_term_jax, x) - integral
        lp = -0.5 * jnp.dot(x, Q_jax @ x)
        return ll + lp

    meta = {
        "n_obs":       n_obs,
        "domain_area": float(c_lumped_obs.sum()),  # obs-domain area (≈ area of obs_bounds)
        "data_term":   data_term,
        "areas":       areas,
        "c_lumped":    c_lumped_obs,   # diagnostic: (exp(mu+x) @ c_lumped).mean() ≈ n_obs
    }
    return log_posterior, meta


def make_log_posterior_with_hyperparams(
    C, G, nodes, simplices, tri, data_points,
    alpha            = 2,
    log_kappa_mean   = None,
    log_kappa_std    = 1.0,
    log_sigma_mean   = 0.0,
    log_sigma_std    = 1.0,
    obs_bounds       = None,
):
    """
    Build a JAX log-posterior that jointly infers (mu, log_kappa, log_sigma, x).

    Model
    -----
        log lambda(s) = mu + sigma * x(s)
        x   ~ N(0, Q^{-1}(kappa))     Matern SPDE prior
        mu  ~ flat                     mean log-intensity
        kappa ~ log-Normal prior       inverse range
        sigma ~ log-Normal prior       marginal std of log-intensity deviations

    sigma separates the amplitude of spatial variation from the range (kappa).
    Without sigma, the amplitude is fixed by kappa via the Matern marginal variance
    sigma_x^2 = 1/(4 pi kappa^2) (for alpha=2, d=2); with kappa=6 this is ~0.002,
    far too small to fit data with ~1 log-unit of variation.

    K(kappa) = kappa^2 * C + G is assembled inside the callable so JAX can
    differentiate through it.  For alpha=2 the precision of x is

        Q(kappa) = K  diag(C_lumped)^{-1}  K

    and its log-determinant is

        log det Q = 2 log det K  -  sum_i log c_i

    Position: {'mu': scalar, 'log_kappa': scalar, 'log_sigma': scalar,
               'x': array(n_nodes,)}.

    Parameterisation note — centred form avoids Neal's funnel
    ----------------------------------------------------------
    x is the log-intensity field directly (not a scaled white noise).
    sigma enters the PRIOR on x, not the likelihood:

        x   ~ N(0, sigma^2 Q^{-1})
        log lambda(s) = mu + x(s)

    The prior log-density is -0.5/sigma^2 * x^T Q x + 0.5 log|Q| - n*log(sigma).
    The -n*log(sigma) Jacobian (n = number of nodes) strongly anchors sigma,
    preventing the runaway that occurs when sigma multiplies x in the likelihood
    (which lets (2*sigma, x/2) give identical likelihoods with a better x prior).

    Parameters
    ----------
    C, G             : sparse FEM matrices from assemble_fem_matrices
    nodes, simplices : mesh geometry
    tri              : scipy.spatial.Delaunay
    data_points      : observed Poisson process points
    alpha            : SPDE smoothness order (integer, default 2)
    log_kappa_mean   : prior mean for log_kappa; defaults to log of 1/(3*range)
                       where range ~ 0.3 * domain diameter
    log_kappa_std    : prior std for log_kappa (default 1.0, weakly informative)
    log_sigma_mean   : prior mean for log_sigma (default 0.0 → sigma=1)
    log_sigma_std    : prior std for log_sigma (default 1.0, allows sigma in ~[0.05, 20])
    obs_bounds       : list of (lo, hi) per dimension, optional.
                       Restricts the likelihood integral ∫ λ ds to the
                       observation domain (see make_log_posterior for details).
    """
    areas     = _simplex_areas(nodes, simplices)
    A         = build_observation_matrix(nodes, tri, data_points)
    data_term = np.asarray(A.sum(axis=0)).ravel()
    n_obs     = len(data_points)

    # Full lumped mass — used for the FEM prior Q (must cover the whole mesh).
    c_lumped = np.asarray(C.sum(axis=1)).ravel()

    # Obs-domain lumped mass — used only for the likelihood integral.
    if obs_bounds is not None:
        lo = np.array([b[0] for b in obs_bounds])
        hi = np.array([b[1] for b in obs_bounds])
        in_obs = np.all((nodes >= lo) & (nodes <= hi), axis=1)
        c_lumped_obs = c_lumped * in_obs
    else:
        c_lumped_obs = c_lumped

    # Default kappa prior: centred on kappa that gives range ~ 30% of domain
    if log_kappa_mean is None:
        domain_diam   = np.sqrt(float(areas.sum())) * 2   # crude estimate
        log_kappa_mean = float(np.log(3.0 / (0.3 * domain_diam)))

    # Precomputed constants (do not depend on kappa)
    log_det_C_inv = -float(np.sum(np.log(c_lumped)))   # -sum log c_i

    # Generalized eigenvalues mu_i of  G v = mu C v  (one-time O(n³) cost).
    # Gives  log det K(kappa) = log det C + sum_i log(kappa² + mu_i),
    # reducing per-step log det from O(n³) to O(n).
    # Valid only when K = kappa²C + G is a scalar pencil (stationary kappa).
    # Non-stationary kappa or H would require SLQ or dense slogdet instead.
    G_dense = G.toarray()
    C_dense = C.toarray()
    mu_eig = eigh(G_dense, C_dense, eigvals_only=True)           # (n,)
    _, log_det_C = np.linalg.slogdet(C_dense)
    log_det_C = float(log_det_C)

    n_nodes = len(nodes)

    # JAX arrays
    C_jax                = jnp.array(C_dense)
    G_jax                = jnp.array(G_dense)
    c_lumped_inv_jax     = jnp.array(1.0 / c_lumped)
    c_lumped_obs_jax     = jnp.array(c_lumped_obs)
    mu_jax               = jnp.array(mu_eig)
    data_term_jax        = jnp.array(data_term)
    log_kappa_mean_j     = jnp.array(log_kappa_mean)
    log_kappa_std_j      = jnp.array(log_kappa_std)
    log_sigma_mean_j     = jnp.array(log_sigma_mean)
    log_sigma_std_j      = jnp.array(log_sigma_std)
    n_nodes_j            = jnp.array(float(n_nodes))

    def log_posterior(position):
        mu        = position["mu"]
        log_kappa = position["log_kappa"]
        log_sigma = position["log_sigma"]
        x         = position["x"]

        kappa = jnp.exp(log_kappa)
        sigma = jnp.exp(log_sigma)

        # ---- build K and Q ------------------------------------------------
        K = kappa ** 2 * C_jax + G_jax                   # (n, n)
        # Q = K diag(c_inv) K  via  (K * c_inv[None,:]) @ K
        Q = (K * c_lumped_inv_jax[None, :]) @ K           # (n, n)

        # ---- log det K via precomputed generalized eigenvalues  O(n) -------
        log_det_K = log_det_C + jnp.sum(jnp.log(kappa ** 2 + mu_jax))
        log_det_Q = 2.0 * log_det_K + log_det_C_inv

        # ---- Gaussian prior on x: x ~ N(0, sigma^2 Q^{-1}) ----------------
        # log p(x|sigma,kappa) = -0.5/sigma^2 * x^T Q x
        #                        + 0.5 log|Q| - n_nodes * log(sigma)
        # The -n_nodes*log(sigma) Jacobian anchors sigma (prevents funnel).
        lp_x = (-0.5 / sigma ** 2 * jnp.dot(x, Q @ x)
                 + 0.5 * log_det_Q
                 - n_nodes_j * log_sigma)

        # ---- LGCP likelihood: log lambda = mu + x (sigma absorbed into x) -
        integral = jnp.dot(c_lumped_obs_jax, jnp.exp(mu + x))
        ll       = n_obs * mu + jnp.dot(data_term_jax, x) - integral

        # ---- hyperpriors on log_kappa and log_sigma -----------------------
        lp_kappa = -0.5 * ((log_kappa - log_kappa_mean_j) / log_kappa_std_j) ** 2
        lp_sigma = -0.5 * ((log_sigma - log_sigma_mean_j) / log_sigma_std_j) ** 2

        return ll + lp_x + lp_kappa + lp_sigma

    meta = {
        "n_obs":          n_obs,
        "domain_area":    float(c_lumped_obs.sum()),
        "data_term":      data_term,
        "areas":          areas,
        "log_kappa_mean": log_kappa_mean,
        "log_sigma_mean": log_sigma_mean,
        "c_lumped":       c_lumped_obs,
    }
    return log_posterior, meta


# ---------------------------------------------------------------------------
# NUTS sampling via BlackJAX
# ---------------------------------------------------------------------------

def run_nuts(
    log_posterior,
    meta,
    n_nodes,
    num_warmup          = 1000,
    num_samples         = 2000,
    thinning            = 1,
    seed                = 0,
    init_position       = None,
    progress_bar_kwargs = None,
    warmup_progress_bar = False,
):
    """
    Run NUTS with BlackJAX window adaptation (tunes step size + mass matrix).

    Parameters
    ----------
    log_posterior        : callable from make_log_posterior[_with_hyperparams]
    meta                 : dict from the same factory
    n_nodes              : int
    num_warmup           : int — adaptation steps (discarded)
    num_samples          : int — posterior draws to return
    thinning             : int — keep every n-th sample; total MCMC steps = num_samples * thinning
    seed                 : int
    init_position        : optional pytree matching log_posterior's position structure;
                           defaults to mu=log(N/area), x=0 (and log_kappa=log_kappa_mean
                           if that key is present in meta)
    progress_bar_kwargs  : dict passed to scan_tqdm, e.g. {"tqdm_type": "notebook"}
    warmup_progress_bar  : bool — show a fastprogress bar during window adaptation

    Returns
    -------
    samples : pytree of arrays with a leading (num_samples,) axis
    info    : BlackJAX info object for the kept samples (acceptance_rate, is_divergent, ...)
    """
    rng_key = jax.random.PRNGKey(seed)

    if init_position is None:
        init_position = {
            "mu": jnp.array(np.log(meta["n_obs"] / meta["domain_area"])),
            "x":  jnp.zeros(n_nodes),
        }
        if "log_kappa_mean" in meta:
            init_position["log_kappa"] = jnp.array(meta["log_kappa_mean"])
        if "log_sigma_mean" in meta:
            init_position["log_sigma"] = jnp.array(meta["log_sigma_mean"])

    # Window adaptation: tunes step size and diagonal mass matrix
    warmup = blackjax.window_adaptation(
        blackjax.nuts, log_posterior, progress_bar=warmup_progress_bar
    )
    rng_key, warmup_key = jax.random.split(rng_key)
    if not warmup_progress_bar:
        print(f"Running {num_warmup} warmup steps...")
    (state, parameters), _ = warmup.run(warmup_key, init_position, num_steps=num_warmup)
    print(f"  step_size = {parameters['step_size']:.4f}")

    # Build tuned NUTS kernel
    kernel = blackjax.nuts(log_posterior, **parameters)

    def thin_steps(state, thin_keys):
        """Run `thinning` steps; return only the final state and its info."""
        def one_step(state, rng_key):
            state, info = kernel.step(rng_key, state)
            return state, info
        state, info = jax.lax.scan(one_step, state, thin_keys)
        last_info = jax.tree.map(lambda x: x[-1], info)
        return state, (state.position, last_info)

    # scan_tqdm expects xs = jnp.arange(n) (scalar integers); derive keys
    # inside the step via fold_in so the tqdm condition x % rate == 0 stays scalar.
    rng_key, sample_key = jax.random.split(rng_key)

    _pb_kwargs = {"desc": "Sampling", "tqdm_type": "std", **(progress_bar_kwargs or {})}

    @scan_tqdm(num_samples, **_pb_kwargs)
    def outer_step(state, i):
        thin_keys = jax.random.split(jax.random.fold_in(sample_key, i), thinning)
        return thin_steps(state, thin_keys)

    # Draw samples: (num_samples * thinning) total MCMC steps
    _, (samples, info) = jax.lax.scan(outer_step, state, jnp.arange(num_samples))
    accept_rate = float(info.acceptance_rate.mean())
    n_diverge   = int(info.is_divergent.sum())
    print(f"  acceptance rate = {accept_rate:.2f},  divergences = {n_diverge}")

    return samples, info


# ---------------------------------------------------------------------------
# Posterior prediction
# ---------------------------------------------------------------------------

def predict_intensity(samples, nodes, tri, eval_points):
    """
    Posterior predictive intensity at eval_points.

    Parameters
    ----------
    samples     : dict from run_nuts — keys include 'mu' (S,), 'x' (S, n_nodes),
                  and optionally 'log_sigma' (S,) when sigma was inferred.
    nodes       : array (n_nodes, d)
    tri         : scipy.spatial.Delaunay
    eval_points : array (n_eval, d)

    Returns
    -------
    lam : array (S, n_eval) — posterior samples of lambda(s);
          NaN for eval_points outside the triangulation.
    """
    eval_points = np.asarray(eval_points, dtype=float)
    d  = nodes.shape[1]
    si = tri.find_simplex(eval_points)
    valid = si >= 0

    mu_samples = np.array(samples["mu"])   # (S,)
    x_samples  = np.array(samples["x"])   # (S, n_nodes)  — x is the full field

    S      = len(mu_samples)
    result = np.full((S, len(eval_points)), np.nan)

    if valid.any():
        idx = si[valid]
        pts = eval_points[valid]

        T  = tri.transform[idx, :d, :]
        v0 = tri.transform[idx,  d, :]
        bary_rest = np.einsum("nij,nj->ni", T, pts - v0)
        bary_0    = 1.0 - bary_rest.sum(axis=1, keepdims=True)
        bary      = np.hstack([bary_0, bary_rest])             # (n_valid, d+1)

        vtx = tri.simplices[idx]                               # (n_valid, d+1)

        # eta(s) = Σ_i x_i phi_i(s) — interpolated log-intensity field
        eta = (x_samples[:, vtx] * bary[None, :, :]).sum(axis=-1)  # (S, n_valid)
        result[:, valid] = np.exp(mu_samples[:, None] + eta)

    return result  # (S, n_eval), NaN for points outside the triangulation
