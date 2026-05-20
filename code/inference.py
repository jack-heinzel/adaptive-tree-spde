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
jax.config.update("jax_enable_x64", True)   # float64 required: linalg.solve in float32
                                              # produces inaccurate x values and divergences
import jax.numpy as jnp
import blackjax
from jax_tqdm import scan_tqdm
from scipy.linalg import eigh as sp_eigh
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
    bary_last = 1.0 - bary_rest.sum(axis=1, keepdims=True)  # (n_valid, 1) — λ for simplices[idx, d]
    # scipy stores the LAST simplex vertex as the reference point (transform[i, d, :]),
    # so bary_rest gives λ for vertices 0..d-1 and bary_last gives λ for vertex d.
    # Order must match tri.simplices[idx] = [v0, v1, ..., vd].
    bary = np.hstack([bary_rest, bary_last])                 # (n_valid, d+1)

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

def make_log_posterior(Q, nodes, simplices, tri, data_points):
    """
    Build a JAX log-posterior with Q fixed (kappa already chosen).

    Position: {'mu': scalar, 'x': array(n_nodes,)}.

    The integral ∫ λ ds is computed over the full mesh (including any buffer
    nodes).  A buffer of width ε adds ~4ε of extra area; for ε=1e-3 this is
    <0.5% of [0,1]² and is absorbed by mu.  Using the full mesh keeps
    data_term and the integral consistent — masking the integral but not
    data_term creates an unbalanced likelihood gradient on buffer nodes that
    causes NUTS divergences in joint hyperparameter inference.
    """
    areas     = _simplex_areas(nodes, simplices)
    A         = build_observation_matrix(nodes, tri, data_points)
    data_term = np.asarray(A.sum(axis=0)).ravel()
    n_obs     = len(data_points)

    d        = nodes.shape[1]
    n_nodes  = len(nodes)
    c_lumped = np.bincount(
        simplices.ravel(),
        weights=np.repeat(areas, d + 1) / (d + 1),
        minlength=n_nodes,
    )

    Q_jax         = jnp.array(Q.toarray())
    c_lumped_jax  = jnp.array(c_lumped)
    data_term_jax = jnp.array(data_term)

    def log_posterior(position):
        mu = position["mu"]
        x  = position["x"]
        integral = jnp.dot(c_lumped_jax, jnp.exp(mu + x))
        ll = n_obs * mu + jnp.dot(data_term_jax, x) - integral
        lp = -0.5 * jnp.dot(x, Q_jax @ x)
        return ll + lp

    meta = {
        "n_obs":       n_obs,
        "domain_area": float(c_lumped.sum()),
        "data_term":   data_term,
        "areas":       areas,
        "c_lumped":    c_lumped,
    }
    return log_posterior, meta


def make_log_posterior_with_hyperparams(
    C, G, nodes, simplices, tri, data_points,
    alpha            = 2,
    log_kappa_mean   = None,
    log_kappa_std    = 1.0,
    log_sigma_mean   = 0.0,
    log_sigma_std    = 1.0,
):
    """
    Build a JAX log-posterior that jointly infers (mu, log_kappa, log_sigma, x).

    Model
    -----
        log lambda(s) = mu + x(s)
        x ~ N(0, sigma^2 Q(kappa)^{-1})        centred Matern SPDE prior
        Q = K(kappa) C_lump^{-1} K(kappa),  K = kappa^2 C + G
        kappa ~ log-Normal prior              inverse range
        sigma ~ log-Normal prior              field amplitude

    Centred parameterisation
    -------------------------
    x is sampled directly.  The prior precision Q/sigma^2 varies with kappa
    and sigma so the NUTS mass matrix adapted at warmup is imperfect elsewhere.
    This is acceptable when the likelihood is sufficiently informative.

    log|K(kappa)| is computed cheaply via the generalised eigenvalue pencil.
    K = kappa^2 C + G  →  G v = mu C v  (solved once, O(n^3)).
    Then  log|K(kappa)| = log|C| + sum_i log(kappa^2 + mu_i),  O(n) per step.
    Its gradient w.r.t. log_kappa is 2*kappa^2 * sum_i 1/(kappa^2 + mu_i), also O(n).

    Log-prior on x:
        lp_x = log|K(kappa)| - n*log(sigma) - xQx / (2*sigma^2)
    where xQx = (Kx)^T C_lump^{-1} (Kx)  — one matrix-vector product, O(n^2).

    Position: {'mu': scalar, 'log_kappa': scalar, 'log_sigma': scalar,
               'x': array(n_nodes,)}.

    Parameters
    ----------
    C, G             : sparse FEM matrices from assemble_fem_matrices
    nodes, simplices : mesh geometry
    tri              : scipy.spatial.Delaunay
    data_points      : observed Poisson process points
    alpha            : SPDE smoothness order (only 2 supported)
    log_kappa_mean   : prior mean for log_kappa; defaults to log(3 / (0.3 * diam))
    log_kappa_std    : prior std for log_kappa (default 1.0)
    log_sigma_mean   : prior mean for log_sigma (default 0.0 → sigma=1)
    log_sigma_std    : prior std for log_sigma (default 1.0)
    """
    if alpha != 2:
        raise NotImplementedError("Only alpha=2 is implemented")

    areas     = _simplex_areas(nodes, simplices)
    A         = build_observation_matrix(nodes, tri, data_points)
    data_term = np.asarray(A.sum(axis=0)).ravel()
    n_obs     = len(data_points)
    n_nodes   = len(nodes)

    c_lumped = np.asarray(C.sum(axis=1)).ravel()

    if log_kappa_mean is None:
        domain_diam    = np.sqrt(float(areas.sum())) * 2
        log_kappa_mean = float(np.log(3.0 / (0.3 * domain_diam)))

    # --- One-time O(n^3) precomputation ---
    C_arr = C.toarray()
    G_arr = G.toarray()

    # Generalised eigenvalues of the pencil (G, C): G v = mu C v.
    # log|K(kappa)| = log|C| + sum_i log(kappa^2 + mu_i)  — O(n) per step.
    mu_eig    = sp_eigh(G_arr, C_arr, eigvals_only=True)   # (n_nodes,)
    log_C_det = float(np.linalg.slogdet(C_arr)[1])

    # Sparse COO structure for K @ x = kappa^2*(C @ x) + (G @ x).
    # K is never formed explicitly; scatter-add over nnz ~ 7n entries — O(n) per step.
    C_coo  = C.tocsr().tocoo()
    spm_rows = C_coo.row
    spm_cols = C_coo.col
    spm_c    = C_coo.data                                        # C nonzero values
    spm_g    = np.asarray(G.tocsr()[spm_rows, spm_cols]).ravel() # G at same positions

    mu_eig_jax    = jnp.array(mu_eig)
    spm_rows_j    = jnp.array(spm_rows)
    spm_cols_j    = jnp.array(spm_cols)
    spm_c_j       = jnp.array(spm_c)
    spm_g_j       = jnp.array(spm_g)
    c_lumped_jax  = jnp.array(c_lumped)
    data_term_jax = jnp.array(data_term)
    log_kappa_mean_j = jnp.array(log_kappa_mean)
    log_kappa_std_j  = jnp.array(log_kappa_std)
    log_sigma_mean_j = jnp.array(log_sigma_mean)
    log_sigma_std_j  = jnp.array(log_sigma_std)
    log_C_det_j      = jnp.array(log_C_det)

    def log_posterior(position):
        mu        = position["mu"]
        log_kappa = position["log_kappa"]
        log_sigma = position["log_sigma"]
        x         = position["x"]

        kappa = jnp.exp(log_kappa)
        sigma = jnp.exp(log_sigma)

        # K @ x via sparse pencil: O(nnz) ~ O(n) for 2-D meshes
        k_vals = kappa**2 * spm_c_j + spm_g_j
        Kx = jnp.zeros(n_nodes).at[spm_rows_j].add(k_vals * x[spm_cols_j])
        xQx = jnp.dot(Kx, Kx / c_lumped_jax)           # x^T Q x, Q = K C_lump^{-1} K

        # log|K(kappa)| via pencil eigenvalues: O(n)
        log_det_K = log_C_det_j + jnp.sum(jnp.log(kappa**2 + mu_eig_jax))
        lp_x = log_det_K - n_nodes * log_sigma - 0.5 * xQx / sigma**2

        integral = jnp.dot(c_lumped_jax, jnp.exp(mu + x))
        ll       = n_obs * mu + jnp.dot(data_term_jax, x) - integral

        lp_kappa = -0.5 * ((log_kappa - log_kappa_mean_j) / log_kappa_std_j)**2
        lp_sigma = -0.5 * ((log_sigma - log_sigma_mean_j) / log_sigma_std_j)**2

        return ll + lp_x + lp_kappa + lp_sigma

    meta = {
        "n_obs":          n_obs,
        "domain_area":    float(c_lumped.sum()),
        "data_term":      data_term,
        "areas":          areas,
        "log_kappa_mean": log_kappa_mean,
        "log_sigma_mean": log_sigma_mean,
        "c_lumped":       c_lumped,
    }
    return log_posterior, meta


def recover_x_samples(samples, C, G):
    """
    Convert whitened eps back to the field x = sigma * K(kappa)^{-1} diag(sqrt(c)) * eps.

    Call this after run_nuts when using make_log_posterior_with_hyperparams.
    Adds key "x" to the returned dict so predict_intensity works unchanged.
    """
    kappas  = np.exp(np.array(samples["log_kappa"]))   # (S,)
    sigmas  = np.exp(np.array(samples["log_sigma"]))   # (S,)
    epses   = np.array(samples["eps"])                 # (S, n)
    C_arr   = np.asarray(C.toarray())
    G_arr   = np.asarray(G.toarray())
    c_sqrt  = np.sqrt(np.asarray(C.sum(axis=1)).ravel())
    xs = np.array([
        sigmas[s] * np.linalg.solve(kappas[s] ** 2 * C_arr + G_arr, c_sqrt * epses[s])
        for s in range(len(kappas))
    ])
    return {**samples, "x": xs}


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
        # "c_lumped_sqrt" in meta signals the whitened (eps) parameterisation;
        # otherwise the centred (x) parameterisation is used.
        field_key = "eps" if "c_lumped_sqrt" in meta else "x"
        init_position = {
            "mu":      jnp.array(np.log(meta["n_obs"] / meta["domain_area"])),
            field_key: jnp.zeros(n_nodes),
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

def predict_intensity(samples, nodes, tri, eval_points, obs_bounds=None):
    """
    Posterior predictive intensity at eval_points.

    Parameters
    ----------
    samples     : dict from run_nuts — keys include 'mu' (S,), 'x' (S, n_nodes).
    nodes       : array (n_nodes, d)
    tri         : scipy.spatial.Delaunay
    eval_points : array (n_eval, d)
    obs_bounds  : list of (lo, hi) per dimension, optional.
        If given, any eval point whose containing triangle has at least one vertex
        outside obs_bounds is masked to NaN.  This prevents buffer nodes (which
        carry no likelihood weight) from inflating posterior variance at the
        observation-domain boundary.

    Returns
    -------
    lam : array (S, n_eval) — posterior samples of lambda(s);
          NaN for eval_points outside the triangulation or adjacent to buffer nodes.
    """
    eval_points = np.asarray(eval_points, dtype=float)
    d  = nodes.shape[1]
    si = tri.find_simplex(eval_points)
    valid = si >= 0

    if obs_bounds is not None and valid.any():
        lo = np.array([b[0] for b in obs_bounds])
        hi = np.array([b[1] for b in obs_bounds])
        in_obs = np.all((nodes >= lo) & (nodes <= hi), axis=1)  # (n_nodes,)
        # Mask out eval points whose triangle contains any buffer vertex
        vtx_in_obs = in_obs[tri.simplices[si[valid]]]           # (n_valid, d+1)
        valid_idx  = np.where(valid)[0]
        valid[valid_idx[~vtx_in_obs.all(axis=1)]] = False

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
        bary_last = 1.0 - bary_rest.sum(axis=1, keepdims=True)  # λ for simplices[idx, d]
        bary      = np.hstack([bary_rest, bary_last])            # (n_valid, d+1)

        vtx = tri.simplices[idx]                               # (n_valid, d+1)

        # eta(s) = Σ_i x_i phi_i(s) — interpolated log-intensity field
        eta = (x_samples[:, vtx] * bary[None, :, :]).sum(axis=-1)  # (S, n_valid)
        result[:, valid] = np.exp(mu_samples[:, None] + eta)

    return result  # (S, n_eval), NaN for points outside the triangulation
