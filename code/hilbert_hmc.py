"""
Hilbert-space HMC kernel, implemented as a BlackJAX-compatible sampler.

Velocity parameterisation
--------------------------
We work in (x, v) coordinates where  v = Q⁻¹ p  and  p ~ N(0, Q)  is the
canonical momentum.  In these coordinates:

    H(x, v) = U(x) + K(v)
    U(x)    = −log L(x) + ½ xᵀ Q x        (prior potential + neg log-likelihood)
    K(v)    = ½ vᵀ Q v,    v ~ N(0, Q⁻¹)  (kinetic energy, Q-weighted norm)

Hamilton's equations in (x, v) become:

    ẋ = v
    Q v̇ = −∇U(x)    →    v̇ = −Q⁻¹ ∇U(x)

Leapfrog integrator (L steps, step size ε):

    v ← v − (ε/2) Q⁻¹ ∇U(x)          half velocity kick   [CG solve]
    for i = 1 … L−1:
        x ← x + ε v                    position drift        [O(N)]
        v ← v − ε Q⁻¹ ∇U(x)           full velocity kick   [CG solve]
    x ← x + ε v                        final drift
    v ← v − (ε/2) Q⁻¹ ∇U(x)          final half kick       [CG solve]

Advantages over the (x, p) form:
  • K(v) = ½ vᵀ Q v requires only Q_matvec — no solve.
  • Velocity refresh v ~ N(0, Q⁻¹) is the same CG solve as prior sampling,
    so no Cholesky factorisation or dense matrix is ever stored.
  • All O(N²)–O(N³) dense-algebra is replaced by O(N × CG-iters) sparse work.

Reference:
    Beskos, Pinski, Sanz-Serna, Stuart (2011). "Hybrid Monte Carlo on Hilbert
    spaces." Stochastic Processes and their Applications 121(10), 2201–2230.
"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class HilbertHMCState(NamedTuple):
    position: jnp.ndarray
    potential: float  # U(x) cached to avoid recomputation at each MH step


class HilbertHMCInfo(NamedTuple):
    acceptance_rate: float
    is_accepted: bool
    energy_change: float  # ΔH; near-zero → well-tuned step size


def build_kernel(
    loglikelihood_fn: Callable,
    Q_matvec: Callable,
    Q_solve: Callable,
    velocity_sample_fn: Callable,
    step_size: float,
    n_leapfrog: int,
):
    grad_ll = jax.grad(loglikelihood_fn)

    def potential(x):
        # U(x) = -log L(x) + ½ xᵀ Q x
        return -loglikelihood_fn(x) + 0.5 * jnp.dot(x, Q_matvec(x))

    def kinetic(v):
        # K(v) = ½ vᵀ Q v  — only needs Q_matvec, no solve
        return 0.5 * jnp.dot(v, Q_matvec(v))

    def grad_potential(x):
        # ∇U(x) = Q x − ∇ log L(x)
        return Q_matvec(x) - grad_ll(x)

    def leapfrog(x, v):
        # Half velocity kick: v -= (ε/2) Q⁻¹ ∇U(x)
        v = v - 0.5 * step_size * Q_solve(grad_potential(x))

        # n_leapfrog - 1 full drift-kick pairs
        def body(_, state):
            x, v = state
            x = x + step_size * v                             # drift  (no solve)
            v = v - step_size * Q_solve(grad_potential(x))   # full kick (CG)
            return x, v

        x, v = jax.lax.fori_loop(0, n_leapfrog - 1, body, (x, v))

        # Final drift + half velocity kick
        x = x + step_size * v
        v = v - 0.5 * step_size * Q_solve(grad_potential(x))
        return x, v

    def kernel(rng_key, state: HilbertHMCState):
        v_key, accept_key = jax.random.split(rng_key)

        # Refresh velocity: v ~ N(0, Q⁻¹)  [same distribution as the field prior]
        v = velocity_sample_fn(v_key)

        x = state.position
        H_curr = state.potential + kinetic(v)

        x_prop, v_prop = leapfrog(x, v)
        U_prop = potential(x_prop)
        H_prop = U_prop + kinetic(v_prop)

        dH = H_prop - H_curr
        accepted = jnp.log(jax.random.uniform(accept_key)) < -dH

        new_position = jax.lax.cond(accepted, lambda: x_prop, lambda: x)
        new_potential = jax.lax.cond(accepted, lambda: U_prop, lambda: state.potential)

        return (
            HilbertHMCState(position=new_position, potential=new_potential),
            HilbertHMCInfo(
                acceptance_rate=jnp.minimum(1.0, jnp.exp(-dH)),
                is_accepted=accepted,
                energy_change=dH,
            ),
        )

    return kernel, potential


class SamplingAlgorithm(NamedTuple):
    init: Callable
    step: Callable


def hilbert_hmc(
    loglikelihood_fn: Callable,
    Q_matvec: Callable,
    Q_solve: Callable,
    velocity_sample_fn: Callable,
    step_size: float,
    n_leapfrog: int,
) -> SamplingAlgorithm:
    """
    Hilbert-space HMC sampler (velocity parameterisation).

    Parameters
    ----------
    loglikelihood_fn : x -> scalar
        Log-likelihood log L(x) only — do NOT include the prior log-density.
    Q_matvec : x -> x
        Applies the prior precision: b = Q x.  Must be JAX-traceable.
        Used for ∇U(x), U(x), and K(v) = ½ vᵀ Q v — no matrix inverse needed.
    Q_solve : b -> x
        Solves Q x = b, i.e. returns Q⁻¹ b.  Must be JAX-traceable and
        JIT-compatible (e.g. jax.scipy.sparse.linalg.cg wrapped in jax.jit).
        Called (L+1) times per HMC step for the leapfrog velocity updates.
    velocity_sample_fn : rng_key -> v
        Returns v ~ N(0, Q⁻¹) — equivalently, solves Q v = z for z ~ N(0, I).
        For a CG-based solver this is just Q_solve applied to standard normals,
        which is the same as prior_sample_fn.
    step_size : float
        Leapfrog step size ε.  Tune so ΔH stays small and acceptance ≈ 65–80 %.
    n_leapfrog : int
        Number of leapfrog steps L per proposal.  Larger L → less correlated
        draws but more gradient evaluations and CG solves per step.

    Returns
    -------
    SamplingAlgorithm
        .init(position)       -> HilbertHMCState
        .step(rng_key, state) -> (HilbertHMCState, HilbertHMCInfo)
    """
    kernel, potential_fn = build_kernel(
        loglikelihood_fn, Q_matvec, Q_solve, velocity_sample_fn, step_size, n_leapfrog
    )

    def init(position: jnp.ndarray) -> HilbertHMCState:
        return HilbertHMCState(
            position=position,
            potential=potential_fn(position),
        )

    return SamplingAlgorithm(init=init, step=kernel)
