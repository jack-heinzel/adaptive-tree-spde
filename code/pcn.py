"""
Preconditioned Crank-Nicolson (pCN) sampler, implemented as a BlackJAX-compatible kernel.

For targets  π(x) ∝ L(x) · N(x; 0, Q^{-1}),  the proposal

    x' = √(1-β²) x + β ξ,    ξ ~ N(0, Q^{-1})

leaves the prior N(0, Q^{-1}) invariant, so only the likelihood ratio enters the
acceptance probability:

    α = min(1, L(x') / L(x))

This makes the acceptance rate scale-free in the dimension of x (dimension-robust),
unlike random-walk MH whose optimal step size shrinks as O(n^{-1/2}).

Reference:
    Cotter, Roberts, Stuart, White (2013). "MCMC Methods for Functions:
    Modifying Old Algorithms to Make Them Faster." Stat. Sci. 28(3), 424-446.
"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class PCNState(NamedTuple):
    position: jnp.ndarray
    loglikelihood: float


class PCNInfo(NamedTuple):
    acceptance_rate: float  # min(1, L'/L) — useful for tuning β
    is_accepted: bool


def build_kernel(
    loglikelihood_fn: Callable,
    prior_sample_fn: Callable,
    beta: float,
) -> Callable:
    scale = jnp.sqrt(1.0 - beta**2)

    def kernel(rng_key, state: PCNState):
        prior_key, accept_key = jax.random.split(rng_key)

        xi = prior_sample_fn(prior_key)
        x_prop = scale * state.position + beta * xi

        ll_prop = loglikelihood_fn(x_prop)
        log_alpha = ll_prop - state.loglikelihood

        accepted = jnp.log(jax.random.uniform(accept_key)) < log_alpha
        new_position = jax.lax.cond(accepted, lambda: x_prop, lambda: state.position)
        new_ll = jax.lax.cond(accepted, lambda: ll_prop, lambda: state.loglikelihood)

        return (
            PCNState(position=new_position, loglikelihood=new_ll),
            PCNInfo(
                acceptance_rate=jnp.minimum(1.0, jnp.exp(log_alpha)),
                is_accepted=accepted,
            ),
        )

    return kernel


class SamplingAlgorithm(NamedTuple):
    init: Callable
    step: Callable


def pcn(
    loglikelihood_fn: Callable,
    prior_sample_fn: Callable,
    beta: float,
) -> SamplingAlgorithm:
    """
    Preconditioned Crank-Nicolson sampler.

    Parameters
    ----------
    loglikelihood_fn : x -> scalar
        Log-likelihood log L(x) only. Do NOT include the prior log-density here —
        it cancels in the acceptance ratio by construction.
    prior_sample_fn : rng_key -> x
        Returns an independent draw ξ ~ N(0, Q^{-1}) from the prior.
        Must be JAX-traceable so it can be JIT-compiled inside jax.lax.scan.
    beta : float in (0, 1]
        Step-size parameter.  β=1 → independent prior draws; β→0 → random-walk MH.
        Tune so the acceptance rate is roughly 20-40 %.

    Returns
    -------
    SamplingAlgorithm
        .init(position)          -> PCNState
        .step(rng_key, state)    -> (PCNState, PCNInfo)
    """
    kernel = build_kernel(loglikelihood_fn, prior_sample_fn, beta)

    def init(position: jnp.ndarray) -> PCNState:
        return PCNState(
            position=position,
            loglikelihood=loglikelihood_fn(position),
        )

    return SamplingAlgorithm(init=init, step=kernel)
