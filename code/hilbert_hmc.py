"""
Hilbert-space HMC and NUTS kernels (velocity parameterisation).

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

Single leapfrog step (step size ε):

    v ← v − (ε/2) Q⁻¹ ∇U(x)          half velocity kick   [CG solve]
    x ← x + ε v                        position drift        [O(N)]
    v ← v − (ε/2) Q⁻¹ ∇U(x)          final half kick       [CG solve]

Advantages over the (x, p) form:
  • K(v) = ½ vᵀ Q v requires only Q_matvec — no solve.
  • Velocity refresh v ~ N(0, Q⁻¹) is one CG solve — same as prior sampling.
  • All O(N²)–O(N³) dense algebra is replaced by O(N × CG-iters) sparse work.

Step size and mass matrix during sampling
------------------------------------------
step_size is a plain Python attribute; update it between calls to .step() to
adapt on the fly (e.g. during a warmup phase).

Q acts as both the prior precision AND the HMC mass matrix.  Changing Q
mid-chain changes the target distribution — do this only in warmup, then fix.
To separate the mass matrix from the prior, pass a different Q_matvec / Q_solve
to the constructor and construct a new sampler; the state position carries over.

References:
    Beskos, Pinski, Sanz-Serna, Stuart (2011). Hybrid Monte Carlo on Hilbert
    spaces.  Stochastic Processes and their Applications 121(10), 2201–2230.

    Hoffman, Gelman (2014). The No-U-Turn Sampler.
    Journal of Machine Learning Research 15, 1593–1623.
"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


# ── shared state / info types ──────────────────────────────────────────────────

class HilbertHMCState(NamedTuple):
    position: jnp.ndarray
    potential: float  # U(x) cached to avoid recomputation at the MH step


class HilbertHMCInfo(NamedTuple):
    acceptance_rate: float
    is_accepted: bool
    energy_change: float   # ΔH; near-zero → well-tuned step size


class HilbertNUTSInfo(NamedTuple):
    acceptance_rate: float  # mean min(1, exp(−ΔH)) over trajectory leaves
    is_accepted: bool       # always True for NUTS (slice sampling handles acceptance)
    energy_change: float    # U(x_prop) − U(x_curr)
    tree_depth: int         # depth at which U-turn fired (or max_depth if not)
    n_leapfrog: int         # total leapfrog steps taken this iteration


class SamplingAlgorithm(NamedTuple):
    init: Callable
    step: Callable


# ── HilbertHMC — fixed trajectory length ──────────────────────────────────────

class HilbertHMC:
    """
    Hilbert-space HMC with a fixed number of leapfrog steps.

    Parameters
    ----------
    loglikelihood_fn : x -> scalar
        Log L(x) only — do NOT include the prior log-density.
    Q_matvec : x -> x
        Applies prior precision: b = Q x.  Must be JAX-traceable.
    Q_solve : b -> x
        Solves Q x = b (e.g. CG).  Called L+1 times per step.
    velocity_sample_fn : rng_key -> v
        Draws v ~ N(0, Q⁻¹), typically Q_solve applied to N(0, I) noise.
    step_size : float
        Leapfrog step size ε.  Mutable — update between .step() calls to adapt.
    n_leapfrog : int
        Number of leapfrog steps per proposal.
    """

    def __init__(
        self,
        loglikelihood_fn: Callable,
        Q_matvec: Callable,
        Q_solve: Callable,
        velocity_sample_fn: Callable,
        step_size: float,
        n_leapfrog: int,
    ):
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.velocity_sample_fn = velocity_sample_fn
        self._Q_matvec = Q_matvec
        self._Q_solve = Q_solve

        grad_ll = jax.grad(loglikelihood_fn)

        def potential(x):
            return -loglikelihood_fn(x) + 0.5 * jnp.dot(x, Q_matvec(x))

        def kinetic(v):
            return 0.5 * jnp.dot(v, Q_matvec(v))

        def grad_potential(x):
            return Q_matvec(x) - grad_ll(x)

        def leapfrog_step(x, v, eps):
            v = v - 0.5 * eps * Q_solve(grad_potential(x))
            x = x + eps * v
            v = v - 0.5 * eps * Q_solve(grad_potential(x))
            return x, v

        self._potential = potential
        self._kinetic = kinetic
        self._leapfrog_step = leapfrog_step
        self._grad_potential = grad_potential

    def _leapfrog(self, x, v):
        """Run self.n_leapfrog steps using the velocity-Störmer-Verlet scheme."""
        eps = self.step_size
        v = v - 0.5 * eps * self._Q_solve(self._grad_potential(x))

        def body(_, state):
            x, v = state
            x = x + eps * v
            v = v - eps * self._Q_solve(self._grad_potential(x))
            return x, v

        x, v = jax.lax.fori_loop(0, self.n_leapfrog - 1, body, (x, v))
        x = x + eps * v
        v = v - 0.5 * eps * self._Q_solve(self._grad_potential(x))
        return x, v

    def init(self, position: jnp.ndarray) -> HilbertHMCState:
        return HilbertHMCState(
            position=position,
            potential=self._potential(position),
        )

    def step(self, rng_key, state: HilbertHMCState):
        v_key, accept_key = jax.random.split(rng_key)
        v = self.velocity_sample_fn(v_key)
        x = state.position
        H_curr = state.potential + self._kinetic(v)

        x_prop, v_prop = self._leapfrog(x, v)
        U_prop = self._potential(x_prop)
        H_prop = U_prop + self._kinetic(v_prop)

        dH = H_prop - H_curr
        accepted = jnp.log(jax.random.uniform(accept_key)) < -dH

        new_pos = jax.lax.cond(accepted, lambda: x_prop, lambda: x)
        new_pot = jax.lax.cond(accepted, lambda: U_prop, lambda: state.potential)

        return (
            HilbertHMCState(position=new_pos, potential=new_pot),
            HilbertHMCInfo(
                acceptance_rate=jnp.minimum(1.0, jnp.exp(-dH)),
                is_accepted=accepted,
                energy_change=dH,
            ),
        )

    def as_sampling_algorithm(self) -> SamplingAlgorithm:
        return SamplingAlgorithm(init=self.init, step=self.step)


# ── HilbertNUTS — No-U-Turn Sampler ───────────────────────────────────────────

class HilbertNUTS(HilbertHMC):
    """
    Hilbert-space NUTS (No-U-Turn Sampler).

    Inherits all leapfrog dynamics from HilbertHMC.  Replaces the fixed-L
    trajectory with a binary doubling tree that expands until the No-U-Turn
    criterion (expressed in terms of the Q-weighted momentum) fires:

        (x⁺ − x⁻) · Q v⁻ ≥ 0   AND   (x⁺ − x⁻) · Q v⁺ ≥ 0

    The proposal is drawn from valid trajectory leaves via progressive
    multinomial sampling — no additional Metropolis step is needed.

    JAX implementation notes
    ------------------------
    The outer doubling loop is a Python for-loop (unrolled at trace time;
    max_depth is a compile-time constant).  Each subtree is built via
    jax.lax.scan.  jax.lax.cond gates the subtree build once the U-turn flag
    goes to zero, so leapfrog steps stop executing after the U-turn fires.

    Parameters
    ----------
    loglikelihood_fn, Q_matvec, Q_solve, velocity_sample_fn, step_size :
        Same as HilbertHMC.
    max_depth : int
        Maximum tree depth.  Trajectory has at most 2^max_depth leapfrog steps.
        Default 10 → 1024 steps maximum.
    delta_energy_max : float
        A leaf is declared invalid if H(x,v) > H_init + delta_energy_max.
        Protects against floating-point blowup.  Default 1000.
    """

    def __init__(
        self,
        loglikelihood_fn: Callable,
        Q_matvec: Callable,
        Q_solve: Callable,
        velocity_sample_fn: Callable,
        step_size: float,
        max_depth: int = 10,
        delta_energy_max: float = 1000.0,
    ):
        super().__init__(
            loglikelihood_fn, Q_matvec, Q_solve,
            velocity_sample_fn, step_size, n_leapfrog=1,
        )
        self.max_depth = max_depth
        self.delta_energy_max = delta_energy_max

    def _no_u_turn(self, x_l, v_l, x_r, v_r) -> jnp.ndarray:
        """True iff the trajectory has not yet turned back (Q-weighted criterion)."""
        dx = x_r - x_l
        return (
            jnp.dot(dx, self._Q_matvec(v_l)) >= 0.0
        ) & (
            jnp.dot(dx, self._Q_matvec(v_r)) >= 0.0
        )

    def _build_subtree(self, key, x0, v0, direction, n_steps, log_slice, H_init):
        """
        Walk n_steps leapfrog steps from (x0, v0) in ±direction.

        Accumulates a proposal via progressive multinomial sampling: the k-th
        valid leaf replaces the running proposal with probability 1/k, yielding
        a uniformly distributed draw over all valid leaves.

        Also accumulates the sum of min(1, exp(−ΔH)) over all leaves for the
        NUTS acceptance diagnostic used in dual-averaging step-size tuning.

        Returns
        -------
        x_end, v_end    trajectory endpoint after n_steps
        x_prop          proposed sample (uniform over valid leaves)
        n_valid         number of valid leaves visited
        accept_sum      sum of min(1, exp(−ΔH_leaf)) over all leaves
        key             updated rng key
        """
        eps = direction * self.step_size

        def body(carry, _):
            x, v, x_prop, n_valid, accept_sum, key = carry
            x_new, v_new = self._leapfrog_step(x, v, eps)
            H_new = self._potential(x_new) + self._kinetic(v_new)
            dH = H_new - H_init

            in_slice = (-H_new >= log_slice) & (dH < self.delta_energy_max)
            n_new = n_valid + jnp.where(in_slice, 1, 0)

            # Progressive Metropolis: replace running proposal with prob 1/n_new
            key, subkey = jax.random.split(key)
            update = in_slice & (
                jax.random.uniform(subkey) < 1.0 / jnp.maximum(n_new.astype(jnp.float32), 1.0)
            )
            x_prop = jax.lax.cond(update, lambda: x_new, lambda: x_prop)

            accept_sum = accept_sum + jnp.minimum(1.0, jnp.exp(-dH))

            return (x_new, v_new, x_prop, n_new, accept_sum, key), None

        init = (
            x0,
            v0,
            x0,                                     # x_prop initialised to x0
            jnp.zeros((), dtype=jnp.int32),         # n_valid
            jnp.zeros((), dtype=jnp.float32),       # accept_sum
            key,
        )
        (x_end, v_end, x_prop, n_valid, accept_sum, key), _ = jax.lax.scan(
            body, init, None, length=n_steps)

        return x_end, v_end, x_prop, n_valid, accept_sum, key

    def step(self, rng_key, state: HilbertHMCState):
        key = rng_key
        key, v_key, u_key = jax.random.split(key, 3)

        v0 = self.velocity_sample_fn(v_key)
        x0 = state.position
        H0 = state.potential + self._kinetic(v0)

        # Slice threshold: log u,  u ~ Uniform(0, exp(−H0))
        log_slice = jnp.log(jax.random.uniform(u_key)) - H0

        x_l = x_r = x0
        v_l = v_r = v0
        x_prop = x0
        n_valid     = jnp.ones((),  dtype=jnp.int32)    # initial state is valid
        accept_sum  = jnp.ones((),  dtype=jnp.float32)  # min(1, exp(0)) = 1
        s           = jnp.ones((),  dtype=jnp.int32)    # continue flag
        tree_depth  = jnp.zeros((), dtype=jnp.int32)
        n_leapfrog  = jnp.zeros((), dtype=jnp.int32)

        for j in range(self.max_depth):
            n_steps = 2**j
            key, dir_key, accept_key, sub_key = jax.random.split(key, 4)
            direction = jnp.where(jax.random.uniform(dir_key) < 0.5, 1, -1)

            # Choose starting endpoint based on direction
            x_start = jax.lax.cond(direction > 0, lambda: x_r, lambda: x_l)
            v_start = jax.lax.cond(direction > 0, lambda: v_r, lambda: v_l)

            # Build subtree only while s=1 (jax.lax.cond short-circuits at runtime)
            def do_build(_):
                return self._build_subtree(
                    sub_key, x_start, v_start, direction, n_steps, log_slice, H0)

            def skip_build(_):
                # Return dummy result with identical shapes
                dummy_n    = jnp.zeros((), dtype=jnp.int32)
                dummy_asum = jnp.zeros((), dtype=jnp.float32)
                return x_start, v_start, x_prop, dummy_n, dummy_asum, sub_key

            x_end, v_end, x_sub_prop, n_sub, asum_sub, key = jax.lax.cond(
                s == 1, do_build, skip_build, None)

            # Update trajectory endpoints
            x_r = jax.lax.cond(direction > 0, lambda: x_end, lambda: x_r)
            v_r = jax.lax.cond(direction > 0, lambda: v_end, lambda: v_r)
            x_l = jax.lax.cond(direction > 0, lambda: x_l,   lambda: x_end)
            v_l = jax.lax.cond(direction > 0, lambda: v_l,   lambda: v_end)

            # Multinomial accept: take subtree proposal with prob n_sub / n_valid
            accept_sub = (s == 1) & (
                jax.random.uniform(accept_key)
                < n_sub.astype(jnp.float32) / jnp.maximum(n_valid.astype(jnp.float32), 1.0)
            )
            x_prop = jax.lax.cond(accept_sub, lambda: x_sub_prop, lambda: x_prop)

            n_valid    = n_valid   + n_sub   * s
            accept_sum = accept_sum + asum_sub * s.astype(jnp.float32)
            n_leapfrog = n_leapfrog + jnp.where(s == 1, n_steps, 0)

            # No-U-Turn check; update s and record depth when it fires
            u_turn_ok = self._no_u_turn(x_l, v_l, x_r, v_r)
            s_new = s & jnp.where(u_turn_ok, 1, 0)
            # tree_depth = j+1 at the first iteration where s drops to 0
            tree_depth = jnp.where((s == 1) & (s_new == 0), j + 1, tree_depth)
            s = s_new

        # If U-turn never fired, depth = max_depth
        tree_depth = jnp.where(s == 1, self.max_depth, tree_depth)

        U_prop = self._potential(x_prop)
        total_leaves = n_leapfrog + 1  # +1 for the initial position

        return (
            HilbertHMCState(position=x_prop, potential=U_prop),
            HilbertNUTSInfo(
                acceptance_rate=accept_sum / total_leaves.astype(jnp.float32),
                is_accepted=jnp.ones((), dtype=jnp.bool_),
                energy_change=U_prop - state.potential,
                tree_depth=tree_depth,
                n_leapfrog=n_leapfrog,
            ),
        )

    def as_sampling_algorithm(self) -> SamplingAlgorithm:
        return SamplingAlgorithm(init=self.init, step=self.step)


# ── factory functions (backward-compatible) ────────────────────────────────────

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

    Returns a SamplingAlgorithm with .init and .step for use in jax.lax.scan
    loops.  For direct access to the underlying class (e.g. to update
    step_size between warmup and sampling), use HilbertHMC directly.
    """
    return HilbertHMC(
        loglikelihood_fn, Q_matvec, Q_solve,
        velocity_sample_fn, step_size, n_leapfrog,
    ).as_sampling_algorithm()


def hilbert_nuts(
    loglikelihood_fn: Callable,
    Q_matvec: Callable,
    Q_solve: Callable,
    velocity_sample_fn: Callable,
    step_size: float,
    max_depth: int = 10,
    delta_energy_max: float = 1000.0,
) -> SamplingAlgorithm:
    """
    Hilbert-space NUTS sampler (velocity parameterisation).

    Returns a SamplingAlgorithm with .init and .step.  For access to
    max_depth, delta_energy_max, or step_size, use HilbertNUTS directly.

    Tuning guidance
    ---------------
    - Target acceptance_rate (HilbertNUTSInfo field) ≈ 0.65–0.85.
      If lower, decrease step_size; if higher with shallow trees, increase it.
    - Target tree_depth ≈ 3–7.  If depth always hits max_depth, increase it
      or reduce step_size.  If depth is always 1–2, step_size may be too small.
    - For step-size dual averaging, use acceptance_rate as the target statistic
      (it equals the mean min(1, exp(−ΔH)) over the trajectory, which is the
      correct NUTS target for the Nesterov dual-averaging scheme).
    """
    return HilbertNUTS(
        loglikelihood_fn, Q_matvec, Q_solve,
        velocity_sample_fn, step_size, max_depth, delta_energy_max,
    ).as_sampling_algorithm()
