"""hyperparameter optimization with a fixed frequency grid.

Standard EFGP hyperparameter hyperparmater tuning 
(:func:`efgp_jax.optimize.optimize_hyperparameters`)
uses a new set of frequencies at every optimizer step.  
The frequency grid is regenerated from the current hyperparameters, so the 
frequencies move and all of
the O(N) work — NUFFT phases, the Toeplitz Gram ``E^H E``, and ``E^H y`` — has to
be recomputed from scratch, and the shapes change so nothing can be JIT-compiled
or differentiated across iterations.

This module implements an algorithm for hyperparameter tuning that keeps the same 
frequency grid across hyperparameter tuning.  
That is, ``Phi = E diag(w)`` where
``E_{nj} = exp(2 pi i xi_j . x_n)`` is fixed and ``w(theta) = sqrt(S_theta(xi) h^d)``
is the only hyperparameter-dependent part

* ``G = E^H E`` (the Toeplitz Gram) and ``b = E^H y`` depend only on the fixed
  frequencies and the data, so they are precomputed once (the O(N) work).
* Each optimizer step recomputes only the length-``M`` weight vector ``w(theta)``
  and solves the ``M x M`` weight-space system indepedent of ``N``.

Because the frequencies are fixed, the
marginal likelihood becomes a fixed-shape, smooth function of ``theta``.  We form
it exactly in weight space via Woodbury and take the Cholesky of the ``M x M``
system, so ``jax.value_and_grad`` gives exact gradients.  
The cost is O(N + M^3) making it well suited to large ``N`` and moderate ``M``.

Weight-space marginal likelihood
--------------------------------
With ``Sigma = Phi Phi^H + sigma^2 I`` and ``A = I_M + Phi^H Phi / sigma^2 =
I_M + diag(w) G diag(w) / sigma^2`` (``w`` real):

    logdet Sigma = logdet A + N log sigma^2
    y^T Sigma^{-1} y = (1/sigma^2) [ y^T y - (1/sigma^2) c^H A^{-1} c ],  c = w * b

    NLL = 0.5 [ y^T Sigma^{-1} y + logdet Sigma + N log(2 pi) ]
"""

from typing import Optional, Tuple

import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from scipy.optimize import minimize as sp_minimize

from .kernels import Kernel, SE, Matern
from .quadrature import get_xis
from .efgp import _parse_domain, _compute_convolution_vector, _is_anisotropic
from .toeplitz import make_toeplitz, toeplitz_apply
from .nufft import _make_phi, nufft_type1, _cmplx


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

def _kernel_factory(kernel0: Kernel):
    """Return a function ``(l, var) -> Kernel`` matching ``kernel0``'s type."""
    if isinstance(kernel0, Matern):
        nu, dim = kernel0.nu, kernel0.dim
        return lambda l, var: Matern(lengthscale=l, variance=var, dim=dim, nu=nu)
    dim = kernel0.dim
    return lambda l, var: SE(lengthscale=l, variance=var, dim=dim)


# ---------------------------------------------------------------------------
# Fixed frequency grid
# ---------------------------------------------------------------------------

def build_fixed_grid(
    kernel0: Kernel,
    domain,
    eps: float,
    l_bounds: Tuple[float, float],
    *,
    use_integral: bool = True,
) -> dict:
    """Build a single frequency grid valid across a lengthscale range.

    The grid truncation is set by the *relative* decay of the spectral density,
    k_hat(0) / k_hat(xi_max) ~ 1/eps.  Variance is a multiplicative prefactor that
    cancels in that ratio, so the grid depends only on the lengthscale (not the
    variance).  Spacing ``h`` shrinks with larger lengthscale while the frequency
    extent grows with smaller lengthscale, so we take the finest spacing and
    widest extent over the endpoints of ``l_bounds`` -- a fixed grid at least
    as accurate as any lengthscale in range.

    Parameters
    ----------
    kernel0 : Kernel
        Isotropic ``SE`` or ``Matern`` (sets kernel type / ``nu`` / ``dim``).
    domain : tuple
        Domain spec, ``(lo, hi)`` or ``((lo1, hi1), ...)``.
    eps : float
        Spectral truncation tolerance.
    l_bounds : (float, float)
        Lengthscale search range.
    use_integral : bool
        Passed through to :func:`get_xis`.

    Returns
    -------
    dict with keys ``xis`` (M, d), ``h`` (float), ``mtot`` (int), ``OUT``
    (tuple), ``M`` (int), ``xcen`` (d,), ``L`` (float), ``d`` (int).
    """
    if _is_anisotropic(kernel0):
        raise NotImplementedError(
            "build_fixed_grid supports isotropic kernels only; "
            "anisotropic (per-dimension lengthscale) is not yet handled."
        )

    d = kernel0.dim
    L, xcen = _parse_domain(domain, d)
    make_kernel = _kernel_factory(kernel0)
    # Grid is variance-independent; hold variance at the initial value as a
    # fixed reference while sweeping only the lengthscale endpoints.
    var_ref = float(jnp.asarray(kernel0.variance))

    l_lo, l_hi = float(l_bounds[0]), float(l_bounds[1])

    h_fine = math.inf
    xi_max = 0.0
    for l_c in (l_lo, l_hi):
        _, h_c, mtot_c = get_xis(make_kernel(l_c, var_ref), eps, L,
                                 use_integral=use_integral)
        h_c = float(h_c)
        h_fine = min(h_fine, h_c)
        xi_max = max(xi_max, ((int(mtot_c) - 1) // 2) * h_c)

    hm = int(math.ceil(xi_max / h_fine))
    xis_1d = jnp.arange(-hm, hm + 1) * h_fine
    mtot = 2 * hm + 1

    if d == 1:
        xis = xis_1d.reshape(-1, 1)
    else:
        grids = jnp.meshgrid(*[xis_1d for _ in range(d)], indexing="ij")
        xis = jnp.stack([g.ravel() for g in grids], axis=-1)

    return {
        "xis": xis,
        "h": h_fine,
        "mtot": mtot,
        "OUT": (mtot,) * d,
        "M": mtot ** d,
        "xcen": xcen,
        "L": L,
        "d": d,
    }


# ---------------------------------------------------------------------------
# One-time, hyperparameter-independent precompute
# ---------------------------------------------------------------------------

def _precompute(x, y, grid, nufft_eps, cdtype):
    """Precompute the data-dependent, hyperparameter-free operators.

    Returns ``(G_dense, b, yTy, N)`` where ``G_dense`` is the dense ``M x M``
    Gram ``E^H E`` and ``b = E^H y``.
    """
    if x.ndim == 1:
        x = x[:, None]
    N = x.shape[0]
    d = grid["d"]
    OUT, h, xcen = grid["OUT"], grid["h"], grid["xcen"]

    # b = E^H y  (one type-1 NUFFT)
    phi = _make_phi(x, xcen, h)
    b = nufft_type1(phi, y.astype(cdtype), OUT, eps=nufft_eps).reshape(-1)

    # G = E^H E is (block-)Toeplitz; build it from the NUFFT convolution vector,
    # then densify by applying the operator to the identity (columns of G).
    m_conv = (grid["mtot"] - 1) // 2
    v_kernel = _compute_convolution_vector(m_conv, x, h, xcen, nufft_eps).astype(cdtype)
    toeplitz_op = make_toeplitz(v_kernel, force_pow2=True)

    M = grid["M"]
    cols = toeplitz_apply(toeplitz_op, jnp.eye(M, dtype=cdtype))  # cols[i] = G e_i
    G_dense = cols.T
    # symmetrize away NUFFT/FFT round-off so the Cholesky sees an exact Hermitian
    G_dense = 0.5 * (G_dense + G_dense.conj().T)

    yTy = jnp.real(jnp.vdot(y.astype(cdtype), y.astype(cdtype)))
    return G_dense, b, yTy, N


# ---------------------------------------------------------------------------
# Differentiable weight-space negative log marginal likelihood
# ---------------------------------------------------------------------------

def _weightspace_nll(grid, G_dense, b, yTy, N, make_kernel, cdtype):
    """Return the (un-jitted) weight-space NLL as a function of ``log_theta``.

    ``log_theta = [log lengthscale, log variance, log sigma^2]``.  Kept separate
    from the jit wrapper so callers that rebuild the grid every step (e.g. a
    non-fixed-grid baseline) can differentiate it eagerly without recompiling.
    """
    xis = grid["xis"]
    hd = grid["h"] ** grid["d"]
    M = grid["M"]
    eyeM = jnp.eye(M, dtype=cdtype)
    log_two_pi = math.log(2 * math.pi)

    def nll(log_theta):
        l = jnp.exp(log_theta[0])
        var = jnp.exp(log_theta[1])
        sig2 = jnp.exp(log_theta[2])

        kernel = make_kernel(l, var)
        s = kernel.spectral_density(xis) * hd  # (M,) real >= 0
        # Safe sqrt: the fixed grid extends past the current bandwidth, so the
        # spectral density underflows to exactly 0 at far frequencies.  There
        # d/dtheta sqrt(s) = s'/(2 sqrt(s)) is 0/0 = NaN, which would poison the
        # gradient (the value sqrt(0)=0 is fine; only the derivative blows up).
        # Mask negligible weights to 0 *before* the sqrt so it never sees a zero.
        safe = s > 1e-16 * jnp.max(s)
        w = jnp.where(safe, jnp.sqrt(jnp.where(safe, s, 1.0)), 0.0)
        wc = w.astype(cdtype)

        A = eyeM + (wc[:, None] * G_dense * wc[None, :]) / sig2
        Lc = jnp.linalg.cholesky(A)  # A = Lc Lc^H, diag real > 0
        logdet_A = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diagonal(Lc))))

        c = wc * b
        z1 = jax.scipy.linalg.solve_triangular(Lc, c, lower=True)
        z = jax.scipy.linalg.solve_triangular(Lc.conj().T, z1, lower=False)
        quad = jnp.real(jnp.vdot(c, z))  # c^H A^{-1} c

        yT_Sinv_y = (yTy - quad / sig2) / sig2
        logdet_Sigma = logdet_A + N * jnp.log(sig2)
        return 0.5 * (yT_Sinv_y + logdet_Sigma + N * log_two_pi)

    return nll


def _make_nll(grid, G_dense, b, yTy, N, make_kernel, cdtype):
    """JIT-compiled value-and-grad of the weight-space NLL (fixed-grid path)."""
    nll = _weightspace_nll(grid, G_dense, b, yTy, N, make_kernel, cdtype)
    return jax.jit(jax.value_and_grad(nll))


# ---------------------------------------------------------------------------
# Optimizer driver
# ---------------------------------------------------------------------------

def optimize_hyperparameters_fixed_freq(
    x: Array,
    y: Array,
    kernel0: Kernel,
    sigmasq0: float,
    eps: float,
    *,
    domain=None,
    l_bounds: Optional[Tuple[float, float]] = None,
    use_integral: bool = True,
    nufft_eps: float = 6e-8,
    maxiter: int = 100,
    tol: float = 1e-6,
    verbose: bool = True,
) -> Tuple[Kernel, float, dict]:
    """Optimize GP hyperparameters on a fixed (shared) frequency grid.

    Builds one fixed frequency grid from ``l_bounds``, precomputes the
    data-dependent operators once, then minimizes the exact weight-space negative
    log marginal likelihood with L-BFGS-B using autodiff gradients.  ``l_bounds``
    both defines the grid and bounds the lengthscale, so every evaluated
    lengthscale stays within the grid's validity.  Variance and noise do not
    affect the grid, so they are optimized with wide internal bounds.

    Parameters
    ----------
    x : Array, shape (n,) or (n, d)
    y : Array, shape (n,)
    kernel0 : Kernel
        Initial isotropic ``SE`` or ``Matern`` (its ``nu``/``dim`` are fixed).
    sigmasq0 : float
        Initial noise variance.
    eps : float
        Spectral truncation tolerance for the grid.
    domain : tuple or None
        Domain spec; inferred from data bounds if None.
    l_bounds : (float, float)
        Lengthscale search range.  Required: defines the grid *and* bounds the
        lengthscale so every evaluated setting stays within the grid's validity.
        Variance and noise do not affect the grid, so they are optimized
        unbounded.
    use_integral, nufft_eps : see EFGP.
    maxiter : int
        Max L-BFGS-B iterations.
    tol : float
        Gradient-norm tolerance.
    verbose : bool

    Returns
    -------
    kernel : Kernel
        Optimized kernel (same subclass as ``kernel0``).
    sigmasq : float
        Optimized noise variance.
    info : dict
        Keys ``nll``, ``nfev``, ``success``, ``M`` (grid size).
    """
    if x.ndim == 1:
        x = x[:, None]

    if domain is None:
        domain = tuple(
            (float(jnp.min(x[:, i])), float(jnp.max(x[:, i])))
            for i in range(x.shape[1])
        )

    if l_bounds is None:
        raise ValueError(
            "l_bounds is required: it defines the fixed frequency grid (and "
            "bounds the lengthscale).  Pass an explicit (lo, hi) range."
        )

    l0 = float(jnp.asarray(kernel0.lengthscale))
    var0 = float(jnp.asarray(kernel0.variance))

    cdtype = _cmplx(x.dtype)

    grid = build_fixed_grid(kernel0, domain, eps, l_bounds,
                               use_integral=use_integral)
    G_dense, b, yTy, N = _precompute(x, y, grid, nufft_eps, cdtype)

    make_kernel = _kernel_factory(kernel0)
    value_and_grad = _make_nll(grid, G_dense, b, yTy, N, make_kernel, cdtype)

    theta0 = np.array([math.log(l0), math.log(var0), math.log(sigmasq0)])
    # Only the lengthscale is box-constrained (grid validity); variance and noise
    # do not affect the grid and are optimized unbounded.
    bounds = [
        (math.log(l_bounds[0]), math.log(l_bounds[1])),
        (None, None),
        (None, None),
    ]

    def objective(theta):
        val, grad = value_and_grad(jnp.asarray(theta))
        if verbose:
            l, var, s2 = np.exp(theta)
            print(f"  l={l:.4f}  var={var:.4f}  noise={s2:.4f}  "
                  f"NLL={float(val):.2f}")
        return float(val), np.asarray(grad, dtype=np.float64)

    res = sp_minimize(objective, theta0, method="L-BFGS-B", jac=True,
                      bounds=bounds, options={"maxiter": maxiter, "gtol": tol})

    kernel_final = make_kernel(float(np.exp(res.x[0])), float(np.exp(res.x[1])))
    sigmasq_final = float(np.exp(res.x[2]))

    info = {
        "nll": float(res.fun),
        "nfev": res.nfev,
        "success": bool(res.success),
        "M": grid["M"],
    }
    return kernel_final, sigmasq_final, info
