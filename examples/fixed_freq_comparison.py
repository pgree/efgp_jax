"""Fixed vs non-fixed frequency grid for hyperparameter tuning.

Both optimizers use EFGP and the M x M Cholesky weight-space marginal likelihood, 
with autodiff gradients. The only thing that differs is the frequency grid:

  * non-fixed : the grid is regenerated from the current hyperparameters at every
                optimizer step, so the NUFFT phases and the Gram E^H E / E^H y
                (the O(N) work) are rebuilt on every evaluation.
  * fixed     : one grid is used throughtout the optimization, and 
                the O(N) operations is done once and each step is just
                the M x M Cholesky.
"""

# =============================== CONFIG ================================
# The knobs worth changing live here.
N            = 1000          # number of training points
EPS          = 1e-4          # spectral truncation tolerance (smaller = more freqs)
MAXITER      = 40            # max L-BFGS-B iterations
DOMAIN       = (0.0, 10.0)   # spatial domain (spans ~many lengthscales)

# True (data-generating) hyperparameters
L_TRUE       = 0.15          # lengthscale
VAR_TRUE     = 2.0           # kernel variance (prefactor)
SIG2_TRUE    = 0.05          # observation noise variance

# Initial hyperparameters for the optimizer (deliberately wrong)
L_INIT       = 0.5
VAR_INIT     = 0.5
SIG2_INIT    = 0.2

# Lengthscale search box for the head-to-head fit.  This defines the fixed
# frequency grid AND bounds the lengthscale.  (Variance and noise are optimized
# unbounded -- they don't affect the grid.)
L_BOUNDS     = (0.05, 5.0)

SEED         = 0

# Lengthscale-bounds sweep (second part of the demo).
# For each factor f, the fixed method searches lengthscales in [L_TRUE/f, L_TRUE*f].
# Wider bounds -> the fixed grid must resolve a smaller l_min and a larger
# l_max, so M (and the per-step M^3 Cholesky cost) grows.  This sweep shows how
# the fixed method's runtime scales with the width of the lengthscale bounds.
L_BOUND_FACTORS = [2.0, 4.0, 8.0, 12.0]
# ======================================================================

import os
import warnings

# Quiet the noisy jax-finufft / XLA / OpenMP messages so the output is readable.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide XLA C++ ERROR logs
os.environ.setdefault("KMP_WARNINGS", "0")          # hide OpenMP warnings
warnings.filterwarnings("ignore")                   # hide Python DeprecationWarnings

import time
import math

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize as sp_minimize

from efgp_jax.kernels import SE
from efgp_jax.efgp import EFGP, _parse_domain
from efgp_jax.quadrature import get_xis
from efgp_jax.nufft import _cmplx
from efgp_jax.fixed_freq import (
    optimize_hyperparameters_fixed_freq,   # the fixed-grid method (in the package)
    _precompute,                           # shared: build E^H E and E^H y for a grid
    _weightspace_nll,                      # shared: exact Cholesky weight-space NLL
    _kernel_factory,
)


def optimize_nonfixed_cholesky(x, y, kernel0, sig0, eps, domain, maxiter):
    """Baseline: same exact Cholesky estimator, but the grid is rebuilt each step.

    This is the "non-fixed" side of the comparison.  It is intentionally the most
    favorable version of the non-fixed approach (eager autodiff, no recompilation
    penalty); the only handicap vs. the fixed method is that it cannot amortize
    the per-step grid construction and O(N) NUFFT work.
    """
    x2 = x[:, None] if x.ndim == 1 else x
    cdtype = _cmplx(x2.dtype)
    L, xcen = _parse_domain(domain, kernel0.dim)
    make_kernel = _kernel_factory(kernel0)
    n_eval = [0]

    def make_grid(kernel):
        xis_1d, h, mtot = get_xis(kernel, eps, L, use_integral=True)
        h, mtot = float(h), int(mtot)
        return {"xis": xis_1d.reshape(-1, 1), "h": h, "mtot": mtot,
                "OUT": (mtot,), "M": mtot, "xcen": xcen, "L": L, "d": 1}

    def objective(theta):
        n_eval[0] += 1
        l, var = float(np.exp(theta[0])), float(np.exp(theta[1]))
        grid = make_grid(make_kernel(l, var))                       # rebuilt every step
        G, b, yTy, Nn = _precompute(x2, y, grid, 6e-8, cdtype)      # O(N) every step
        nll = _weightspace_nll(grid, G, b, yTy, Nn, make_kernel, cdtype)
        val, grad = jax.value_and_grad(nll)(jnp.asarray(theta))     # eager autodiff
        return float(val), np.asarray(grad, dtype=np.float64)

    l0 = float(jnp.asarray(kernel0.lengthscale))
    var0 = float(jnp.asarray(kernel0.variance))
    theta0 = np.array([math.log(l0), math.log(var0), math.log(sig0)])
    r = 10.0
    bounds = [(math.log(l0 / r), math.log(l0 * r)),
              (math.log(var0 / r), math.log(var0 * r)),
              (math.log(sig0 * 1e-4), math.log(sig0 * 1e4))]
    res = sp_minimize(objective, theta0, method="L-BFGS-B", jac=True,
                      bounds=bounds, options={"maxiter": maxiter, "gtol": 1e-6})
    return (float(np.exp(res.x[0])), float(np.exp(res.x[1])),
            float(np.exp(res.x[2])), n_eval[0])


def sweep_lengthscale_bounds(x, y):
    """Show how the fixed method's runtime scales with the lengthscale bounds.

    Widening the lengthscale bounds enlarges the (variance-independent) fixed
    grid M.  The optimizer is started at the true hyperparameters so nfev is small
    and uniform across runs -- the time differences then reflect the grid size M
    (i.e. the O(M^3) Cholesky), not the convergence path.
    """
    kernel_start = SE(lengthscale=L_TRUE, variance=VAR_TRUE, dim=1)

    print("\n\nLengthscale-bounds sweep (fixed frequencies method)")
    print("-" * 70)
    print(f"{'factor':>7}{'l_bounds':>20}{'M':>7}{'time(s)':>9}{'nfev':>6}"
          f"{'l_recov':>10}")
    for fct in L_BOUND_FACTORS:
        l_bounds = (L_TRUE / fct, L_TRUE * fct)
        t = time.perf_counter()
        k, _, info = optimize_hyperparameters_fixed_freq(
            x, y, kernel_start, SIG2_TRUE, EPS, domain=DOMAIN,
            l_bounds=l_bounds, maxiter=MAXITER, verbose=False)
        dt = time.perf_counter() - t
        print(f"{fct:>7.0f}{f'[{l_bounds[0]:.4f}, {l_bounds[1]:.2f}]':>20}"
              f"{info['M']:>7}{dt:>9.2f}{info['nfev']:>6}"
              f"{float(k.lengthscale):>10.4f}", flush=True)
    print("-" * 70)


def main():
    # ----- generate data from a GP prior with known hyperparameters -----
    print(f"Generating N={N} points from a GP prior "
          f"(l={L_TRUE}, var={VAR_TRUE}, sig2={SIG2_TRUE})", flush=True)
    x = jnp.linspace(*DOMAIN, N)
    kernel_true = SE(lengthscale=L_TRUE, variance=VAR_TRUE, dim=1)
    f = EFGP(kernel_true, domain=DOMAIN, eps=1e-6).sample(x, key=jax.random.PRNGKey(SEED))
    y = f + math.sqrt(SIG2_TRUE) * jax.random.normal(jax.random.PRNGKey(SEED + 1), x.shape)

    kernel0 = SE(lengthscale=L_INIT, variance=VAR_INIT, dim=1)

    # ----- fixed grid (amortized, jitted) -----
    print("Fitting: fixed grid + Cholesky ...", flush=True)
    t = time.perf_counter()
    k_fix, s_fix, info = optimize_hyperparameters_fixed_freq(
        x, y, kernel0, SIG2_INIT, EPS, domain=DOMAIN, l_bounds=L_BOUNDS,
        maxiter=MAXITER, verbose=False)
    dt_fix = time.perf_counter() - t
    l_fix, v_fix = float(k_fix.lengthscale), float(k_fix.variance)

    # ----- non-fixed grid (rebuilt each step), same estimator -----
    print("Fitting: non-fixed grid + Cholesky ...", flush=True)
    t = time.perf_counter()
    l_nf, v_nf, s_nf, nfev_nf = optimize_nonfixed_cholesky(
        x, y, kernel0, SIG2_INIT, EPS, DOMAIN, MAXITER)
    dt_nf = time.perf_counter() - t

    # ----- report -----
    max_rel = max(abs(l_nf - l_fix) / l_fix,
                  abs(v_nf - v_fix) / v_fix,
                  abs(s_nf - s_fix) / s_fix)

    print(f"\nN = {N}   eps = {EPS}   maxiter = {MAXITER}   "
          f"(truth: l={L_TRUE} var={VAR_TRUE} sig2={SIG2_TRUE})")
    print("-" * 70)
    print(f"{'variant':<26}{'time(s)':>9}{'nfev':>6}{'l':>9}{'var':>8}{'sig2':>9}")
    print(f"{'non-fixed grid + chol':<26}{dt_nf:>9.2f}{nfev_nf:>6}"
          f"{l_nf:>9.4f}{v_nf:>8.3f}{s_nf:>9.4f}")
    print(f"{'fixed grid + chol (M='+str(info['M'])+')':<26}{dt_fix:>9.2f}"
          f"{info['nfev']:>6}{l_fix:>9.4f}{v_fix:>8.3f}{s_fix:>9.4f}")
    print("-" * 70)
    print(f"speedup: {dt_nf / dt_fix:.1f}x     "
          f"max relative param disagreement: {max_rel:.1e}")

    # ----- how does the fixed method's timing scale with lengthscale bounds? -----
    sweep_lengthscale_bounds(x, y)


if __name__ == "__main__":
    main()
