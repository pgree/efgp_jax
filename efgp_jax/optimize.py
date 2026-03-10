"""Hyperparameter optimization for EFGP via L-BFGS."""

from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from scipy.optimize import minimize as sp_minimize

from .kernels import Kernel, SE, Matern
from .efgp import efgp_gradient


def optimize_hyperparameters(
    x: Array,
    y: Array,
    kernel0: Kernel,
    sigmasq0: float,
    eps: float,
    *,
    key: Array,
    maxiter: int = 50,
    trace_samples: int = 30,
    cg_tol: Optional[float] = None,
    nufft_eps: float = 6e-8,
    use_integral: bool = True,
    log_marginal_probes: int = 50,
    log_marginal_steps: int = 30,
    verbose: bool = True,
    use_precond: bool = False,
) -> Tuple[Kernel, float, dict]:
    """Optimize GP hyperparameters by minimizing negative log marginal likelihood.

    Uses L-BFGS-B (via scipy) with gradients from efgp_gradient.
    Optimization is performed in log-space to ensure positivity.

    Parameters
    ----------
    x : Array, shape (n,) or (n, d)
    y : Array, shape (n,)
    kernel0 : Kernel
        Initial kernel (with initial hyperparameters).
    sigmasq0 : float
        Initial noise variance.
    eps : float
        EFGP spectral truncation tolerance.
    key : Array
        JAX PRNG key.
    maxiter : int
        Maximum number of L-BFGS iterations.
    trace_samples : int
        Number of Hutchinson trace samples for gradient estimation.
    cg_tol : float or None
        CG solver tolerance (defaults to eps/100).
    nufft_eps : float
    use_integral : bool
    log_marginal_probes : int
        Number of SLQ probes for log-determinant estimation.
    log_marginal_steps : int
        Number of Lanczos steps for log-determinant estimation.
    verbose : bool
        Print progress at each function evaluation.

    Returns
    -------
    kernel : Kernel
        Optimized kernel.
    sigmasq : float
        Optimized noise variance.
    info : dict
        Optimization info with keys 'nll', 'nfev', 'success'.
    """
    if cg_tol is None:
        cg_tol = eps / 100

    def _make_kernel(l_val, var_val):
        if isinstance(kernel0, Matern):
            return Matern(lengthscale=l_val, variance=var_val,
                         dim=kernel0.dim, nu=kernel0.nu)
        else:
            return SE(lengthscale=l_val, variance=var_val, dim=kernel0.dim)

    # Mutable key state for the objective function
    state = {'key': key}

    theta0 = np.array([
        np.log(kernel0.lengthscale),
        np.log(kernel0.variance),
        np.log(sigmasq0),
    ])

    def objective(log_theta):
        state['key'], subkey = jax.random.split(state['key'])
        l_val = float(np.exp(log_theta[0]))
        var_val = float(np.exp(log_theta[1]))
        sig2_val = float(np.exp(log_theta[2]))
        kernel = _make_kernel(l_val, var_val)

        grad, lml = efgp_gradient(
            x, y, sigmasq=sig2_val,
            kernel=kernel,
            eps=eps, trace_samples=trace_samples,
            key=subkey,
            cg_tol=cg_tol,
            nufft_eps=nufft_eps,
            use_integral=use_integral,
            compute_log_marginal=True,
            log_marginal_probes=log_marginal_probes,
            log_marginal_steps=log_marginal_steps,
            use_precond=use_precond,
        )

        # Chain rule: d/d(log_theta) = d/d(theta) * theta
        grad_log = np.array(grad) * np.array([l_val, var_val, sig2_val])
        nll = float(-lml)

        if verbose:
            print(f"  l={l_val:.4f}  var={var_val:.4f}  "
                  f"noise={sig2_val:.4f}  NLL={nll:.2f}")

        return nll, grad_log.astype(np.float64)

    res = sp_minimize(objective, theta0, method='L-BFGS-B', jac=True,
                      options={'maxiter': maxiter})

    kernel_final = _make_kernel(
        float(np.exp(res.x[0])),
        float(np.exp(res.x[1])),
    )
    sigmasq_final = float(np.exp(res.x[2]))

    info = {
        'nll': float(res.fun),
        'nfev': res.nfev,
        'success': res.success,
    }

    return kernel_final, sigmasq_final, info
