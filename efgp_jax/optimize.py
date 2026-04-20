"""Hyperparameter optimization for EFGP via L-BFGS."""

from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from scipy.optimize import minimize as sp_minimize

from .kernels import Kernel, SE, Matern, log_marginal
from .efgp import EFGP


def optimize_hyperparameters(
    x: Array,
    y: Array,
    kernel0: Kernel,
    sigmasq0: float,
    eps: float,
    *,
    domain=None,
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

    Uses L-BFGS-B (via scipy) with gradients from EFGPPosterior.gradient.
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
    domain : tuple or None
        Domain specification, e.g. (0, 1) for 1D or ((0, 1), (0, 1)) for 2D.
        If None, inferred from data bounds.
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

    # Infer domain from data if not provided
    if domain is None:
        if x.ndim == 1:
            domain = (float(jnp.min(x)), float(jnp.max(x)))
        else:
            domain = tuple(
                (float(jnp.min(x[:, i])), float(jnp.max(x[:, i])))
                for i in range(x.shape[1])
            )

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

        gp = EFGP(kernel, domain, eps,
                   nufft_eps=nufft_eps, cg_tol=cg_tol,
                   use_integral=use_integral, use_precond=use_precond)
        posterior = gp.condition(x, y, sig2_val)
        grad, lml = posterior.gradient(
            subkey,
            trace_samples=trace_samples,
            compute_log_marginal=True,
            log_marginal_probes=log_marginal_probes,
            log_marginal_steps=log_marginal_steps,
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


# ---------------------------------------------------------------------------
# Alternative: exact MLL via jaxopt (pytree-native, pure JAX)
# ---------------------------------------------------------------------------

def optimize_hyperparameters_exact(
    x: Array,
    y: Array,
    kernel0: Kernel,
    sigmasq0: float,
    *,
    maxiter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[Kernel, float, dict]:
    """Exact MLL optimization via jaxopt L-BFGS (Cholesky-based, O(N^3)).

    Alternative to :func:`optimize_hyperparameters`.  Leverages the
    pytree-registered kernel: parameters are ``(log_kernel, log_sigmasq)``
    where ``log_kernel`` is a ``SE``/``Matern`` whose leaves are log-hypers.
    ``jaxopt.LBFGS`` treats this as a standard pytree and handles the
    flatten/unflatten internally.  Gradients come from ``jax.grad`` through
    the Cholesky in :func:`log_marginal`, so they are exact (no Hutchinson /
    SLQ noise).

    Suitable for small-to-moderate ``N`` (typically up to a few thousand).
    For large ``N`` use :func:`optimize_hyperparameters`, which uses EFGP +
    stochastic gradient estimators.

    Parameters
    ----------
    x, y : Array
        Training data.
    kernel0 : Kernel
        Initial kernel (hyperparameters in natural space).
    sigmasq0 : float
        Initial noise variance.
    maxiter : int
        Max L-BFGS iterations.
    tol : float
        Gradient-norm tolerance for convergence.
    verbose : bool
        If True, jaxopt prints progress.

    Returns
    -------
    kernel : Kernel
        Optimized kernel (same subclass as ``kernel0``).
    sigmasq : float
        Optimized noise variance.
    info : dict
        Keys: ``nll``, ``n_iter``, ``grad_norm``.
    """
    try:
        import jaxopt
    except ImportError as e:
        raise ImportError(
            "optimize_hyperparameters_exact requires jaxopt "
            "(`pip install jaxopt`)."
        ) from e

    # Parameters: log-space for positivity. Kernel is a pytree, so tree_map
    # applies jnp.log to lengthscale and variance leaves, leaving aux intact.
    log_kernel0 = jax.tree_util.tree_map(jnp.log, kernel0)
    log_sig0 = jnp.log(jnp.asarray(sigmasq0))

    def nll(params):
        log_kernel, log_sig = params
        kernel = jax.tree_util.tree_map(jnp.exp, log_kernel)
        sigmasq = jnp.exp(log_sig)
        return -log_marginal(x, y, sigmasq, kernel)

    solver = jaxopt.LBFGS(fun=nll, maxiter=maxiter, tol=tol, verbose=verbose)
    params_opt, state = solver.run((log_kernel0, log_sig0))

    log_kernel_fin, log_sig_fin = params_opt
    kernel_fin = jax.tree_util.tree_map(jnp.exp, log_kernel_fin)
    sigmasq_fin = float(jnp.exp(log_sig_fin))

    info = {
        'nll': float(state.value),
        'n_iter': int(state.iter_num),
        'grad_norm': float(state.error),
    }
    return kernel_fin, sigmasq_fin, info
