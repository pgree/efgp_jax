"""Main EFGP algorithm: gradient, predict, fit."""

import math
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class

import numpy as np

from .kernels import Kernel, SE
from .quadrature import get_xis
from .cg import cg_solve, cg_solve_batched
from .toeplitz import ToeplitzND, make_toeplitz, toeplitz_apply
from .nufft import _make_phi, nufft_type1, nufft_type2, _cmplx


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_domain(domain, d):
    """Parse domain specification into L (extent) and xcen (center).

    Parameters
    ----------
    domain : tuple
        1D: (lo, hi) — a single interval.
        nD: ((lo1, hi1), (lo2, hi2), ...) — one interval per dimension.

    Returns
    -------
    L : float
        Max extent across dimensions.
    xcen : array, shape (d,)
        Midpoint of the domain.
    """
    # 1D shorthand: (lo, hi)
    if d == 1 and not isinstance(domain[0], (tuple, list)):
        lo, hi = float(domain[0]), float(domain[1])
        return jnp.asarray(hi - lo), jnp.array([(lo + hi) / 2.0])

    # nD: tuple of intervals
    assert len(domain) == d, f"Expected {d} intervals, got {len(domain)}"
    los = [float(interval[0]) for interval in domain]
    his = [float(interval[1]) for interval in domain]
    extents = [hi - lo for lo, hi in zip(los, his)]
    centers = [(lo + hi) / 2.0 for lo, hi in zip(los, his)]
    L = max(extents)
    return jnp.asarray(L), jnp.array(centers)


def _compute_convolution_vector(m, x, h, xcen, nufft_eps=6e-8):
    """Multi-D type-1 NUFFT convolution vector: v[k] = sum_n exp(2 pi i <k, x_n>).

    Parameters
    ----------
    m : int or tuple of int
        Half-width per dimension. If int, same for all dimensions.
    h : float or Array
        Frequency spacing, scalar or per-dimension array.
    xcen : Array
        Center for NUFFT phase computation.
    """
    if x.ndim == 1:
        x = x[:, None]
    N, d = x.shape
    cdtype = _cmplx(x.dtype)
    c = jnp.ones(N, dtype=cdtype)
    if isinstance(m, (tuple, list)):
        OUT = tuple(4 * mi + 1 for mi in m)
    else:
        OUT = tuple([4 * m + 1] * d)
    phi = _make_phi(x, xcen, h)
    return nufft_type1(phi, c, OUT, eps=nufft_eps)


def _is_anisotropic(kernel):
    """Check if kernel has per-dimension lengthscales."""
    return getattr(kernel, 'is_anisotropic', False)


def _create_Gv(ws, toeplitz_op, cdtype):
    """Gv(v) = D T D v  where D = diag(ws), T = Toeplitz."""
    ns = toeplitz_op.ns

    def Gv(v):
        v = v.astype(cdtype)
        if v.ndim <= 1:
            return ws * toeplitz_apply(toeplitz_op, ws * v)
        else:
            # batched
            shape_in = (v.shape[0], *ns)
            ws_block = ws.reshape(1, *ns)
            v_block = v.reshape(shape_in)
            Tv = toeplitz_apply(toeplitz_op, ws_block * v_block)
            result = ws_block * Tv
            return result.reshape(v.shape)

    return Gv


def _create_A_mean(ws, toeplitz_op, sigmasq, cdtype):
    Gv = _create_Gv(ws, toeplitz_op, cdtype)

    def A_mean(beta):
        return Gv(beta) + sigmasq * beta

    return A_mean


def _create_A_var(ws, toeplitz_op, sigmasq, cdtype):
    Gv = _create_Gv(ws, toeplitz_op, cdtype)

    def A_var(gamma):
        return Gv(gamma) / sigmasq + gamma

    return A_var


def _make_diag_precond(ws, N, sigmasq, mode="mean"):
    """Diagonal preconditioner for EFGP CG solves.

    For A_mean = G + σ²I:  diag = |ws|² · N + σ²
    For A_var  = G/σ² + I: diag = |ws|² · N / σ² + 1
    """
    ws_sq = jnp.real(ws * jnp.conj(ws))
    if mode == "mean":
        diag = ws_sq * N + sigmasq
    else:
        diag = ws_sq * N / sigmasq + 1.0
    return lambda v: v / diag


# ---------------------------------------------------------------------------
# Log-determinant via Stochastic Lanczos Quadrature
# ---------------------------------------------------------------------------

def logdet_slq(
    ws: Array,
    sigma2: float,
    toeplitz_op: ToeplitzND,
    *,
    probes: int = 100,
    steps: int = 25,
    key: Array,
    n: int,
) -> Array:
    """Estimate log det(I + sigma^{-2} D T D) via Hutchinson + Lanczos.

    Parameters
    ----------
    ws : Array, shape (m,)
    sigma2 : float
    toeplitz_op : ToeplitzND
    probes, steps : int
    key : jax PRNG key
    n : int, number of training points (for the n*log(sigma2) offset)

    Returns
    -------
    Array
    """
    ws_real = jnp.real(ws)
    m = ws_real.size
    cdtype = ws.dtype if jnp.iscomplexobj(ws) else _cmplx(ws.dtype)
    rdtype = jnp.float64 if cdtype == jnp.complex128 else jnp.float32

    Gv = _create_Gv(ws_real, toeplitz_op, cdtype)

    def Av(v):
        return v + (1.0 / sigma2) * Gv(v)

    # Generate all probe keys at once
    keys = jax.random.split(key, probes)

    def _lanczos_one(subkey):
        z = 2.0 * jax.random.bernoulli(subkey, shape=(m,)).astype(rdtype) - 1.0
        q = z / jnp.linalg.norm(z)

        def scan_fn(carry, _):
            q_cur, q_prev, beta_prev = carry
            v = jnp.real(Av(q_cur)) - beta_prev * q_prev
            alpha = jnp.dot(q_cur, v)
            v = v - alpha * q_cur
            beta = jnp.linalg.norm(v)
            # Always update (even if beta is tiny — tridiagonal entry is just 0)
            q_next = jnp.where(beta > 1e-12, v / beta, jnp.zeros_like(v))
            beta_out = jnp.where(beta > 1e-12, beta, 0.0)
            return (q_next, q_cur, beta_out), (alpha, beta_out)

        init_carry = (q, jnp.zeros_like(q), jnp.array(0.0, dtype=rdtype))
        _, (alphas, betas) = jax.lax.scan(scan_fn, init_carry, None, length=steps)

        # Build tridiagonal matrix T
        T_mat = jnp.diag(alphas) + jnp.diag(betas[:-1], 1) + jnp.diag(betas[:-1], -1)
        evals, evecs = jnp.linalg.eigh(T_mat)
        evals = jnp.clip(evals, 1e-18)
        w1 = evecs[0]
        quad = jnp.sum(w1 ** 2 * jnp.log(evals)) * jnp.dot(z, z)
        return quad

    quads = jax.vmap(_lanczos_one)(keys)
    logdet = jnp.mean(quads) + n * jnp.log(sigma2)
    return logdet


# ---------------------------------------------------------------------------
# EFGP class
# ---------------------------------------------------------------------------

@register_pytree_node_class
class EFGP:
    """Spectral approximation of a GP prior.

    Precomputes the frequency grid and spectral weights from the kernel,
    domain, and truncation tolerance ``eps``.  These do *not* depend on
    training data.

    Use :meth:`condition` to bind training locations, observations, and a
    noise model, producing an :class:`EFGPPosterior`.

    Parameters
    ----------
    kernel : Kernel
        Kernel object (e.g. ``SE(...)`` or ``Matern(...)``).
    domain : tuple
        1D: ``(lo, hi)`` — the interval on which the GP lives.
        nD: ``((lo1, hi1), (lo2, hi2), ...)`` — one interval per dimension.
    eps : float
        Spectral truncation tolerance (smaller = more frequencies).
    nufft_eps : float
    cg_tol : float
    use_integral : bool
    use_precond : bool
    """

    def __init__(self, kernel, domain, eps, *,
                 nufft_eps=6e-8, cg_tol=1e-6, use_integral=True,
                 use_precond=False):
        self.kernel = kernel
        self.domain = domain
        self.eps = eps
        self.nufft_eps = nufft_eps
        self.cg_tol = cg_tol
        self.use_integral = use_integral
        self.use_precond = use_precond

        d = kernel.dim
        self.d = d
        cdtype = jnp.complex128  # default to float64

        L, xcen = _parse_domain(domain, d)
        self.L = L
        self.xcen = xcen

        if _is_anisotropic(kernel):
            l_arr = np.asarray(kernel.lengthscale)
            grids_1d = []
            for i in range(d):
                k1d = SE(lengthscale=float(l_arr[i]), variance=kernel.variance, dim=1)
                xis_1d_i, h_i, mtot_i = get_xis(k1d, eps, self.L,
                                                  use_integral=use_integral)
                grids_1d.append((xis_1d_i, h_i, mtot_i))
            h = jnp.array([g[1] for g in grids_1d])
            mtot = tuple(g[2] for g in grids_1d)
            OUT = mtot
            grids = jnp.meshgrid(*[g[0] for g in grids_1d], indexing="ij")
            xis = jnp.stack([g.ravel() for g in grids], axis=-1)
            ws = jnp.sqrt(kernel.spectral_density(xis).astype(cdtype) * jnp.prod(h))
        else:
            xis_1d, h, mtot = get_xis(kernel, eps, self.L,
                                       use_integral=use_integral)
            grids = jnp.meshgrid(*[xis_1d for _ in range(d)], indexing="ij")
            xis = jnp.stack([g.ravel() for g in grids], axis=-1)
            ws = jnp.sqrt(kernel.spectral_density(xis).astype(cdtype) * h ** d)
            OUT = (mtot,) * d

        self.xis = xis
        self.h = jnp.asarray(h)
        self.mtot = mtot
        self.ws = ws
        self.OUT = OUT
        self.cdtype = cdtype
        self.M = ws.shape[0]

    # --- pytree protocol ---------------------------------------------------

    def tree_flatten(self):
        children = (self.kernel, self.xis, self.h, self.ws, self.xcen, self.L)
        aux = (
            self.domain,
            self.eps,
            self.nufft_eps,
            self.cg_tol,
            self.use_integral,
            self.use_precond,
            self.d,
            self.mtot,
            self.OUT,
            self.cdtype,
            self.M,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            domain,
            eps,
            nufft_eps,
            cg_tol,
            use_integral,
            use_precond,
            d,
            mtot,
            OUT,
            cdtype,
            M,
        ) = aux
        kernel, xis, h, ws, xcen, L = children
        obj = cls.__new__(cls)
        obj.kernel = kernel
        obj.domain = domain
        obj.eps = eps
        obj.nufft_eps = nufft_eps
        obj.cg_tol = cg_tol
        obj.use_integral = use_integral
        obj.use_precond = use_precond
        obj.d = d
        obj.L = L
        obj.xcen = xcen
        obj.xis = xis
        obj.h = h
        obj.mtot = mtot
        obj.ws = ws
        obj.OUT = OUT
        obj.cdtype = cdtype
        obj.M = M
        return obj

    def sample(self, x, key, n_samples=1, *, nufft_eps=1e-12):
        """Draw samples from the GP prior at locations x.

        Parameters
        ----------
        x : Array, shape (N,) or (N, d)
        key : JAX PRNG key
        n_samples : int
        nufft_eps : float

        Returns
        -------
        Array, shape (n_samples, N) or (N,) if n_samples == 1.
        """
        if x.ndim == 1:
            x = x[:, None]

        phi = _make_phi(x, self.xcen, self.h)

        # Complex normal: z = a + ib, a,b ~ N(0,1)
        key1, key2 = jax.random.split(key)
        z_real = jax.random.normal(key1, shape=(n_samples, self.M))
        z_imag = jax.random.normal(key2, shape=(n_samples, self.M))
        z = (z_real + 1j * z_imag).astype(self.cdtype)

        # f(x) = sum_j z_j * w_j * exp(2 pi i xi_j . x)
        wc = self.ws[None, :] * z  # (n_samples, M)
        samples = jnp.real(nufft_type2(phi, wc.reshape(n_samples, *self.OUT),
                                        eps=nufft_eps))

        if n_samples == 1:
            return samples[0]
        return samples

    def eval_basis(self, x, indices=None):
        """Evaluate weighted basis functions at given points.

        Each basis function is phi_j(x) = w_j * exp(2 pi i xi_j . x).
        Returns the real and imaginary (cos and sin) parts separately.

        Parameters
        ----------
        x : Array, shape (N,) or (N, d)
            Evaluation points.
        indices : array-like of int, optional
            Which basis function indices to evaluate.
            Defaults to all M basis functions.

        Returns
        -------
        cos_part : Array, shape (N, len(indices))
            w_j * cos(2 pi xi_j . x) for each selected j.
        sin_part : Array, shape (N, len(indices))
            w_j * sin(2 pi xi_j . x) for each selected j.
        """
        if x.ndim == 1:
            x = x[:, None]

        xis = self.xis  # (M, d)
        ws = jnp.real(self.ws)

        if indices is not None:
            indices = jnp.asarray(indices)
            xis = xis[indices]
            ws = ws[indices]

        # phases: (N, K) where K = number of selected basis functions
        phases = 2 * math.pi * ((x - self.xcen) @ xis.T)  # (N, K)
        cos_part = ws[None, :] * jnp.cos(phases)
        sin_part = ws[None, :] * jnp.sin(phases)
        return cos_part, sin_part

    def condition(self, x, y, sigmasq):
        """Condition on observations, returning a posterior object.

        Parameters
        ----------
        x : Array, shape (n,) or (n, d)
            Training locations.
        y : Array, shape (n,)
            Observations.
        sigmasq : float
            Observation noise variance.

        Returns
        -------
        EFGPPosterior
        """
        return EFGPPosterior(self, x, y, sigmasq)


@register_pytree_node_class
class EFGPPosterior:
    """Posterior GP conditioned on observations.

    Created by :meth:`EFGP.condition`.  Caches the CG solution so that
    :meth:`predict` (with or without variance) reuses the same solve.

    Parameters
    ----------
    prior : EFGP
    x : Array, shape (n,) or (n, d)
    y : Array, shape (n,)
    sigmasq : float
    """

    def __init__(self, prior, x, y, sigmasq):
        self.prior = prior
        if x.ndim == 1:
            x = x[:, None]
        self.x = x
        self.N = x.shape[0]
        self.y = y
        self.sigmasq = jnp.asarray(sigmasq)
        self._beta = None  # lazily computed CG solution

        # Compute NUFFT phases and Toeplitz operator for training locations
        p = prior
        self.phi = _make_phi(x, p.xcen, p.h)
        if isinstance(p.mtot, tuple):
            m_conv = tuple((m - 1) // 2 for m in p.mtot)
        else:
            m_conv = (p.mtot - 1) // 2
        v_kernel = _compute_convolution_vector(
            m_conv, x, p.h, p.xcen, p.nufft_eps
        ).astype(p.cdtype)
        self.toeplitz_op = make_toeplitz(v_kernel, force_pow2=True)

    # --- pytree protocol ---------------------------------------------------

    def tree_flatten(self):
        children = (
            self.prior,
            self.x,
            self.y,
            self.sigmasq,
            self.phi,
            self.toeplitz_op,
            self._beta,
        )
        aux = (self.N,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (N,) = aux
        prior, x, y, sigmasq, phi, toeplitz_op, _beta = children
        obj = cls.__new__(cls)
        obj.prior = prior
        obj.x = x
        obj.y = y
        obj.sigmasq = sigmasq
        obj.phi = phi
        obj.toeplitz_op = toeplitz_op
        obj._beta = _beta
        obj.N = N
        return obj

    def _solve(self):
        """Solve for the mean coefficients (cached)."""
        if self._beta is not None:
            return self._beta
        p = self.prior
        fadj = lambda v: nufft_type1(
            self.phi, v.astype(p.cdtype), p.OUT, eps=p.nufft_eps
        ).reshape(-1)
        A_mean = _create_A_mean(p.ws, self.toeplitz_op, self.sigmasq, p.cdtype)
        M_inv = _make_diag_precond(p.ws, self.N, self.sigmasq, "mean") if p.use_precond else None
        Fy = fadj(self.y.astype(p.cdtype))
        self._beta = cg_solve(A_mean, p.ws * Fy, tol=p.cg_tol, M_inv_apply=M_inv)
        return self._beta

    def predict(self, x_new, *, return_var=False, max_cg_iter=1000):
        """Posterior mean (and optionally variance) at new locations.

        Parameters
        ----------
        x_new : Array, shape (n_new,) or (n_new, d)
        return_var : bool
            If True, return ``(mean, var)`` instead of just ``mean``.
        max_cg_iter : int
            Max CG iterations for the variance solves.

        Returns
        -------
        mean : Array, shape (n_new,)
        var : Array, shape (n_new,)   (only if ``return_var=True``)
        """
        p = self.prior
        if x_new.ndim == 1:
            x_new = x_new[:, None]

        beta = self._solve()
        phi_new = _make_phi(x_new, p.xcen, p.h)
        yhat = jnp.real(nufft_type2(
            phi_new, (p.ws * beta).reshape(p.OUT), eps=p.nufft_eps
        ))

        if not return_var:
            return yhat

        TWO_PI = 2 * math.pi
        A_var = _create_A_var(p.ws, self.toeplitz_op, self.sigmasq, p.cdtype)
        M_inv_var = _make_diag_precond(p.ws, self.N, self.sigmasq, "var") if p.use_precond else None
        xis_flat = p.xis.reshape(-1, p.d)
        fx = jnp.exp(TWO_PI * 1j * ((x_new - p.xcen) @ xis_flat.T)).astype(p.cdtype)
        rhs_var = p.ws[None, :] * jnp.conj(fx)
        gamma = cg_solve_batched(A_var, rhs_var, tol=p.cg_tol, max_iter=max_cg_iter,
                                 M_inv_apply=M_inv_var)
        var = jnp.real(jnp.sum(fx * (p.ws[None, :] * gamma), axis=-1))
        var = jnp.clip(var, 0.0)

        return yhat, var

    def sample(self, x_new, key, n_samples=1):
        """Draw posterior samples at x_new via the Matheron rule.

        Parameters
        ----------
        x_new : Array, shape (n_new,) or (n_new, d)
        key : JAX PRNG key
        n_samples : int

        Returns
        -------
        samples : Array, shape (n_new,) if n_samples==1, else (n_samples, n_new)
        """
        p = self.prior
        if x_new.ndim == 1:
            x_new = x_new[:, None]

        key1, key2, key3 = jax.random.split(key, 3)
        z_real = jax.random.normal(key1, shape=(n_samples, p.M))
        z_imag = jax.random.normal(key2, shape=(n_samples, p.M))
        z = (z_real + 1j * z_imag).astype(p.cdtype)

        wz = (p.ws[None, :] * z).reshape(n_samples, *p.OUT)
        f_prior_x = jnp.real(nufft_type2(self.phi, wz, eps=p.nufft_eps))

        phi_new = _make_phi(x_new, p.xcen, p.h)
        f_prior_xnew = jnp.real(nufft_type2(phi_new, wz, eps=p.nufft_eps))

        noise = jax.random.normal(key3, shape=(n_samples, self.N))
        r = self.y[None, :] - f_prior_x - jnp.sqrt(self.sigmasq) * noise

        Fr = nufft_type1(self.phi, r.astype(p.cdtype), p.OUT, eps=p.nufft_eps)
        Fr = Fr.reshape(n_samples, -1)

        A_mean = _create_A_mean(p.ws, self.toeplitz_op, self.sigmasq, p.cdtype)
        M_inv = _make_diag_precond(p.ws, self.N, self.sigmasq, "mean") if p.use_precond else None
        beta = cg_solve_batched(A_mean, p.ws[None, :] * Fr, tol=p.cg_tol,
                                M_inv_apply=M_inv)

        correction_coeffs = (p.ws[None, :] * beta).reshape(n_samples, *p.OUT)
        correction = jnp.real(nufft_type2(phi_new, correction_coeffs, eps=p.nufft_eps))

        samples = f_prior_xnew + correction
        if n_samples == 1:
            return samples[0]
        return samples

    def gradient(self, key, *, trace_samples=10, compute_log_marginal=False,
                 log_marginal_probes=100, log_marginal_steps=25):
        """Gradient of the negative log marginal likelihood w.r.t. hyperparameters.

        Returns grad of shape (num_hypers,) = [d/dl, d/dvar, d/dsigmasq].
        If ``compute_log_marginal`` is True, returns ``(grad, log_marginal)``
        where ``log_marginal`` is a JAX scalar array (not a Python float).
        """
        p = self.prior
        cg_tol = p.cg_tol
        nufft_eps = p.nufft_eps
        N, d = self.N, p.d
        cdtype = p.cdtype
        ws = p.ws
        toeplitz_op = self.toeplitz_op

        # Spectral gradients
        dl, dvar = p.kernel.spectral_grad(p.xis)
        h_vol = jnp.prod(jnp.asarray(p.h)) if _is_anisotropic(p.kernel) else p.h ** d
        if dl.ndim == 2:
            # anisotropic: dl is (M, d), dvar is (M,)
            Dprime = jnp.concatenate([dl, dvar[:, None]], axis=-1).astype(cdtype) * h_vol
        else:
            Dprime = jnp.stack([dl, dvar], axis=-1).astype(cdtype) * h_vol  # (M, 2)

        fadj = lambda v: nufft_type1(self.phi, v.astype(cdtype), p.OUT, eps=nufft_eps).reshape(-1)
        fwd = lambda fk: nufft_type2(self.phi, fk.reshape(p.OUT) if fk.ndim == 1 else fk, eps=nufft_eps)

        A_apply = _create_A_mean(ws, toeplitz_op, self.sigmasq, cdtype)
        M_inv = _make_diag_precond(ws, N, self.sigmasq, "mean") if p.use_precond else None

        # Solve A beta = ws * F* y
        Fy = fadj(self.y.astype(cdtype))
        rhs = ws * Fy
        beta = cg_solve(A_apply, rhs, tol=cg_tol, M_inv_apply=M_inv)
        beta = beta * ws
        z_pred = fwd(beta)
        alpha = (self.y.astype(cdtype) - z_pred) / self.sigmasq

        # Term 2: (alpha' D' alpha, alpha' alpha)
        fadj_alpha = (Fy - toeplitz_apply(toeplitz_op, beta)) / self.sigmasq
        Hk = Dprime.shape[-1]  # number of kernel hyperparameters
        term2_parts = [jnp.vdot(fadj_alpha, Dprime[:, i] * fadj_alpha) for i in range(Hk)]
        term2_parts.append(jnp.vdot(alpha, alpha))
        term2 = jnp.stack(term2_parts)

        # Monte-Carlo trace (term 1)
        T = trace_samples
        num_hypers = Hk + 1  # kernel grads + noise

        key, subkey = jax.random.split(key)
        rdtype = jnp.float64 if cdtype == jnp.complex128 else jnp.float32
        Z = 2.0 * jax.random.bernoulli(subkey, shape=(T, N)).astype(rdtype) - 1.0
        Z_c = Z.astype(cdtype)

        # F* Z
        fadjZ_list = []
        for i in range(T):
            fadjZ_list.append(fadj(Z_c[i]))
        fadjZ_flat = jnp.stack(fadjZ_list)  # (T, M)

        # Build all RHS for batched CG
        Di_FZ_all = jnp.concatenate(
            [Dprime[:, i][None, :] * fadjZ_flat for i in range(Hk)], axis=0
        )  # (Hk*T, M)

        # NUFFT type-2 for kernel terms
        rhs_all_kernel = []
        for i in range(Hk * T):
            rhs_all_kernel.append(fwd(Di_FZ_all[i]))
        rhs_all_kernel = jnp.stack(rhs_all_kernel).reshape(Hk, T, -1)  # (Hk, T, N)

        # Toeplitz for B
        B_all_kernel = []
        for i in range(Hk * T):
            B_all_kernel.append(ws * toeplitz_apply(toeplitz_op, Di_FZ_all[i]))
        B_all_kernel = jnp.stack(B_all_kernel).reshape(Hk, T, -1)  # (Hk, T, M)

        # Noise term
        rhs_noise = Z_c  # (T, N)
        B_noise = ws[None, :] * fadjZ_flat  # (T, M)

        R_all = jnp.concatenate(
            [rhs_all_kernel.reshape(Hk * T, -1), rhs_noise], axis=0
        )  # (num_hypers*T, N)
        B_all = jnp.concatenate(
            [B_all_kernel.reshape(Hk * T, -1), B_noise], axis=0
        )  # (num_hypers*T, M)

        # Batched CG
        Beta_all = cg_solve_batched(A_apply, B_all, tol=cg_tol, M_inv_apply=M_inv)
        Beta_all = Beta_all * ws[None, :]

        # Compute alpha for each probe
        fwdBeta_list = []
        for i in range(num_hypers * T):
            fwdBeta_list.append(fwd(Beta_all[i]))
        fwdBeta = jnp.stack(fwdBeta_list)

        Alpha_batch = (R_all - fwdBeta) / self.sigmasq
        Alpha_batch = Alpha_batch.reshape(num_hypers, T, -1)
        term1 = jnp.mean(jnp.sum(Z_c[None, :, :] * Alpha_batch, axis=2), axis=1)

        # Gradient
        grad = 0.5 * (term1 - term2)

        if compute_log_marginal:
            key, subkey = jax.random.split(key)
            det_term = logdet_slq(
                ws, self.sigmasq, toeplitz_op,
                probes=log_marginal_probes,
                steps=log_marginal_steps,
                key=subkey, n=N,
            )
            vdot_term = jnp.vdot(self.y.astype(cdtype), alpha).real
            log_marg = -0.5 * vdot_term - 0.5 * det_term - 0.5 * N * math.log(2 * math.pi)
            return jnp.real(grad), log_marg

        return jnp.real(grad)
