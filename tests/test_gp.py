"""Tests for the vanilla GP module."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from efgp_jax.kernels import SE, Matern
from efgp_jax.gp import posterior_covariance, predict, sample_posterior


def _toy_data(n=30, seed=0):
    key = jax.random.PRNGKey(seed)
    x = jnp.linspace(0, 1, n)
    y = jnp.sin(2 * jnp.pi * x) + 0.1 * jax.random.normal(key, (n,))
    return x, y


def test_predict_shape():
    x, y = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_new = jnp.linspace(0, 1, 20)
    mean = predict(x, y, x_new, 0.01, kernel)
    assert mean.shape == (20,)


def test_predict_with_variance():
    x, y = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_new = jnp.linspace(0, 1, 20)
    mean, var = predict(x, y, x_new, 0.01, kernel, compute_var=True)
    assert mean.shape == (20,)
    assert var.shape == (20,)
    assert jnp.all(var >= -1e-8)  # variance should be non-negative (up to numerics)


def test_posterior_covariance_shape():
    x, _ = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_trgs = jnp.linspace(0, 1, 15)
    cov = posterior_covariance(x, x_trgs, 0.01, kernel)
    assert cov.shape == (15, 15)
    # Should be approximately symmetric
    assert jnp.allclose(cov, cov.T, atol=1e-10)


def test_posterior_covariance_positive_diagonal():
    x, _ = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_trgs = jnp.linspace(0, 1, 15)
    cov = posterior_covariance(x, x_trgs, 0.01, kernel)
    assert jnp.all(jnp.diag(cov) > 0)


def test_sample_posterior_shape():
    x, y = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_new = jnp.linspace(0, 1, 10)
    key = jax.random.PRNGKey(42)
    samples = sample_posterior(x, y, x_new, 0.01, kernel, key, n_samples=5)
    assert samples.shape == (10, 5)


def test_predict_interpolates():
    """Posterior mean should pass close to training data when noise is small."""
    x, y = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    mean = predict(x, y, x, 1e-4, kernel)
    # With small noise, predictions at training points should be close to y
    assert jnp.max(jnp.abs(mean - y)) < 0.25


def test_predict_matern():
    x, y = _toy_data()
    kernel = Matern(lengthscale=0.3, variance=1.0, nu=2.5, dim=1)
    x_new = jnp.linspace(0, 1, 20)
    mean = predict(x, y, x_new, 0.01, kernel)
    assert mean.shape == (20,)
    assert jnp.all(jnp.isfinite(mean))


# ---------------------------------------------------------------------------
# JAX traceability tests: the kernel classes are registered pytrees, so
# ``predict`` / ``posterior_covariance`` / ``sample_posterior`` must be usable
# with ``jax.jit`` (no static_argnames for the kernel), ``jax.grad`` through
# the kernel hyperparameters, and ``jax.vmap`` across a batch of kernels.
# ---------------------------------------------------------------------------


def test_predict_jittable():
    """jit(predict) with kernel as a regular pytree arg — no static_argnames."""
    x, y = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_new = jnp.linspace(0, 1, 20)

    jitted = jax.jit(predict)
    m_ref = predict(x, y, x_new, 0.01, kernel)
    m_jit = jitted(x, y, x_new, 0.01, kernel)

    assert m_jit.shape == m_ref.shape
    # jit fusion can produce rounding differences at the last ULP; allclose is
    # the right check rather than bit-equality.
    assert jnp.allclose(m_jit, m_ref, rtol=1e-12, atol=1e-12)

    # Changing hypers should re-trace-free (same compiled function) and give a
    # different answer than the first call.
    kernel2 = SE(lengthscale=0.1, variance=1.0, dim=1)
    m2 = jitted(x, y, x_new, 0.01, kernel2)
    assert not jnp.allclose(m2, m_jit)


def test_predict_jittable_with_variance():
    """jit(predict, compute_var=True) — compute_var is a static Python bool."""
    x, y = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_new = jnp.linspace(0, 1, 20)

    jitted = jax.jit(predict, static_argnames=("compute_var",))
    mean_ref, var_ref = predict(x, y, x_new, 0.01, kernel, compute_var=True)
    mean_jit, var_jit = jitted(x, y, x_new, 0.01, kernel, compute_var=True)

    assert jnp.allclose(mean_jit, mean_ref, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(var_jit, var_ref, rtol=1e-10, atol=1e-12)


def test_posterior_covariance_jittable():
    x, _ = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_trgs = jnp.linspace(0, 1, 15)

    jitted = jax.jit(posterior_covariance)
    c_ref = posterior_covariance(x, x_trgs, 0.01, kernel)
    c_jit = jitted(x, x_trgs, 0.01, kernel)

    assert c_jit.shape == c_ref.shape
    assert jnp.allclose(c_jit, c_ref, rtol=1e-12, atol=1e-12)


def test_sample_posterior_jittable():
    """jit(sample_posterior) — n_samples is static (it sets a shape)."""
    x, y = _toy_data()
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    x_new = jnp.linspace(0, 1, 10)
    key = jax.random.PRNGKey(42)

    jitted = jax.jit(sample_posterior, static_argnames=("n_samples",))
    s_ref = sample_posterior(x, y, x_new, 0.01, kernel, key, n_samples=5)
    s_jit = jitted(x, y, x_new, 0.01, kernel, key, n_samples=5)

    assert s_jit.shape == (10, 5)
    assert jnp.allclose(s_jit, s_ref, rtol=1e-10, atol=1e-10)


def test_predict_grad_hypers():
    """grad of sum(predict(...)) w.r.t. kernel.lengthscale and kernel.variance."""
    x, y = _toy_data()
    x_new = jnp.linspace(0, 1, 20)

    def loss(lengthscale, variance):
        kernel = SE(lengthscale=lengthscale, variance=variance, dim=1)
        return jnp.sum(predict(x, y, x_new, 0.01, kernel))

    grads = jax.grad(loss, argnums=(0, 1))(jnp.asarray(0.3), jnp.asarray(1.0))
    dL_dl, dL_dv = grads
    assert dL_dl.shape == ()
    assert dL_dv.shape == ()
    assert jnp.isfinite(dL_dl)
    assert jnp.isfinite(dL_dv)

    # Sanity: the gradient should be non-trivial for this problem.
    assert jnp.abs(dL_dl) > 1e-6
    assert jnp.abs(dL_dv) > 1e-6


def test_posterior_covariance_grad_hypers():
    x, _ = _toy_data()
    x_trgs = jnp.linspace(0, 1, 15)

    def loss(lengthscale, variance):
        kernel = SE(lengthscale=lengthscale, variance=variance, dim=1)
        return jnp.sum(posterior_covariance(x, x_trgs, 0.01, kernel))

    dL_dl, dL_dv = jax.grad(loss, argnums=(0, 1))(
        jnp.asarray(0.3), jnp.asarray(1.0)
    )
    assert dL_dl.shape == ()
    assert dL_dv.shape == ()
    assert jnp.isfinite(dL_dl)
    assert jnp.isfinite(dL_dv)


def test_sample_posterior_grad_hypers():
    x, y = _toy_data()
    x_new = jnp.linspace(0, 1, 10)
    key = jax.random.PRNGKey(0)

    def loss(lengthscale, variance):
        kernel = SE(lengthscale=lengthscale, variance=variance, dim=1)
        samples = sample_posterior(
            x, y, x_new, 0.01, kernel, key, n_samples=3
        )
        return jnp.sum(samples)

    dL_dl, dL_dv = jax.grad(loss, argnums=(0, 1))(
        jnp.asarray(0.3), jnp.asarray(1.0)
    )
    assert dL_dl.shape == ()
    assert dL_dv.shape == ()
    assert jnp.isfinite(dL_dl)
    assert jnp.isfinite(dL_dv)


def test_predict_grad_through_kernel_pytree():
    """grad can differentiate the kernel object directly (as a pytree)."""
    x, y = _toy_data()
    x_new = jnp.linspace(0, 1, 20)
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)

    def loss(kernel):
        return jnp.sum(predict(x, y, x_new, 0.01, kernel))

    g = jax.grad(loss)(kernel)
    # The gradient w.r.t. a pytree is a pytree of the same structure.
    assert isinstance(g, SE)
    assert g.lengthscale.shape == ()
    assert g.variance.shape == ()
    assert jnp.isfinite(g.lengthscale)
    assert jnp.isfinite(g.variance)


def test_predict_vmap_over_kernels():
    """vmap predict over a batch of SE kernels with different lengthscales."""
    x, y = _toy_data()
    x_new = jnp.linspace(0, 1, 20)
    lengthscales = jnp.array([0.1, 0.2, 0.3, 0.5])

    def make_kernel(l):
        return SE(lengthscale=l, variance=jnp.asarray(1.0), dim=1)

    kernels = jax.vmap(make_kernel)(lengthscales)

    def f(kernel):
        return predict(x, y, x_new, 0.01, kernel)

    batched = jax.vmap(f)(kernels)
    assert batched.shape == (lengthscales.size, x_new.size)

    # Compare against a plain Python loop.
    stacked = jnp.stack(
        [predict(x, y, x_new, 0.01, SE(lengthscale=l, variance=1.0, dim=1))
         for l in lengthscales]
    )
    assert jnp.allclose(batched, stacked, rtol=1e-10, atol=1e-10)


def test_predict_vmap_over_both_hypers():
    """vmap over lengthscale and variance simultaneously."""
    x, y = _toy_data()
    x_new = jnp.linspace(0, 1, 20)
    lengthscales = jnp.array([0.1, 0.2, 0.3])
    variances = jnp.array([0.5, 1.0, 2.0])

    def make_kernel(l, v):
        return SE(lengthscale=l, variance=v, dim=1)

    kernels = jax.vmap(make_kernel)(lengthscales, variances)

    def f(kernel):
        return predict(x, y, x_new, 0.01, kernel)

    batched = jax.vmap(f)(kernels)
    assert batched.shape == (3, x_new.size)
    assert jnp.all(jnp.isfinite(batched))


def test_posterior_covariance_vmap_over_kernels():
    x, _ = _toy_data()
    x_trgs = jnp.linspace(0, 1, 15)
    lengthscales = jnp.array([0.2, 0.3, 0.4])

    def make_kernel(l):
        return SE(lengthscale=l, variance=jnp.asarray(1.0), dim=1)

    kernels = jax.vmap(make_kernel)(lengthscales)

    def f(kernel):
        return posterior_covariance(x, x_trgs, 0.01, kernel)

    batched = jax.vmap(f)(kernels)
    assert batched.shape == (3, 15, 15)

    stacked = jnp.stack(
        [posterior_covariance(x, x_trgs, 0.01,
                              SE(lengthscale=l, variance=1.0, dim=1))
         for l in lengthscales]
    )
    assert jnp.allclose(batched, stacked, rtol=1e-10, atol=1e-10)


def test_sample_posterior_vmap_over_kernels():
    x, y = _toy_data()
    x_new = jnp.linspace(0, 1, 10)
    key = jax.random.PRNGKey(7)
    lengthscales = jnp.array([0.2, 0.3, 0.4])

    def make_kernel(l):
        return SE(lengthscale=l, variance=jnp.asarray(1.0), dim=1)

    kernels = jax.vmap(make_kernel)(lengthscales)

    def f(kernel):
        return sample_posterior(x, y, x_new, 0.01, kernel, key, n_samples=4)

    batched = jax.vmap(f)(kernels)
    assert batched.shape == (3, 10, 4)
    assert jnp.all(jnp.isfinite(batched))


def test_predict_jit_matches_nonjit_numerically():
    """Bit-level check that jit doesn't drift outside floating-point noise."""
    x, y = _toy_data(n=50, seed=1)
    kernel = SE(lengthscale=0.25, variance=0.8, dim=1)
    x_new = jnp.linspace(0, 1, 30)

    m_ref = predict(x, y, x_new, 0.02, kernel)
    m_jit = jax.jit(predict)(x, y, x_new, 0.02, kernel)

    # Within a few ULPs at float64.
    assert jnp.max(jnp.abs(m_ref - m_jit)) < 1e-12
