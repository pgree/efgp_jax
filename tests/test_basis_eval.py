"""Tests for basis_eval: NUFFT-based Fourier basis evaluation."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from efgp_jax.basis_eval import (
    make_basis,
    evaluate_basis_nufft,
    evaluate_basis_dense,
    apply_basis_nufft,
    sample_gp_nufft,
)
from efgp_jax.kernels import SE, Matern, pairwise_distances


def test_make_basis_se():
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    basis = make_basis(kernel, L=1.0, eps=1e-4)
    assert basis.mtot == basis.xis.shape[0]
    assert basis.mtot % 2 == 1
    assert basis.h > 0
    assert jnp.all(basis.weights >= 0)


def test_make_basis_matern():
    kernel = Matern(lengthscale=0.3, variance=1.0, nu=2.5, dim=1)
    basis = make_basis(kernel, L=1.0, eps=1e-4)
    assert basis.mtot == basis.xis.shape[0]
    assert basis.h > 0


def test_nufft_matches_dense():
    """NUFFT evaluation should match dense evaluation to high accuracy."""
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    basis = make_basis(kernel, L=1.0, eps=1e-4)
    x = jnp.linspace(0, 1, 500)

    Phi_dense = evaluate_basis_dense(basis, x)
    Phi_nufft = evaluate_basis_nufft(basis, x)

    assert Phi_nufft.shape == Phi_dense.shape
    assert jnp.allclose(Phi_nufft, Phi_dense, atol=1e-10)


def test_apply_basis_single():
    """apply_basis_nufft with a coefficient vector should match dense Phi @ c."""
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    basis = make_basis(kernel, L=1.0, eps=1e-4)
    x = jnp.linspace(0, 1, 200)
    coeffs = jax.random.normal(jax.random.PRNGKey(0), (basis.mtot,))

    vals = apply_basis_nufft(basis, x, coeffs)
    Phi = evaluate_basis_dense(basis, x)
    vals_ref = Phi @ coeffs

    assert jnp.allclose(vals, vals_ref, atol=1e-10)


def test_apply_basis_batched():
    """apply_basis_nufft should handle a batch of coefficient vectors."""
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    basis = make_basis(kernel, L=1.0, eps=1e-4)
    x = jnp.linspace(0, 1, 200)
    B = 4
    coeffs = jax.random.normal(jax.random.PRNGKey(1), (B, basis.mtot))

    vals = apply_basis_nufft(basis, x, coeffs)
    assert vals.shape == (B, x.shape[0])

    Phi = evaluate_basis_dense(basis, x)
    for i in range(B):
        assert jnp.allclose(vals[i], Phi @ coeffs[i], atol=1e-10)


def test_sample_gp_nufft_shape():
    """sample_gp_nufft should return the correct shapes."""
    kernel = SE(lengthscale=0.3, variance=1.0, dim=1)
    basis = make_basis(kernel, L=1.0, eps=1e-4)
    x = jnp.linspace(0, 1, 200)
    key = jax.random.PRNGKey(0)

    s1 = sample_gp_nufft(basis, x, key, n_samples=1)
    assert s1.shape == (200,)

    s5 = sample_gp_nufft(basis, x, key, n_samples=5)
    assert s5.shape == (5, 200)


def test_sample_gp_nufft_matches_dense():
    """NUFFT samples should match dense computation with the same random draw."""
    l, var = 0.3, 1.0
    kernel = SE(lengthscale=l, variance=var, dim=1)
    basis = make_basis(kernel, L=1.0, eps=1e-6)
    x = jnp.linspace(0, 1, 200)
    n_samples = 5
    key = jax.random.PRNGKey(7)

    # Reproduce what sample_gp_nufft does, but with dense matmul
    from efgp_jax.nufft import _cmplx
    cdtype = _cmplx(x.dtype)
    key1, key2 = jax.random.split(key)
    z_real = jax.random.normal(key1, shape=(n_samples, basis.mtot))
    z_imag = jax.random.normal(key2, shape=(n_samples, basis.mtot))
    z = (z_real + 1j * z_imag).astype(cdtype)

    Phi = evaluate_basis_dense(basis, x)  # (N, M), already includes weights
    dense_samples = jnp.real(Phi @ z.T).T  # (n_samples, N)

    nufft_samples = sample_gp_nufft(basis, x, key, n_samples=n_samples)

    assert jnp.allclose(nufft_samples, dense_samples, atol=1e-10)


def test_sample_gp_nufft_statistics():
    """Empirical mean and covariance of NUFFT samples should match the GP prior."""
    l, var = 0.3, 1.0
    kernel = SE(lengthscale=l, variance=var, dim=1)
    basis = make_basis(kernel, L=1.0, eps=1e-6)
    x = jnp.linspace(0.2, 0.8, 20)
    N = x.shape[0]
    n_samples = 50_000

    key = jax.random.PRNGKey(42)
    samples = sample_gp_nufft(basis, x, key, n_samples=n_samples)

    K_true = kernel(pairwise_distances(x[:, None], x[:, None]))

    # Check mean ~ 0 (standard error of mean is sqrt(var / n_samples) ~ 0.004)
    emp_mean = jnp.mean(samples, axis=0)
    assert jnp.all(jnp.abs(emp_mean) < 0.05), f"max |mean| = {jnp.max(jnp.abs(emp_mean))}"

    # Check covariance ~ K_true
    K_emp = jnp.cov(samples.T)
    rel_err = jnp.linalg.norm(K_emp - K_true) / jnp.linalg.norm(K_true)
    assert rel_err < 0.02, f"relative covariance error = {rel_err}"


def test_kernel_reconstruction():
    """Phi @ Phi^H should approximate the kernel matrix."""
    l, var = 0.3, 1.0
    kernel = SE(lengthscale=l, variance=var, dim=1)
    basis = make_basis(kernel, L=1.0, eps=1e-6)
    x = jnp.linspace(0.2, 0.8, 20)

    Phi = evaluate_basis_nufft(basis, x)
    K_approx = jnp.real(Phi @ jnp.conj(Phi).T)

    K_true = kernel(pairwise_distances(x[:, None], x[:, None]))

    assert jnp.allclose(K_approx, K_true, atol=1e-4)
