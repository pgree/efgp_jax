# EFGP JAX

This repo was originally built by prompting Claude Code to convert [this](https://github.com/danbider/gp-quadrature)
PyTorch-based code to JAX.

This repo includes code for:
- discretizing Gaussian processes (GPs) into Fourier expansions with Gaussian coefficients
- GP regression in O(M log M + N) operations
- hyperparameter learning via optimizing negative log marginal likelihood with stochastic gradient estimation

Based on the method described in:

> Philip Greengard, Manas Rachh, Alex H Barnett, "Equispaced Fourier representations for efficient Gaussian process regression from a billion data points," *SIAM/ASA Journal on Uncertainty Quantification*, 2025.

## Installation

```bash
pip install -e .
```

### Dependencies

- `jax >= 0.4.0`
- `jaxlib >= 0.4.0`
- `jax-finufft >= 0.1.0`
- `numpy >= 1.21.0`
- `scipy >= 1.7.0`

Optional:
- `matplotlib` (plotting)
- `pytest` (testing)

## Quick start

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from efgp_jax.kernels import SE
from efgp_jax.efgp import EFGP

# Training data
x = jnp.linspace(0, 1, 1000)
y = jnp.sin(6 * x) + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (1000,))

# Build EFGP model and condition on data
kernel = SE(lengthscale=0.1, variance=1.0, dim=1)
gp = EFGP(kernel, domain=(0, 1), eps=1e-4)
posterior = gp.condition(x, y, sigmasq=0.01)

# Posterior mean and variance
x_new = jnp.linspace(0, 1, 500)
yhat, var = posterior.predict(x_new, return_var=True)

# Posterior samples
samples = posterior.sample(x_new, key=jax.random.PRNGKey(1), n_samples=10)
```

## Hyperparameter learning

The `examples/time_series.py` script demonstrates the full pipeline: data generation, hyperparameter optimization via EFGP + L-BFGS, and posterior prediction with sampling.

```bash
python examples/time_series.py
```

## GP discretization

The `examples/gp_discretization_1d.py` script visualizes the spectral representation: basis functions, kernel approximation, spectral density with quadrature nodes, and prior samples. A 2D version is in `examples/gp_discretization_2d.py`.

```bash
python examples/gp_discretization_1d.py
```

## Testing

```bash
pytest tests/
```

## Package structure

```
efgp_jax/
  kernels.py       # SE and Matern kernels with spectral densities
  quadrature.py    # Frequency grid generation
  nufft.py         # jax-finufft wrapper (Type-1 and Type-2 NUFFT)
  basis_eval.py    # NUFFT-accelerated Fourier basis evaluation
  toeplitz.py      # FFT-based Toeplitz matrix-vector products
  cg.py            # Conjugate gradient solver (JIT-compatible)
  efgp.py          # Core: EFGP prior, EFGPPosterior, gradient, predict, variance, sample
  gp.py            # Dense GP methods (reference implementations)
  optimize.py      # Hyperparameter optimization via L-BFGS
tests/             # Test suite
examples/          # End-to-end examples
```

## License

MIT
