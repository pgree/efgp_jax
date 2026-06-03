"""Time series hyperparameter learning example (JAX port of gp-quadrature).

Generates synthetic GP data with a gap, learns hyperparameters via EFGP
+ L-BFGS, then plots posterior mean, samples, and variance.

Usage:
    python examples/time_series.py
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from efgp_jax.kernels import SE, pairwise_distances, estimate_hyperparameters
from efgp_jax.efgp import EFGP
from efgp_jax.optimize import optimize_hyperparameters

# -----------------------------------------------------------------------
# 1) Generate synthetic data
# -----------------------------------------------------------------------
print("=" * 60)
print("Time series example: hyperparameter learning with EFGP")
print("=" * 60)

np.random.seed(1)
n = 1_000
d = 1
true_lengthscale = 0.1
true_variance = 1.0
true_noise_var = 0.01  # (0.1)^2

# Random points on [0, 1], sorted
x_all = np.sort(np.random.rand(n))

# Sample from GP prior via Cholesky
kernel_true = SE(lengthscale=true_lengthscale, variance=true_variance, dim=d)
x_jnp = jnp.array(x_all)[:, None]
K = kernel_true(pairwise_distances(x_jnp, x_jnp))
K_noise = K + true_noise_var * jnp.eye(n)
L = jnp.linalg.cholesky(K_noise)
z = jnp.array(np.random.randn(n))
y_all = (L @ z)

# Remove points in the middle to create a gap
gap_half = n // 10
mid = n // 2
keep = np.concatenate([np.arange(mid - gap_half), np.arange(mid + gap_half, n)])
x = jnp.array(x_all[keep])
y = jnp.array(np.array(y_all)[keep])
n_obs = x.shape[0]
print(f"Data: {n_obs} observations (removed gap of {2*gap_half} points in center)")
print(f"True hyperparameters: l={true_lengthscale}, var={true_variance}, "
      f"noise={true_noise_var}")

# -----------------------------------------------------------------------
# 2) Hyperparameter learning
# -----------------------------------------------------------------------
print("\n--- Hyperparameter learning (L-BFGS) ---")
EPSILON = 1e-4

# Initialize from data-driven heuristic
l0, var0, sig0 = estimate_hyperparameters(x, y)
kernel0 = SE(lengthscale=l0, variance=var0, dim=d)
print(f"Initial hyperparameters: l={l0:.4f}, var={var0:.4f}, noise={sig0:.4f}")

key = jax.random.PRNGKey(0)
t0 = time.time()
kernel_learned, sig2_final, info = optimize_hyperparameters(
    x, y, kernel0=kernel0, sigmasq0=sig0, eps=EPSILON,
    key=key, maxiter=50, trace_samples=30,
)
t_opt = time.time() - t0

l_final = kernel_learned.lengthscale
var_final = kernel_learned.variance
print(f"\nLearned: l={l_final:.4f}, var={var_final:.4f}, noise={sig2_final:.6f}"
      f"  ({info['nfev']} evals, {t_opt:.1f}s)")

# -----------------------------------------------------------------------
# 3) Build EFGP model with learned hyperparameters
# -----------------------------------------------------------------------
print("\n--- Posterior prediction ---")
n_new = 1000
x_new = jnp.linspace(0, 1, n_new)

gp = EFGP(kernel_learned, domain=(float(jnp.min(x)), float(jnp.max(x))), eps=EPSILON)
posterior = gp.condition(x, y, sig2_final)

# -----------------------------------------------------------------------
# 4) Posterior mean + variance
# -----------------------------------------------------------------------
t0 = time.time()
yhat, var = posterior.predict(x_new, return_var=True)
t_var = time.time() - t0
print(f"Posterior mean + variance: {t_var:.4f}s")

sd = jnp.sqrt(var)

# -----------------------------------------------------------------------
# 5) Posterior samples (EFGP, uses all training data)
# -----------------------------------------------------------------------
print("\n--- Posterior sampling (EFGP Matheron rule) ---")
n_samples = 10
key, subkey = jax.random.split(key)
t0 = time.time()
samples = posterior.sample(x_new, key=subkey, n_samples=n_samples)
t_sample = time.time() - t0
print(f"Posterior samples ({n_samples}): {t_sample:.4f}s")
samples = samples.T  # (n_new, n_samples) for plotting

# -----------------------------------------------------------------------
# 6) Plots
# -----------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Panel 1: Data + posterior mean
ax = axes[0]
ax.scatter(np.array(x), np.array(y), s=1, alpha=0.3, color="blue", label="observations")
ax.plot(np.array(x_new), np.array(yhat), color="red", linewidth=1.5, label="posterior mean")
ax.set_ylabel("y")
ax.set_title(f"Learned: l={l_final:.3f}, var={var_final:.3f}, noise={sig2_final:.5f}")
ax.legend(fontsize=8)

# Panel 2: Mean + posterior samples
ax = axes[1]
ax.scatter(np.array(x), np.array(y), s=1, alpha=0.3, color="blue", label="observations")
ax.plot(np.array(x_new), np.array(yhat), color="red", linewidth=1.5, label="posterior mean")
x_np = np.array(x_new)
for i in range(n_samples):
    label = "posterior samples" if i == 0 else None
    ax.plot(x_np, np.array(samples[:, i]), color="green", linewidth=0.5, alpha=0.7, label=label)
ax.set_ylabel("y")
ax.legend(fontsize=8)

# Panel 3: Mean + confidence band
ax = axes[2]
ax.scatter(np.array(x), np.array(y), s=1, alpha=0.3, color="blue", label="observations")
ax.plot(x_np, np.array(yhat), color="red", linewidth=1.5, label="posterior mean")
ax.fill_between(
    x_np,
    np.array(yhat - 2 * sd),
    np.array(yhat + 2 * sd),
    alpha=0.3, color="red", label=r"$\pm 2\sigma$",
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(fontsize=8)

fig.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "time_series.png")
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
