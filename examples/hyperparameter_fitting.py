"""Hyperparameter fitting example.

Generates data from a GP prior with KNOWN ("true") hyperparameters, then starts
the optimizer from deliberately-wrong initial values and checks that minimizing
the negative log marginal likelihood recovers the truth. Finally conditions on
the data with the fitted hyperparameters and plots the data, posterior mean,
+/- 2 sigma band, and posterior samples.

The domain is chosen to span many lengthscales so that all three
hyperparameters (lengthscale, variance, noise) are individually identifiable;
on a domain only a few lengthscales wide, variance and lengthscale are not
separately identifiable from a single realization.

Usage:
    python examples/hyperparameter_fitting.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from efgp_jax.kernels import SE
from efgp_jax.efgp import EFGP
from efgp_jax.optimize import optimize_hyperparameters

# -----------------------------------------------------------------------
# 1) Generate data from a GP prior with KNOWN ("true") hyperparameters
# -----------------------------------------------------------------------
l_true, var_true, sigmasq_true = 0.15, 2.0, 0.05
domain = (0, 10)  # spans ~67 lengthscales so all 3 params are identifiable
x = jnp.linspace(*domain, 4000)
kernel_true = SE(lengthscale=l_true, variance=var_true, dim=1)
f = EFGP(kernel_true, domain=domain, eps=1e-6).sample(x, key=jax.random.PRNGKey(0))
y = f + jnp.sqrt(sigmasq_true) * jax.random.normal(jax.random.PRNGKey(1), x.shape)

# -----------------------------------------------------------------------
# 2) Fit hyperparameters from deliberately-wrong initial values
# -----------------------------------------------------------------------
kernel0 = SE(lengthscale=0.5, variance=0.5, dim=1)
sigmasq0 = 0.2
kernel, sigmasq, info = optimize_hyperparameters(
    x, y, kernel0, sigmasq0=sigmasq0, eps=1e-4,
    domain=domain, key=jax.random.PRNGKey(2),
    trace_samples=50, verbose=False,
)

print(f"{'param':<12}{'true':>10}{'init':>10}{'recovered':>12}")
print(f"{'lengthscale':<12}{l_true:>10.4f}{kernel0.lengthscale:>10.4f}{kernel.lengthscale:>12.4f}")
print(f"{'variance':<12}{var_true:>10.4f}{kernel0.variance:>10.4f}{kernel.variance:>12.4f}")
print(f"{'sigmasq':<12}{sigmasq_true:>10.4f}{sigmasq0:>10.4f}{sigmasq:>12.4f}")
print(f"\nNLL={info['nll']:.2f}  nfev={info['nfev']}  success={info['success']}")

# -----------------------------------------------------------------------
# 3) Condition on the data with the fitted hyperparameters
# -----------------------------------------------------------------------
gp = EFGP(kernel, domain=domain, eps=1e-4)
posterior = gp.condition(x, y, sigmasq=sigmasq)

x_new = jnp.linspace(*domain, 500)
yhat, var = posterior.predict(x_new, return_var=True)
sd = jnp.sqrt(jnp.clip(var, a_min=0.0))
samples = posterior.sample(x_new, key=jax.random.PRNGKey(3), n_samples=10)

# -----------------------------------------------------------------------
# 4) Plot
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, s=5, color="0.6", alpha=0.4, label="data")
ax.plot(x_new, samples.T, color="C1", lw=0.7, alpha=0.5)
ax.plot([], [], color="C1", lw=0.7, label="posterior samples")
ax.fill_between(x_new, yhat - 2 * sd, yhat + 2 * sd,
                color="C0", alpha=0.2, label=r"$\pm 2\sigma$")
ax.plot(x_new, yhat, color="C0", lw=2, label="posterior mean")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

fig.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "hyperparameter_fitting.png")
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
