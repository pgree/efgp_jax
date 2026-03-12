"""Compare EFGP with and without diagonal preconditioning.

Runs the same time series pipeline (hyperparameter learning, prediction,
variance, sampling) twice — once without preconditioning and once with the
diagonal preconditioner — and reports timings and accuracy for each.

Usage:
    python examples/time_series_precond.py
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
# 1) Generate synthetic data (same as time_series.py)
# -----------------------------------------------------------------------
print("=" * 60)
print("Preconditioning comparison: EFGP with vs without diagonal PCG")
print("=" * 60)

np.random.seed(1)
n = 18_000
d = 1
true_lengthscale = 0.05
true_variance = 1.0
true_noise_var = 0.01

x_all = np.sort(np.random.rand(n))
kernel_true = SE(lengthscale=true_lengthscale, variance=true_variance, dim=d)
x_jnp = jnp.array(x_all)[:, None]
K = kernel_true(pairwise_distances(x_jnp, x_jnp))
K_noise = K + true_noise_var * jnp.eye(n)
L = jnp.linalg.cholesky(K_noise)
z = jnp.array(np.random.randn(n))
y_all = (L @ z)

gap_half = n // 10
mid = n // 2
keep = np.concatenate([np.arange(mid - gap_half), np.arange(mid + gap_half, n)])
x = jnp.array(x_all[keep])
y = jnp.array(np.array(y_all)[keep])
n_obs = x.shape[0]
print(f"Data: {n_obs} observations (gap of {2*gap_half} removed)")
print(f"True: l={true_lengthscale}, var={true_variance}, noise={true_noise_var}")

EPSILON = 1e-4
l0, var0, sig0 = estimate_hyperparameters(x, y)
kernel0 = SE(lengthscale=l0, variance=var0, dim=d)
print(f"Initial: l={l0:.4f}, var={var0:.4f}, noise={sig0:.4f}")

n_new = 1000
x_new = jnp.linspace(0, 1, n_new)
n_samples = 10
domain = (float(jnp.min(x)), float(jnp.max(x)))

results = {}

for label, use_precond in [("No preconditioner", False), ("Diagonal preconditioner", True)]:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # --- Hyperparameter learning ---
    key = jax.random.PRNGKey(0)
    t0 = time.time()
    kernel_learned, sig2_final, info = optimize_hyperparameters(
        x, y, kernel0=kernel0, sigmasq0=sig0, eps=EPSILON,
        domain=domain, key=key, maxiter=50, trace_samples=30,
        use_precond=use_precond,
    )
    t_opt = time.time() - t0
    l_final = kernel_learned.lengthscale
    var_final = kernel_learned.variance
    print(f"Learned: l={l_final:.4f}, var={var_final:.4f}, noise={sig2_final:.6f}"
          f"  ({info['nfev']} evals, {t_opt:.1f}s)")

    # --- Posterior prediction ---
    gp = EFGP(kernel_learned, domain=domain, eps=EPSILON, use_precond=use_precond)
    posterior = gp.condition(x, y, sig2_final)

    t0 = time.time()
    yhat, var = posterior.predict(x_new, return_var=True)
    t_pred = time.time() - t0
    sd = jnp.sqrt(var)
    print(f"Predict (mean + var): {t_pred:.4f}s")

    # --- Posterior sampling ---
    key, subkey = jax.random.split(jax.random.PRNGKey(0))
    _, subkey = jax.random.split(subkey)
    t0 = time.time()
    samples = posterior.sample(x_new, key=subkey, n_samples=n_samples)
    t_sample = time.time() - t0
    samples = samples.T
    print(f"Sample ({n_samples}): {t_sample:.4f}s")

    results[label] = {
        "t_opt": t_opt, "t_pred": t_pred, "t_sample": t_sample,
        "l": l_final, "kvar": var_final, "sig2": sig2_final,
        "nfev": info["nfev"], "nll": info["nll"],
        "yhat": yhat, "pred_var": var, "sd": sd, "samples": samples,
    }

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")

r_no = results["No preconditioner"]
r_pc = results["Diagonal preconditioner"]

print(f"\n{'Metric':<30s} {'No precond':>14s} {'Diag precond':>14s} {'Speedup':>10s}")
print("-" * 70)
for name, key in [("Optimization (s)", "t_opt"),
                   ("Predict+Var (s)", "t_pred"),
                   ("Sampling (s)", "t_sample")]:
    v1, v2 = r_no[key], r_pc[key]
    print(f"{name:<30s} {v1:>14.2f} {v2:>14.2f} {v1/v2:>9.1f}x")

print(f"\n{'Hyperparameter':<30s} {'No precond':>14s} {'Diag precond':>14s}")
print("-" * 60)
print(f"{'lengthscale':<30s} {r_no['l']:>14.4f} {r_pc['l']:>14.4f}")
print(f"{'variance':<30s} {r_no['kvar']:>14.4f} {r_pc['kvar']:>14.4f}")
print(f"{'noise var':<30s} {r_no['sig2']:>14.6f} {r_pc['sig2']:>14.6f}")
print(f"{'NLL':<30s} {r_no['nll']:>14.2f} {r_pc['nll']:>14.2f}")
print(f"{'L-BFGS evals':<30s} {r_no['nfev']:>14d} {r_pc['nfev']:>14d}")

# Accuracy comparison: how close are the predictions?
mean_diff = float(jnp.max(jnp.abs(r_no["yhat"] - r_pc["yhat"])))
var_diff = float(jnp.max(jnp.abs(r_no["pred_var"] - r_pc["pred_var"])))
print(f"\nMax |mean difference|:     {mean_diff:.2e}")
print(f"Max |variance difference|: {var_diff:.2e}")

# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for col, (label, r) in enumerate(results.items()):
    ax = axes[0, col]
    ax.scatter(np.array(x), np.array(y), s=1, alpha=0.3, color="blue")
    ax.plot(np.array(x_new), np.array(r["yhat"]), color="red", linewidth=1.5)
    ax.fill_between(
        np.array(x_new),
        np.array(r["yhat"] - 2 * r["sd"]),
        np.array(r["yhat"] + 2 * r["sd"]),
        alpha=0.3, color="red",
    )
    ax.set_title(f"{label}\nopt={r['t_opt']:.1f}s  pred={r['t_pred']:.2f}s  sample={r['t_sample']:.2f}s")
    ax.set_ylabel("y")

    ax = axes[1, col]
    ax.scatter(np.array(x), np.array(y), s=1, alpha=0.3, color="blue")
    ax.plot(np.array(x_new), np.array(r["yhat"]), color="red", linewidth=1.5)
    for i in range(n_samples):
        ax.plot(np.array(x_new), np.array(r["samples"][:, i]),
                color="green", linewidth=0.5, alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "time_series_precond.png")
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
