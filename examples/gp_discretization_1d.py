"""1D GP discretization via equispaced Fourier expansion.

Demonstrates the spectral representation of a 1D GP without any data or
hyperparameter learning.  Plots basis functions, kernel approximation,
spectral density with quadrature nodes, and prior samples.

Usage:
    python examples/gp_discretization_1d.py
"""

import os
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from efgp_jax.kernels import SE
from efgp_jax.efgp import EFGP

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------
domain = (0, 1)
lengthscale = 0.05
variance = 1.0
eps = 1e-4

kernel = SE(lengthscale=lengthscale, variance=variance, dim=1)
gp = EFGP(kernel, domain=domain, eps=eps)

xis = gp.xis.ravel()
ws = gp.ws
M = gp.M

print(f"Kernel: SE(l={lengthscale}, var={variance}, eps={eps})")
print(f"Domain: {domain},  eps={eps}")
print(f"Spectral grid: M={M} frequencies, spacing h={gp.h:.6f}")

x = jnp.linspace(domain[0], domain[1], 1000)

# -----------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# --- Panel 1: First few basis functions (cos and sin parts) ---
ax = axes[0, 0]
n_show = 5
pos_idx = jnp.where(xis > 0)[0][:n_show]
cos_part, sin_part = gp.eval_basis(x, indices=pos_idx)
for i in range(len(pos_idx)):
    ax.plot(np.array(x), np.array(cos_part[:, i]),
            label=rf"$w_j \cos(2\pi \xi_{{{i+1}}} x)$", alpha=0.8)
    ax.plot(np.array(x), np.array(sin_part[:, i]),
            linestyle="--", alpha=0.5)
ax.set_xlabel("x")
ax.set_ylabel("basis function value")
ax.set_title(f"First {n_show} basis functions (cos solid, sin dashed)")
ax.legend(fontsize=7, loc="upper right")

# --- Panel 2: Exact kernel vs spectral approximation ---
ax = axes[0, 1]
r = jnp.linspace(0, gp.L, 500)
K_exact = kernel(r)
ws_sq = jnp.real(ws ** 2)
K_approx = jnp.array([jnp.sum(ws_sq * jnp.cos(2 * math.pi * xis * float(ri)))
                       for ri in r])

ax.plot(np.array(r), np.array(K_exact), 'k-', linewidth=2, label="exact $k(r)$")
ax.plot(np.array(r), np.array(K_approx), 'r--', linewidth=1.5,
        label=f"spectral approx (M={M})")
ax.set_xlabel("r")
ax.set_ylabel("k(r)")
ax.set_title("Kernel: exact vs spectral approximation")
ax.legend(fontsize=8)

# --- Panel 3: Spectral density + quadrature nodes ---
ax = axes[1, 0]
xi_dense = jnp.linspace(float(xis[0]) * 1.2, float(xis[-1]) * 1.2, 500)
S_dense = kernel.spectral_density(xi_dense)

ax.plot(np.array(xi_dense), np.array(S_dense), 'k-', linewidth=1.5,
        label="$S(\\xi)$")
ax.stem(np.array(xis), np.array(kernel.spectral_density(xis)),
        linefmt='r-', markerfmt='ro', basefmt=' ', label=f"quadrature nodes (M={M})")
ax.set_xlabel(r"$\xi$ (frequency)")
ax.set_ylabel(r"$S(\xi)$")
ax.set_title("Spectral density and quadrature nodes")
ax.legend(fontsize=8)

# --- Panel 4: Prior samples ---
ax = axes[1, 1]
n_samples = 5
key = jax.random.PRNGKey(42)
samples = gp.sample(x, key, n_samples=n_samples)

for i in range(n_samples):
    ax.plot(np.array(x), np.array(samples[i]), linewidth=0.8, alpha=0.8)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title(f"Prior samples (SE, l={lengthscale})")

fig.suptitle(f"EFGP spectral discretization:  SE kernel,  l={lengthscale},  "
             f"var={variance},  eps={eps:0.1e}, M={M} modes", fontsize=13, y=1.01)
fig.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "gp_discretization_1d.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved plot to {out_path}")
