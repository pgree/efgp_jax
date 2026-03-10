"""2D GP discretization via equispaced Fourier expansion.

Demonstrates the spectral representation of a 2D GP without any data or
hyperparameter learning.  Plots the 2D spectral density with quadrature
nodes, kernel slices (exact vs approximate), and prior samples on a grid.
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
L = 1.0 # GP defined on [0, L]^2
lengthscale = 0.1
variance = 1.0
eps = 1e-3

kernel = SE(lengthscale=lengthscale, variance=variance, dim=2)
gp = EFGP(kernel, L=L, eps=eps)

xis = gp.xis  # (M, 2)
ws = gp.ws
M = gp.M
mtot = gp.mtot

print(f"Kernel: SE(l={lengthscale}, var={variance}, dim=2, eps={eps})")
print(f"Domain: [0, {L}]^2")
print(f"Spectral grid: {mtot} x {mtot} = {M} frequencies, spacing h={gp.h:.6f}")

# Evaluation grid
n_grid = 200
x1 = jnp.linspace(0, L, n_grid)
x2 = jnp.linspace(0, L, n_grid)
X1, X2 = jnp.meshgrid(x1, x2, indexing="ij")
x_flat = jnp.stack([X1.ravel(), X2.ravel()], axis=-1)  # (n_grid^2, 2)

# -----------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# --- Panel 1: 2D spectral density + quadrature nodes ---
ax = axes[0, 0]
xi_range = float(jnp.max(jnp.abs(xis))) * 1.1
xi_dense = jnp.linspace(-xi_range, xi_range, 200)
XI1, XI2 = jnp.meshgrid(xi_dense, xi_dense, indexing="ij")
xi_grid = jnp.stack([XI1.ravel(), XI2.ravel()], axis=-1)
S_grid = kernel.spectral_density(xi_grid).reshape(200, 200)

im = ax.pcolormesh(np.array(xi_dense), np.array(xi_dense),
                   np.array(S_grid).T, shading="auto", cmap="viridis")
ax.scatter(np.array(xis[:, 0]), np.array(xis[:, 1]),
           s=1, color="red", alpha=0.5, label=f"nodes ({M})")
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_xlabel(r"$\xi_1$")
ax.set_ylabel(r"$\xi_2$")
ax.set_title(r"Spectral density $S(\xi)$ + quadrature nodes")
ax.legend(fontsize=8, markerscale=5)
ax.set_aspect("equal")

# --- Panel 2: Kernel slice along x1 (x2=0) ---
ax = axes[0, 1]
r = jnp.linspace(0, L, 300)
K_exact = kernel(r)

# Spectral approximation: k(r,0) = sum_j |w_j|^2 cos(2pi xi_j1 * r)
ws_sq = jnp.real(ws ** 2)  # (M,)
xi1 = xis[:, 0]  # (M,)
K_approx = jnp.array([jnp.sum(ws_sq * jnp.cos(2 * math.pi * xi1 * float(ri)))
                       for ri in r])

ax.plot(np.array(r), np.array(K_exact), 'k-', linewidth=2, label="exact $k(r, 0)$")
ax.plot(np.array(r), np.array(K_approx), 'r--', linewidth=1.5,
        label=f"spectral approx (M={M})")
ax.set_xlabel("$r$")
ax.set_ylabel("$k(r, 0)$")
ax.set_title("Kernel slice along $x_1$ axis")
ax.legend(fontsize=8)

# --- Panel 3: Kernel slice along diagonal (x1=x2=r/sqrt(2)) ---
ax = axes[1, 0]
r_diag = jnp.linspace(0, L * math.sqrt(2), 300)
K_exact_diag = kernel(r_diag)

# Spectral approx along diagonal: k(r/sqrt(2), r/sqrt(2))
# = sum_j |w_j|^2 cos(2pi (xi_j1 + xi_j2) * r / sqrt(2))
xi_sum = xis[:, 0] + xis[:, 1]
K_approx_diag = jnp.array([
    jnp.sum(ws_sq * jnp.cos(2 * math.pi * xi_sum * float(ri) / math.sqrt(2)))
    for ri in r_diag
])

ax.plot(np.array(r_diag), np.array(K_exact_diag), 'k-', linewidth=2,
        label=r"exact $k(r)$, diagonal")
ax.plot(np.array(r_diag), np.array(K_approx_diag), 'r--', linewidth=1.5,
        label=f"spectral approx (M={M})")
ax.set_xlabel("$r$")
ax.set_ylabel(r"$k(r/\sqrt{2},\, r/\sqrt{2})$")
ax.set_title("Kernel slice along diagonal")
ax.legend(fontsize=8)

# --- Panel 4: Prior samples ---
ax = axes[1, 1]
n_samples = 1
key = jax.random.PRNGKey(42)
samples = gp.sample(x_flat, key, n_samples=n_samples)  # (n_grid^2,)
sample_grid = np.array(samples.reshape(n_grid, n_grid))

im = ax.pcolormesh(np.array(x1), np.array(x2), sample_grid.T,
                   shading="auto", cmap="RdBu_r")
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title(f"Prior sample (SE, l={lengthscale})")
ax.set_aspect("equal")

fig.suptitle(f"EFGP 2D spectral discretization (isotropic):  SE kernel,  l={lengthscale},  "
             f"var={variance},  eps={eps:0.1e},  M={M} modes", fontsize=13, y=1.01)
fig.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "gp_discretization_2d.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved isotropic plot to {out_path}")


# -----------------------------------------------------------------------
# Anisotropic example
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("Anisotropic kernel")
print("=" * 60)

l_aniso = [0.05, 0.2]
kernel_aniso = SE(lengthscale=l_aniso, variance=variance, dim=2)
gp_aniso = EFGP(kernel_aniso, L=L, eps=eps)

xis_a = gp_aniso.xis
ws_a = gp_aniso.ws
M_a = gp_aniso.M

print(f"Kernel: SE(l={l_aniso}, var={variance}, dim=2, eps={eps})")
print(f"Spectral grid: {gp_aniso.OUT[0]} x {gp_aniso.OUT[1]} = {M_a} frequencies")

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

# --- Panel 1: 2D spectral density + quadrature nodes ---
ax = axes2[0, 0]
xi_range_a = float(jnp.max(jnp.abs(xis_a))) * 1.1
xi_dense_a = jnp.linspace(-xi_range_a, xi_range_a, 200)
XI1a, XI2a = jnp.meshgrid(xi_dense_a, xi_dense_a, indexing="ij")
xi_grid_a = jnp.stack([XI1a.ravel(), XI2a.ravel()], axis=-1)
S_grid_a = kernel_aniso.spectral_density(xi_grid_a).reshape(200, 200)

im = ax.pcolormesh(np.array(xi_dense_a), np.array(xi_dense_a),
                   np.array(S_grid_a).T, shading="auto", cmap="viridis")
ax.scatter(np.array(xis_a[:, 0]), np.array(xis_a[:, 1]),
           s=1, color="red", alpha=0.5, label=f"nodes ({M_a})")
fig2.colorbar(im, ax=ax, shrink=0.8)
ax.set_xlabel(r"$\xi_1$")
ax.set_ylabel(r"$\xi_2$")
ax.set_title(r"Spectral density $S(\xi)$ + quadrature nodes")
ax.legend(fontsize=8, markerscale=5)
ax.set_aspect("equal")

# --- Panel 2: Kernel slices along each axis ---
ax = axes2[0, 1]
r = jnp.linspace(0, L, 300)
# Exact anisotropic: k(r,0) = var * exp(-0.5 * r^2 / l1^2)
K_exact_x1 = variance * jnp.exp(-0.5 * r ** 2 / l_aniso[0] ** 2)
K_exact_x2 = variance * jnp.exp(-0.5 * r ** 2 / l_aniso[1] ** 2)

ws_sq_a = jnp.real(ws_a ** 2)
xi1_a = xis_a[:, 0]
xi2_a = xis_a[:, 1]
K_approx_x1 = jnp.array([jnp.sum(ws_sq_a * jnp.cos(2 * math.pi * xi1_a * float(ri)))
                           for ri in r])
K_approx_x2 = jnp.array([jnp.sum(ws_sq_a * jnp.cos(2 * math.pi * xi2_a * float(ri)))
                           for ri in r])

ax.plot(np.array(r), np.array(K_exact_x1), 'k-', linewidth=2, label=f"exact $k(r,0)$, $l_1$={l_aniso[0]}")
ax.plot(np.array(r), np.array(K_approx_x1), 'r--', linewidth=1.5, label="approx $k(r,0)$")
ax.plot(np.array(r), np.array(K_exact_x2), 'k-', linewidth=2, alpha=0.5, label=f"exact $k(0,r)$, $l_2$={l_aniso[1]}")
ax.plot(np.array(r), np.array(K_approx_x2), 'b--', linewidth=1.5, alpha=0.7, label="approx $k(0,r)$")
ax.set_xlabel("$r$")
ax.set_ylabel("$k$")
ax.set_title("Kernel slices along each axis")
ax.legend(fontsize=7)

# --- Panel 3: Isotropic prior sample for comparison ---
ax = axes2[1, 0]
key = jax.random.PRNGKey(42)
sample_iso = gp.sample(x_flat, key, n_samples=1)
sample_iso_grid = np.array(sample_iso.reshape(n_grid, n_grid))

im = ax.pcolormesh(np.array(x1), np.array(x2), sample_iso_grid.T,
                   shading="auto", cmap="RdBu_r")
fig2.colorbar(im, ax=ax, shrink=0.8)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title(f"Isotropic prior sample (l={lengthscale})")
ax.set_aspect("equal")

# --- Panel 4: Anisotropic prior sample ---
ax = axes2[1, 1]
sample_aniso = gp_aniso.sample(x_flat, key, n_samples=1)
sample_aniso_grid = np.array(sample_aniso.reshape(n_grid, n_grid))

im = ax.pcolormesh(np.array(x1), np.array(x2), sample_aniso_grid.T,
                   shading="auto", cmap="RdBu_r")
fig2.colorbar(im, ax=ax, shrink=0.8)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title(f"Anisotropic prior sample (l={l_aniso})")
ax.set_aspect("equal")

fig2.suptitle(f"EFGP 2D anisotropic:  SE kernel,  l={l_aniso},  "
              f"var={variance},  eps={eps:0.1e},  M={M_a} modes", fontsize=13, y=1.01)
fig2.tight_layout()
out_path2 = os.path.join(SCRIPT_DIR, "gp_discretization_2d_aniso.png")
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved anisotropic plot to {out_path2}")
