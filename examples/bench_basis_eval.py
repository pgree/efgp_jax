"""Benchmark dense matvec vs NUFFT for evaluating a Fourier expansion.

Uses EFGP to set up the spectral frequencies and weights, then compares:
  - Dense:  Phi @ c  where Phi is the N x M basis matrix (O(NM))
  - NUFFT:  type-2 NUFFT evaluation (O(M log M + N))

M is controlled by the lengthscale.

Usage:
    python examples/bench_basis_eval.py
"""

import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from efgp_jax.kernels import SE
from efgp_jax.efgp import EFGP
from efgp_jax.nufft import _make_phi, nufft_type2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TWO_PI = 2 * math.pi


def bench(fn, warmup=2, repeats=5):
    """Time a zero-arg function, returning median wall-clock seconds."""
    for _ in range(warmup):
        out = fn()
        jax.block_until_ready(out)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


# --- Configuration ---
Ns_all = [500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 10_000_000]
N_dense_max = 100_000  # skip dense for N above this

configs = [
    {"lengthscale": 0.005, "eps": 1e-6},
    {"lengthscale": 0.001, "eps": 1e-6},
]

fig, ax = plt.subplots(figsize=(8, 5))

for cfg in configs:
    kernel = SE(lengthscale=cfg["lengthscale"], variance=1.0, dim=1)
    gp = EFGP(kernel, domain=(0, 1), eps=cfg["eps"])
    M = gp.M
    print(f"l={cfg['lengthscale']} -> M={M}")

    coeffs = jnp.ones(M, dtype=gp.cdtype)
    fk = (gp.ws * coeffs).reshape(gp.OUT)

    Ns_dense = []
    times_dense = []
    Ns_nufft = []
    times_nufft = []

    for N in Ns_all:
        x = jnp.linspace(0, 1, N)[:, None]
        phi = _make_phi(x, gp.xcen, gp.h)

        # NUFFT: always run
        t_nufft = bench(lambda: jnp.real(nufft_type2(phi, fk)))
        Ns_nufft.append(N)
        times_nufft.append(t_nufft)

        # Dense: skip for large N
        if N <= N_dense_max:
            xis_flat = gp.xis.reshape(-1, gp.d)
            Phi = jnp.exp(TWO_PI * 1j * (x @ xis_flat.T)).astype(gp.cdtype)
            wc = gp.ws * coeffs
            t_dense = bench(lambda: jnp.real(Phi @ wc))
            Ns_dense.append(N)
            times_dense.append(t_dense)
            speedup = t_dense / t_nufft
            print(f"  N={N:>10d}  dense={t_dense:.5f}s  nufft={t_nufft:.5f}s  speedup={speedup:.1f}x")
        else:
            print(f"  N={N:>10d}  dense=skipped       nufft={t_nufft:.5f}s")

    ax.plot(Ns_dense, times_dense, "o--", label=f"Dense matvec  (M={M})", alpha=0.7)
    ax.plot(Ns_nufft, times_nufft, "s-",  label=f"NUFFT apply   (M={M})", alpha=0.7)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("N (number of target points)")
ax.set_ylabel("Time (seconds)")
ax.set_title("Fourier expansion evaluation: Dense (O(NM)) vs NUFFT (O(M log M + N))")
ax.legend(fontsize=7)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "bench_basis_eval.png")
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
