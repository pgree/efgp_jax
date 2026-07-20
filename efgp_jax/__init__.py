"""EFGP JAX: Equispaced Fourier Gaussian Process regression in JAX."""

from .kernels import Kernel, SE, Matern
from .efgp import EFGP, EFGPPosterior
from .optimize import optimize_hyperparameters, optimize_hyperparameters_autodiff
from .fixed_freq import (
    optimize_hyperparameters_fixed_freq,
    build_fixed_grid,
)
