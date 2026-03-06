"""EFGP JAX: Equispaced Fourier Gaussian Process regression in JAX."""

from .kernels import Kernel, SE, Matern
from .efgp import EFGP, EFGPPosterior
