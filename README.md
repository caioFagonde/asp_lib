# Analytic Structural Physics (ASP) Core

[![PyPI version](https://badge.fury.io/py/asp-physics.svg)](https://badge.fury.io/py/asp-physics)
[![Rust](https://github.com/yourorg/asp-physics/actions/workflows/ci.yml/badge.svg)](https://github.com/yourorg/asp-physics/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ASP** is a high-performance computational framework that translates the mathematical theory of resurgence, transseries, and exact WKB analysis into operational aerospace engineering and quantum physics. 

By treating physical observables as boundary values of resurgent analytic functions, ASP replaces standard discrete-time numerical integration (e.g., RK45) with **Adaptive Spectral Picard Iteration**. Singularities (such as the $1/r^2$ Kepler collision) are mapped out of the physical domain using Kustaanheimo-Stiefel (KS) regularizations. This topologically removes the Borel singularity, yielding entire functions that converge at exponential $\mathcal{O}(N \log N)$ rates.

## Core Capabilities

* **Spectral Astrodynamics:** KS-regularized Circular Restricted Three-Body Problem (CR3BP) propagation. Bypasses the algebraic convergence limits and truncation errors of standard shooting methods.
* **Borel-Padé Engine:** Blind singularity detection extracting instanton actions from factorially divergent perturbative series using SVD-based robust Padé approximants and Froissart doublet filtering.
* **Newton-Kantorovich Certification:** Machine-verified error bounds utilizing strict interval arithmetic, guaranteeing trajectory fidelity for safety-critical missions.
* **Zero-Cost Python/Rust FFI:** Python-friendly API backed by a bare-metal Rust engine. Utilizes `rustdct` for $\mathcal{O}(N \log N)$ Chebyshev transforms and zero-copy `ndarray` views via `PyO3`.

## Installation

The package is distributed as a pre-compiled binary wheel. No Rust toolchain or Docker container is required for the core library.

```bash
pip install asp-physics