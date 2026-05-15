# Analytic Structural Physics (ASP) Core

[![PyPI version](https://badge.fury.io/py/asp-physics.svg)](https://badge.fury.io/py/asp-physics)
[![Rust](https://github.com/yourorg/asp-physics/actions/workflows/release.yml/badge.svg)](https://github.com/yourorg/asp-physics/actions)

**ASP** is a high-performance computational framework that translates the mathematical theory of resurgence, transseries, and exact WKB analysis into operational aerospace engineering and quantum physics. 

By treating physical observables as boundary values of resurgent analytic functions, ASP replaces standard discrete-time numerical integration (e.g., RK45) with **Adaptive Spectral Picard Iteration**. Singularities (such as the $1/r^2$ Kepler collision) are mapped out of the physical domain using Kustaanheimo-Stiefel (KS) regularizations, yielding entire functions that converge at exponential $\mathcal{O}(N \log N)$ rates.

## Features
* **Spectral Astrodynamics:** KS-regularized CR3BP propagation bypassing the limits of standard shooting methods.
* **Borel-Padé Engine:** Blind singularity detection extracting instanton actions from factorially divergent perturbative series.
* **Newton-Kantorovich Certification:** Machine-verified error bounds using interval arithmetic, guaranteeing trajectory fidelity for safety-critical missions.
* **Zero-Cost Python/Rust FFI:** Python-friendly API backed by a bare-metal Rust engine.

## Installation

```bash
pip install asp-physics