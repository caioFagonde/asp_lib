# Analytic Structural Physics (ASP) Core Engine v1.0.0

[![PyPI version](https://badge.fury.io/py/asp-physics.svg)](https://badge.fury.io/py/asp-physics)
[![Rust](https://github.com/yourorg/asp-physics/actions/workflows/ci.yml/badge.svg)](https://github.com/yourorg/asp-physics/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ASP** is a high-performance computational framework that translates the mathematical theory of resurgence, transseries, and exact WKB analysis into operational aerospace engineering and quantum physics.

By treating physical observables as boundary values of resurgent analytic functions, ASP replaces standard discrete-time numerical integration (e.g., RK45) with **Adaptive Spectral Picard Iteration**. Singularities (such as the $1/r^2$ Kepler collision) are mapped out of the physical domain using Kustaanheimo-Stiefel (KS) regularizations. This topologically removes the Borel singularity, yielding entire functions that converge at exponential $\mathcal{O}(N \log N)$ rates.

## System Architecture

The ASP engine is a tri-layer polyglot system:
1. **Python (Orchestrator)**: High-level API, SymPy AST generation, and Borel plane topological analysis.
2. **Rust (Core)**: Bare-metal execution of $\mathcal{O}(N \log N)$ Chebyshev transforms, strict Newton-Kantorovich interval arithmetic, and Udwadia-Kalaba exact constraint projections. Bound to Python via zero-copy `PyO3`/`maturin`.
3. **Go (Cluster)**: gRPC-based distributed microservice cluster for massive trajectory fan-out (Arbitrary Polynomial Chaos).

## Core Capabilities

* **Spectral Astrodynamics:** KS-regularized Circular Restricted Three-Body Problem (CR3BP) propagation. Bypasses the algebraic convergence limits of standard shooting methods.
* **Borel-Padé Engine:** Blind singularity detection extracting instanton actions from factorially divergent perturbative series using SVD-based robust Padé approximants and Froissart doublet filtering.
* **Newton-Kantorovich Certification:** Machine-verified error bounds utilizing strict interval arithmetic, guaranteeing trajectory fidelity for safety-critical missions.
* **Arbitrary Polynomial Chaos (APC):** Distributed stochastic uncertainty quantification bypassing Monte Carlo limits via exact orthogonal polynomial collocation.

## Installation

### Prerequisites
* Python 3.10+
* Rust Toolchain (1.78+)
* Go 1.21+
* Protobuf Compiler (`protoc`)
* Docker & Docker Compose (for distributed cluster)

### Automated Setup
For UNIX-based systems (Linux/macOS), run the provided installation script to automatically provision the environment, compile the gRPC stubs, and build the Rust extension:

```bash
chmod +x install.sh
./install.sh