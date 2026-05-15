# tests/test_core.py

import math
import numpy as np
import pytest
import asp._asp_core as asp_core

def test_milestone_1_picard_integration():
    """Validates O(N log N) Chebyshev integration and FFI boundary."""
    N = 64
    k = np.arange(N + 1, dtype=float)
    sigma = np.cos(np.pi * k / N)
    
    F_vals = np.exp(sigma)
    c_F = asp_core.chebyshev_fit(F_vals)
    
    u_init = 1.0
    delta_s = 2.0 
    c_u = asp_core.chebyshev_integrate(c_F, u_init, delta_s)
    
    # Evaluate using Clenshaw
    sigma_test = np.linspace(-1, 1, 100)
    b2 = np.zeros_like(sigma_test)
    b1 = np.zeros_like(sigma_test)
    for j in range(len(c_u) - 1, 0, -1):
        b0 = c_u[j] + 2.0 * sigma_test * b1 - b2
        b2 = b1
        b1 = b0
    u_num = c_u[0] + sigma_test * b1 - b2
    
    u_exact = u_init + (np.exp(sigma_test) - np.exp(-1.0))
    max_error = np.max(np.abs(u_num - u_exact))
    
    assert max_error < 1e-13, f"Integration precision failure: {max_error}"

def test_milestone_2_borel_pade_extraction():
    """Validates blind singularity detection and Borel-Padé-Laplace resummation."""
    N_coeffs = 40
    raw_coeffs = [float((-1)**n * math.factorial(n)) for n in range(N_coeffs)]
    
    # 1. Singularity Extraction
    poles = asp_core.extract_singularities_py(raw_coeffs)
    found_singularity = any(abs(p - (-1.0)) < 1e-4 for p in poles)
    assert found_singularity, "Failed to detect Borel singularity at zeta = -1.0"
    
    # 2. Resummation
    z_val = 1.0
    exact_val = 0.596347362323194
    resummed_val = asp_core.borel_pade_laplace_py(raw_coeffs, z_val, 150)
    
    error = abs(resummed_val - exact_val)
    assert error < 1e-12, f"Resummation precision failure: {error}"

def test_milestone_5_uk_constraint():
    """Validates Udwadia-Kalaba Exact Constraint Projection (Thesis Paper 21)."""
    
    # Simple Pendulum (length L = 2.0, gravity g = 9.81)
    # State: x = 2.0, y = 0.0, vx = 0.0, vy = 5.0
    L = 2.0
    g = 9.81
    
    x, y = 2.0, 0.0
    vx, vy = 0.0, 5.0
    
    # Unconstrained dynamics (M = I)
    M = np.eye(2, dtype=np.float64)
    Q = np.array([0.0, -g], dtype=np.float64) # Gravity acts downwards
    
    # Constraint A * q_ddot = b
    # x*x_ddot + y*y_ddot = -vx^2 - vy^2
    A = np.array([[x, y]], dtype=np.float64)
    b = np.array([-vx**2 - vy**2], dtype=np.float64)
    
    # Compute UK Constraint Force via Rust FFI
    Fc = asp_core.compute_uk_force(M, A, b, Q, 1e-12)
    
    # Total acceleration q_ddot = Q + Fc
    q_ddot = Q + Fc
    
    # Verify constraint is exactly satisfied: A * q_ddot == b
    constraint_residual = np.abs(np.dot(A, q_ddot) - b)[0]
    
    assert constraint_residual < 1e-13, f"UK projection failed. Residual: {constraint_residual}"
    
    # Verify the fast-path (M=I) yields identical results
    Fc_id = asp_core.compute_uk_force_identity(A, b, Q, 1e-12)
    identity_residual = np.max(np.abs(Fc - Fc_id))
    
    assert identity_residual < 1e-13, "M=I fast path diverged from general mass matrix path."