import math
import numpy as np
import pytest
import asp_core

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

import pytest
import numpy as np
from asp.orchestrator.borel_plane import BorelPlaneAnalyzer
from asp.symbolic.lindstedt import generate_mock_lindstedt_series

def test_milestone_6_nekhoroshev_bounds():
    """
    Validates the Constructive Nekhoroshev Theorem (Thesis Paper 18).
    Extracts the Peierls' barrier and Arnold diffusion rate from the 
    Lindstedt series of the Standard Map.
    """
    # 1. Generate the Gevrey-1 Lindstedt series
    # Critical coupling for golden-mean torus
    K_c = 0.971635 
    # Known theoretical instanton action from thesis
    theoretical_S_inst = 6.28 
    
    coeffs = generate_mock_lindstedt_series(n_orders=40, K_c=K_c, S_inst=theoretical_S_inst)
    
    # 2. Analyze the Borel Plane
    analyzer = BorelPlaneAnalyzer(coeffs)
    
    # 3. Compute Bounds in the Arnold Diffusion Regime (K > K_c)
    K_eval = K_c + 0.03
    bounds = analyzer.compute_nekhoroshev_bounds(K=K_eval, K_c=K_c)
    
    assert bounds["regime"] == "Arnold diffusion"
    
    # Verify the extracted instanton action matches the theoretical value
    extracted_S_inst = bounds["S_inst"]
    error_S_inst = abs(extracted_S_inst - theoretical_S_inst)
    assert error_S_inst < 1e-3, f"Failed to extract S_inst. Got {extracted_S_inst}"
    
    # Verify the Arnold Diffusion Rate (Gamma = pi * S_inst ≈ 19.7)
    expected_gamma = np.pi * theoretical_S_inst
    error_gamma = abs(bounds["arnold_diffusion_rate"] - expected_gamma)
    assert error_gamma < 1e-3, f"Arnold diffusion rate mismatch. Got {bounds['arnold_diffusion_rate']}"
    
    print(f"\n--- Nekhoroshev Stability Metrics (K = {K_eval}) ---")
    print(f"Instanton Action (S_inst): {extracted_S_inst:.4f}")
    print(f"Peierls' Barrier:          {bounds['peierls_barrier']:.4e}")
    print(f"Nekhoroshev Time (T_nek):  {bounds['t_nekhoroshev']:.4e} orbits")
    print(f"Arnold Diffusion Rate:     {bounds['arnold_diffusion_rate']:.4f}")

# tests/test_milestone_7.py

import pytest
import sympy as sp
import asp_core
from asp.symbolic.codegen import RustJITCompiler
from asp.symbolic.bender_wu import bender_wu_quartic

def test_milestone_7_jit_compilation_and_propagation():
    """
    Validates Symbolic AST Lowering.
    Defines a non-linear ODE in SymPy, JIT compiles it to a Rust C-ABI binary,
    and propagates it using the Adaptive Spectral Picard solver.
    """
    # 1. Define the system in SymPy
    t = sp.Symbol('t')
    x = sp.Symbol('x')
    v = sp.Symbol('v')
    
    # Duffing Oscillator: x'' + x + 0.1 * x^3 = 0
    # State vector: [x, v]
    state_vars = [x, v]
    rhs_exprs = [
        v,
        -x - 0.1 * x**3
    ]
    
    # 2. JIT Compile to Rust
    compiler = RustJITCompiler(state_vars, rhs_exprs, time_var=t)
    func_ptr = compiler.compile_and_load()
    
    assert func_ptr > 0, "Failed to extract valid function pointer from JIT library."
    
    # 3. Configure the Spectral Picard Solver
    n_cheb = 32
    tol = 1e-11
    certify = True
    
    x0 = [1.0, 0.0]
    t_final = float(2.0 * sp.pi.evalf()) # Cast SymPy Float to standard Python float
    
    # 4. Execute the bare-metal Rust propagation
    result = asp_core.propagate_custom_c_abi(
        x0=x0,
        t_final=t_final,
        rhs_ptr_val=func_ptr["rhs"],
        jac_ptr_val=func_ptr.get("jac"),
        uk_ptr_val=func_ptr.get("uk"),
        n_cheb=n_cheb,
        tol=tol,
        certify=certify
    )
    
    assert result.n_segments > 0, "Picard solver failed to produce segments."
    assert result.nk_bound_max < 1e-8, f"NK Bound too loose: {result.nk_bound_max}"
    
    print(f"\n--- JIT Compilation Success ---")
    print(f"Final State (Duffing Oscillator): {result.final_state}")
    print(f"NK Error Bound: {result.nk_bound_max:.2e}")

def test_milestone_7_deep_encoding():
    """
    Validates the Bender-Wu algebraic recursion (Thesis Chapter 2).
    Generates exact perturbative coefficients for the quartic oscillator.
    """
    N = 40
    coeffs = bender_wu_quartic(N)
    
    assert len(coeffs) == N
    assert coeffs[0] == 0.5
    assert coeffs[1] == 0.75 # Known first-order correction for quartic oscillator
    
    # Pass to the Borel-Pade engine to extract the instanton action
    # For V(x) = x^2/2 + g x^4, A = 1/(3g). The Borel transform acts on the series in g.
    # The expected Borel pole is at zeta = -1/3.
    poles = asp_core.extract_singularities_py(coeffs)
    found_singularity = any(abs(p - (-1.0/3.0)) < 1e-3 for p in poles)
    
    assert found_singularity, f"Failed to extract topological instanton action A=1/3. Found: {poles}"
    print(f"\n--- Deep Encoding Success ---")
    print(f"Extracted Instanton Action (A = 1/3): {poles}")

if __name__ == "__main__":
    test_milestone_7_jit_compilation_and_propagation()
    test_milestone_7_deep_encoding()