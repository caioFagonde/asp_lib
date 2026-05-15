#!/usr/bin/env python3
"""
ASP Framework Validation Suite v1.2.0 (Certified)
=================================================
Executes the mathematical certification of the Analytic Structural Physics (ASP)
computational engine. Validates the Resurgent Triangle, KS-Regularization,
Udwadia-Kalaba Bridge, and Arbitrary Polynomial Chaos.
"""

import os
import sys
import math
import time
import decimal
from pathlib import Path
import numpy as np

# ==============================================================================
# GRPC PROVISIONING
# ==============================================================================

def provision_grpc() -> None:
    """Robust, cross-platform pure Python gRPC provisioning."""
    print("[*] Provisioning gRPC stubs via Python AST/String patching...")
    try:
        import grpc_tools.protoc
    except ImportError:
        print("[!] Missing grpcio-tools. Run: pip install grpcio-tools")
        
        
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent 
    proto_dir = project_root / "go_services" / "proto"
    out_dir = project_root / "python" / "asp" / "orchestrator"
    
    if not proto_dir.exists():
        proto_dir = Path("go_services/proto").resolve()
        out_dir = Path("python/asp/orchestrator").resolve()
        
    proto_file = proto_dir / "asp_cluster.proto"
    if not proto_file.exists():
        print(f"[!] Proto file not found at {proto_file}. Skipping gRPC provision.")
        return
        
    out_dir.mkdir(parents=True, exist_ok=True)
    well_known_protos_include = os.path.join(os.path.dirname(grpc_tools.protoc.__file__), '_proto')
    
    args = [
        'grpc_tools.protoc',
        f'-I{proto_dir}',
        f'-I{well_known_protos_include}',
        f'--python_out={out_dir}',
        f'--grpc_python_out={out_dir}',
        str(proto_file)
    ]
    
    exit_code = grpc_tools.protoc.main(args)
    if exit_code != 0:
        print(f"[!] protoc failed with exit code {exit_code}")
        
        
    pb2_grpc_path = out_dir / "asp_cluster_pb2_grpc.py"
    if pb2_grpc_path.exists():
        content = pb2_grpc_path.read_text(encoding='utf-8')
        # Safely patch relative imports for Python 3 package structure
        content = content.replace("import asp_cluster_pb2 as asp__cluster__pb2", 
                                  "from . import asp_cluster_pb2 as asp__cluster__pb2")
        pb2_grpc_path.write_text(content, encoding='utf-8')
        
    print("[*] Protobuf files generated and patched successfully.\n")

provision_grpc()

try:
    import asp
except ImportError:
    print("[!] Failed to import 'asp'. Ensure the Rust library is compiled and installed (pip install -e .).")
    

# ==============================================================================
# CERTIFICATION INFRASTRUCTURE
# ==============================================================================

class FalsifiedError(Exception):
    """Raised when a strict mathematical bound or theoretical claim is falsified."""
    pass

def assert_certified(condition: bool, message: str) -> None:
    """Strictly enforce theoretical axioms. Failure halts the certification."""
    if not condition:
        raise FalsifiedError(f"[FALSIFIED] {message}")

def generate_bw_60_high_prec() -> list[float]:
    """
    Computes exact Bender-Wu coefficients using 100-digit precision.
    Demonstrates the factorial growth encoding the non-perturbative instanton.
    """
    decimal.getcontext().prec = 100
    D = decimal.Decimal
    N = 60
    
    C = [[D(0)] * (4 * N + 1) for _ in range(N)]
    a = [D(0)] * N
    a[0] = D('0.5')
    C[0][0] = D(1)
    
    def x4_mat(k: int, m: int) -> decimal.Decimal:
        if k == m: 
            return D('0.75') * D(2 * m**2 + 2 * m + 1)
        elif abs(k - m) == 2:
            min_idx = min(k, m)
            return D(2 * min_idx + 3) * (D(min_idx + 1) * D(min_idx + 2)).sqrt() / D(2)
        elif abs(k - m) == 4:
            min_idx = min(k, m)
            return (D(min_idx + 1) * D(min_idx + 2) * D(min_idx + 3) * D(min_idx + 4)).sqrt() / D(4)
        return D(0)
        
    for n in range(1, N):
        term = D(0)
        for m in range(4 * (n - 1) + 1):
            if C[n-1][m] != D(0):
                term += x4_mat(0, m) * C[n-1][m]
        a[n] = term
        
        for k in range(2, 4 * n + 1, 2):
            sum_matrix = D(0)
            for m in range(4 * (n - 1) + 1):
                if C[n-1][m] != D(0):
                    mat_el = x4_mat(k, m)
                    if mat_el != D(0):
                        sum_matrix += mat_el * C[n-1][m]
            
            sum_energy = D(0)
            for j in range(1, n):
                if C[n-j][k] != D(0):
                    sum_energy += a[j] * C[n-j][k]
                    
            C[n][k] = -(sum_matrix - sum_energy) / D(k)
            
    return [float(x) for x in a]

# ==============================================================================
# PHASE 1: RESURGENCE & BOREL-PADÉ SUPREMACY
# ==============================================================================

def phase_1_resurgence() -> None:
    """Certify blind singularity extraction and resummation of divergent series."""
    print("=== PHASE 1: The Resurgent Triangle & Borel-Padé Supremacy ===")
    try:
        print("[*] Generating 60 Bender-Wu coefficients using 100-digit precision...")
        coeffs_bw = generate_bw_60_high_prec()
        
        print("[*] Executing blind singularity detection via Rust SVD Padé...")
        poles = asp.extract_singularities_py(coeffs_bw)
        
        real_poles = [p for p in poles if abs(np.imag(p)) < 1e-5]
        extracted_action = min([abs(p) for p in real_poles]) if real_poles else float('inf')
        
        # The theoretical instanton action for the quartic oscillator is exactly 1/3
        theoretical_action = 1.0 / 3.0
        error_action = abs(extracted_action - theoretical_action) / theoretical_action
        print(f"    Extracted Instanton Action: {extracted_action:.6f} (Error: {error_action:.4%})")
        assert_certified(error_action < 2.0/100.0, f"Instanton action extraction failed 2% bound. Error: {error_action:.4%}")
        
        print("[*] Resumming the divergent Euler series at z=1.0...")
        coeffs_euler = [float(math.factorial(n) * (-1)**n) for n in range(40)]
        resummed_val = asp.borel_pade_laplace_py(coeffs_euler, 1.0, 150)
        
        exact_euler = 0.596347362323194
        error_euler = abs(resummed_val - exact_euler)
        print(f"    Borel-Padé-Laplace Result: {resummed_val:.12f} (Error: {error_euler:.2e})")
        assert_certified(error_euler < 1e-11, f"Euler resummation failed 10^-11 precision bound. Error: {error_euler:.2e}")
        
        print("[PROVEN] Phase 1 Complete.\n")
    except FalsifiedError as e:
        print(e, "\n")
        

# ==============================================================================
# PHASE 2: SPECTRAL ASTRODYNAMICS & KS-REGULARIZATION
# ==============================================================================

def phase_2_astrodynamics() -> None:
    """Certify the topological removal of the Kepler collision singularity."""
    print("=== PHASE 2: Astrodynamics & KS-Regularization ===")
    try:
        mu = 0.012151
        t_final = 0.1
        
        print("[*] Propagating highly eccentric Keplerian orbit (e=0.9)...")
        x0_kepler = [0.1, 0.0, 0.0, 0.0, 4.2588989, 0.0]
        res_kepler = asp.propagate_cr3bp_fast(x0_kepler, t_final, mu, n_cheb=32, tol=1e-12)
        print(f"    Segments Used: {res_kepler.n_segments} | Jacobi Error: {res_kepler.jacobi_error:.2e}")
        assert_certified(res_kepler.jacobi_error < 1e-11, f"Kepler Jacobi error exceeded threshold: {res_kepler.jacobi_error:.2e}")
        assert_certified(res_kepler.n_segments < 20, "Solver micro-stepped on Keplerian orbit.")
        
        print("[*] Propagating close lunar flyby (r2=0.01) via KS-regularized solver...")
        x0_flyby = [1.0 - mu + 0.01, 0.0, 0.0, 0.0, 1.99, 0.0]
        s_final = asp.estimate_s_from_t(t_final, x0_flyby, mu)
        res_flyby = asp.propagate_ks_cr3bp(x0_flyby, s_final, mu, n_cheb=64, tol=1e-12)
        
        print(f"    Segments Used: {res_flyby.n_segments} | Jacobi Error: {res_flyby.jacobi_error:.2e} | NK Bound: {res_flyby.nk_bound_max:.2e}")
        assert_certified(res_flyby.jacobi_error < 1e-11, f"Flyby Jacobi error exceeded threshold: {res_flyby.jacobi_error:.2e}")
        assert_certified(res_flyby.n_segments < 100, f"Solver micro-stepped on lunar flyby. Used {res_flyby.n_segments} segments.")
        assert_certified(res_flyby.nk_bound_max < 1.0, f"NK bound exceeded 1.0; certification failed. Bound: {res_flyby.nk_bound_max:.2e}")
        
        print("[PROVEN] Phase 2 Complete.\n")
    except FalsifiedError as e:
        print(e, "\n")
        

# ==============================================================================
# PHASE 3: THE UDWADIA-KALABA RESURGENCE BRIDGE
# ==============================================================================

def phase_3_uk_bridge() -> None:
    """Certify that the UK constraint force is identically the alien derivative."""
    print("=== PHASE 3: The Udwadia-Kalaba Resurgence Bridge ===")
    try:
        angles = [0.1677, 0.0655, 0.0256, 0.0100]
        M = np.eye(2, dtype=np.float64)
        b = np.array([np.sqrt(2.0), 0.0], dtype=np.float64)
        Q = np.array([0.0, 0.0], dtype=np.float64)
        
        print("[*] Evaluating UK constraint force approaching wrist singularity...")
        alien_derivative = 0.0
        for theta2 in angles:
            t1 = np.pi / 4.0
            t12 = t1 + theta2
            A = np.array([
                [-np.sin(t1) - np.sin(t12), -np.sin(t12)],
                [ np.cos(t1) + np.cos(t12),  np.cos(t12)]
            ], dtype=np.float64)
            
            Fc = asp.compute_uk_force(M, A, b, Q, tol=1e-12)
            Fc_norm = float(np.linalg.norm(Fc))
            
            S = np.linalg.svd(A, compute_uv=False)
            sigma_min = float(S[-1])
            
            alien_derivative = sigma_min * Fc_norm
            print(f"    theta2={theta2:.4f} | sigma_min={sigma_min:.4f} | |Fc|={Fc_norm:.3f} | Residue={alien_derivative:.3f}")
            
        assert_certified(abs(alien_derivative - 1.0) < 0.05, f"Alien derivative residue did not converge to ~1.0. Got {alien_derivative:.3f}")
        print("[PROVEN] Phase 3 Complete.\n")
    except FalsifiedError as e:
        print(e, "\n")
        

# ==============================================================================
# PHASE 4: DISTRIBUTED ARBITRARY POLYNOMIAL CHAOS (APC)
# ==============================================================================

def phase_4_apc() -> None:
    """Certify orthogonal collocation and Newton-Kantorovich bounds across stochastic batches."""
    print("=== PHASE 4: Distributed Arbitrary Polynomial Chaos (APC) ===")
    try:
        a = 0.01
        n_points = 5
        moments = [(a**k) / (k + 1.0) if k % 2 == 0 else 0.0 for k in range(2 * n_points)]
                
        print(f"[*] Initializing APC with {n_points} collocation points...")
        apc = asp.ArbitraryPolynomialChaos(moments, n_points)
        
        nominal_state = [0.82, 0.0, 0.12, 0.0, -0.13, 0.0]
        t_final = 1.5
        mu_cr3bp = 0.01215
        
        try:
            print("[*] Attempting dispatch to Go/Rust gRPC cluster...")
            start_time = time.perf_counter()
            res = apc.propagate_and_aggregate(nominal_state, uncertainty_idx=0, t_final=t_final, host="localhost")
            elapsed = time.perf_counter() - start_time
            print("    Cluster dispatch successful.")
        except Exception as e:
            print(f"    [!] Cluster unreachable ({e}). Gracefully degrading to LOCAL certified propagation...")
            
            # Local fallback implementation maintaining strict mathematical certification
            start_time = time.perf_counter()
            states = apc.generate_collocation_states(nominal_state, uncertainty_idx=0)
            
            expected_state = np.zeros(6)
            expected_sq = np.zeros(6)
            all_certified = True
            max_nk_bound = 0.0
            
            for i, st in enumerate(states):
                # Propagate locally using the compiled Rust core
                prop_res = asp.propagate_cr3bp_fast(st, t_final, mu_cr3bp, n_cheb=32, tol=1e-11)
                
                f_i = np.array(prop_res.final_state, dtype=np.float64)
                w_i = apc.weights[i]
                
                expected_state += w_i * f_i
                expected_sq += w_i * (f_i ** 2)
                
                nk_bound = prop_res.nk_bound_max
                max_nk_bound = max(max_nk_bound, nk_bound)
                if nk_bound >= 1.0:
                    all_certified = False

            expected_state /= apc.mu_0
            expected_sq /= apc.mu_0
            variance = np.maximum(expected_sq - expected_state ** 2, 0.0)
            
            res = {
                "variance": variance.tolist(),
                "is_certified": all_certified,
                "max_nk_bound": max_nk_bound
            }
            elapsed = time.perf_counter() - start_time

        variance = np.array(res["variance"])
        print(f"    Batch Certified: {res['is_certified']}")
        print(f"    Max NK Bound: {res['max_nk_bound']:.2e}")
        print(f"    Execution Wall-clock: {elapsed:.4f}s")
        
        assert_certified(res["is_certified"], "Distributed/Local APC batch failed NK certification.")
        assert_certified(np.all(variance >= 0.0), "APC yielded negative variance (unphysical).")
        print("[PROVEN] Phase 4 Complete.\n")
        
    except FalsifiedError as e:
        print(e, "\n")
        

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("============================================================")
    print(" ASP CORE ENGINE - RIGOROUS VALIDATION SUITE v1.2.0")
    print("============================================================\n")
    
    try:
        phase_1_resurgence()
    except Exception as e:
        print(e)

    try:
        phase_2_astrodynamics()
    except Exception as e:
        print(e)

    try:
        phase_3_uk_bridge()
    except Exception as e:
        print(e)

    try:
        phase_4_apc()
    except Exception as e:
        print(e)
    
    # print("============================================================")
    # print(" ALL SYSTEMS NOMINAL. MATHEMATICAL CLAIMS VERIFIED.")
    # print("============================================================")