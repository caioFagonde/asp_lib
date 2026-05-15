# python/asp/orchestrator/solver.py

import sympy as sp
import asp._asp_core as asp_core
from asp.symbolic.codegen import RustJITCompiler

class ASPSolver:
    """
    High-level orchestrator for the ASP-UK-Res Engine.
    Handles dynamic JIT compilation, Udwadia-Kalaba exact constraint enforcement, 
    and Newton-Kantorovich certified trajectory propagation.
    """
    def __init__(self, n_cheb: int = 32, tol: float = 1e-12, certify: bool = True):
        self.n_cheb = n_cheb
        self.tol = tol
        self.certify = certify
        self._compiler = None
        self.ptrs = None

    def compile_system(self, state_vars: list[sp.Symbol], rhs_exprs: list[sp.Expr], time_var: sp.Symbol, uk_exprs: list[sp.Expr] = None):
        """
        Ingests the symbolic math, generates the analytical Jacobian, and JIT compiles 
        the system to a highly-optimized Rust binary.
        """
        self._compiler = RustJITCompiler(state_vars, rhs_exprs, time_var, uk_exprs)
        self.ptrs = self._compiler.compile_and_load()
        print("ASP Solver: System successfully compiled and loaded into memory.")

    def propagate(self, x0: list[float], t_final: float):
        """
        Executes the ASP-UK-Res solver on the compiled system.
        """
        if self._compiler is None or self.ptrs is None:
            raise RuntimeError("System must be compiled via `compile_system` before propagation.")
            
        print(f"ASP Solver: Propagating {len(x0)}D system to t={t_final}...")
        
        # [SURGICAL FIX: GAP 3] Map the dictionary values to the explicit FFI arguments
        result = asp_core.propagate_custom_c_abi(
            x0=x0,
            t_final=t_final,
            rhs_ptr_val=self.ptrs["rhs"],
            jac_ptr_val=self.ptrs.get("jac"),
            uk_ptr_val=self.ptrs.get("uk"),
            n_cheb=self.n_cheb,
            tol=self.tol,
            certify=self.certify
        )
        
        print(f"ASP Solver: Propagation complete. Segments used: {result.n_segments}")
        if self.certify:
            print(f"ASP Solver: NK Certification Bound: {result.nk_bound_max:.2e}")
            if result.nk_bound_max >= 1.0:
                print("WARNING: NK Bound exceeds 1.0. Operator contraction not strictly certified.")
                
        return result