# python/asp/symbolic/codegen.py

import os
import sys
import ctypes
import tempfile
import subprocess
import hashlib
import sympy as sp
from sympy.printing.rust import RustCodePrinter

class RustJITCompiler:
    """
    Dynamically compiles SymPy ODE systems, Jacobians, and UK constraints 
    into bare-metal Rust C-ABI shared libraries with SHA-256 caching.
    """
    def __init__(self, state_vars: list[sp.Symbol], rhs_exprs: list[sp.Expr], time_var: sp.Symbol, uk_exprs: list[sp.Expr] = None):
        self.state_vars = state_vars
        self.rhs_exprs = rhs_exprs
        self.time_var = time_var
        self.uk_exprs = uk_exprs
        self.ndim = len(state_vars)
        
        # Analytically derive the Jacobian for exact NK Certification
        print("ASP JIT: Analytically computing Jacobian matrix for NK certification...")
        rhs_matrix = sp.Matrix(self.rhs_exprs)
        self.jac_matrix = rhs_matrix.jacobian(self.state_vars)
        
        self._lib = None
        self._rhs_ptr = None
        self._jac_ptr = None
        self._uk_ptr = None
        
        self.cache_dir = os.path.join(tempfile.gettempdir(), "asp_jit_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _hash_system(self) -> str:
        """Creates a deterministic hash of the mathematical system to enable caching."""
        system_str = str(self.rhs_exprs) + str(self.jac_matrix) + str(self.uk_exprs)
        return hashlib.sha256(system_str.encode('utf-8')).hexdigest()[:16]

    def _generate_c_abi_function(self, func_name: str, exprs: list[sp.Expr], out_dim: int) -> str:
        """Generates a generic C-ABI compliant Rust function over a flattened array."""
        lines = [
            "#[no_mangle]",
            f"pub extern \"C\" fn {func_name}(",
            "    x_ptr: *const f64,",
            "    t_ptr: *const f64,",
            "    out_ptr: *mut f64,",
            "    ndim: usize,",
            "    npts: usize,",
            ") {",
            "    let x = unsafe { std::slice::from_raw_parts(x_ptr, ndim * npts) };",
            "    let t = unsafe { std::slice::from_raw_parts(t_ptr, npts) };",
            f"    let out = unsafe {{ std::slice::from_raw_parts_mut(out_ptr, {out_dim} * npts) }};",
            "",
            "    for j in 0..npts {"
        ]

        # Map C-contiguous flat memory to SymPy variables
        for i, var in enumerate(self.state_vars):
            lines.append(f"        let {var.name} = x[{i} * npts + j];")
        
        lines.append(f"        let {self.time_var.name} = t[j];")
        lines.append("")

        printer = RustCodePrinter()
        # Assign computed expressions
        for i, expr in enumerate(exprs):
            rust_expr = printer.doprint(expr)
            lines.append(f"        out[{i} * npts + j] = {rust_expr};")

        lines.append("    }")
        lines.append("}")
        return "\n".join(lines)

    def generate_rust_code(self) -> str:
        code_blocks = []
        
        # 1. RHS Vector Field
        code_blocks.append(self._generate_c_abi_function("evaluate_rhs", self.rhs_exprs, self.ndim))
        
        # 2. Jacobian Matrix (Flattened to 1D)
        jac_flat = [self.jac_matrix[i, j] for i in range(self.ndim) for j in range(self.ndim)]
        code_blocks.append(self._generate_c_abi_function("evaluate_jac", jac_flat, self.ndim * self.ndim))
        
        # 3. UK Constraint Projection (Optional)
        if self.uk_exprs is not None:
            code_blocks.append(self._generate_c_abi_function("evaluate_uk", self.uk_exprs, self.ndim))
            
        return "\n\n".join(code_blocks)

    def compile_and_load(self) -> dict:
        """
        Compiles to a shared library (utilizing cache if available) and returns 
        the memory addresses of the function pointers.
        """
        if self._lib is not None:
            return {"rhs": self._rhs_ptr, "jac": self._jac_ptr, "uk": self._uk_ptr}

        sys_hash = self._hash_system()
        ext = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
        lib_file = os.path.join(self.cache_dir, f"asp_sys_{sys_hash}{ext}")

        if not os.path.exists(lib_file):
            print(f"ASP JIT: Cache miss. Compiling system {sys_hash} via rustc...")
            code = self.generate_rust_code()
            rs_file = os.path.join(self.cache_dir, f"asp_sys_{sys_hash}.rs")
            
            with open(rs_file, "w") as f:
                f.write(code)

            compile_cmd = [
                "rustc",
                "--crate-type", "cdylib",
                "-C", "opt-level=3", # Maximum performance optimization
                "-C", "target-cpu=native", # Utilize AVX/SIMD instructions of host
                "-o", lib_file,
                rs_file
            ]
            
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Rust JIT Compilation failed:\n{result.stderr}")
        else:
            print(f"ASP JIT: Cache hit. Loading system {sys_hash}...")

        self._lib = ctypes.CDLL(lib_file)
        
        self._rhs_ptr = ctypes.cast(self._lib.evaluate_rhs, ctypes.c_void_p).value
        self._jac_ptr = ctypes.cast(self._lib.evaluate_jac, ctypes.c_void_p).value
        
        if self.uk_exprs is not None:
            self._uk_ptr = ctypes.cast(self._lib.evaluate_uk, ctypes.c_void_p).value

        return {"rhs": self._rhs_ptr, "jac": self._jac_ptr, "uk": self._uk_ptr}