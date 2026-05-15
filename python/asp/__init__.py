# python/asp/__init__.py

import asp._asp_core as _core

# Robust FFI Class Resolution
PicardConfig = getattr(_core, "PicardConfig", getattr(_core, "PicardConfigPy", None))
PicardSolver = getattr(_core, "PicardSolver", getattr(_core, "PicardSolverPy", None))
PropagationResult = getattr(_core, "PropagationResult", getattr(_core, "PropagationResultPy", None))
LiftDiagnosis = getattr(_core, "LiftDiagnosis", getattr(_core, "LiftDiagnosisPy", None))
KsCr3bpResult = getattr(_core, "KsCr3bpResult", getattr(_core, "KsCr3bpResultPy", None))

# FFI Function Mapping
propagate_cr3bp_fast = _core.propagate_cr3bp_fast
cr3bp_jacobi = _core.cr3bp_jacobi
estimate_s_from_t = _core.estimate_s_from_t
propagate_ks_cr3bp = _core.propagate_ks_cr3bp
diagnose_system = _core.diagnose_system
ks_map_py = _core.ks_map_py
ks_inverse_py = _core.ks_inverse_py
lc_map_py = _core.lc_map_py
lc_inverse_py = _core.lc_inverse_py
borel_pade_laplace_py = _core.borel_pade_laplace_py
extract_singularities_py = _core.extract_singularities_py
resurgent_pseudoinverse = _core.resurgent_pseudoinverse
compute_uk_force = _core.compute_uk_force
compute_uk_force_identity = _core.compute_uk_force_identity
propagate_custom_c_abi = _core.propagate_custom_c_abi
evaluate_clenshaw = _core.evaluate_clenshaw
median_resummation_py = _core.median_resummation_py
compute_apc_collocation = _core.compute_apc_collocation

# Python Orchestrator & Symbolic Modules
from .symbolic.codegen import RustJITCompiler
from .symbolic.bender_wu import bender_wu_quartic
from .symbolic.lindstedt import generate_mock_lindstedt_series
from .orchestrator.borel_plane import BorelPlaneAnalyzer
from .orchestrator.apc import ArbitraryPolynomialChaos

__all__ = [
    "PicardConfig",
    "PicardSolver",
    "PropagationResult",
    "LiftDiagnosis",
    "KsCr3bpResult",
    "propagate_cr3bp_fast",
    "cr3bp_jacobi",
    "estimate_s_from_t",
    "propagate_ks_cr3bp",
    "diagnose_system",
    "ks_map_py",
    "ks_inverse_py",
    "lc_map_py",
    "lc_inverse_py",
    "borel_pade_laplace_py",
    "extract_singularities_py",
    "resurgent_pseudoinverse",
    "compute_uk_force",
    "compute_uk_force_identity",
    "propagate_custom_c_abi",
    "evaluate_clenshaw",
    "median_resummation_py",
    "compute_apc_collocation",
    "BorelPlaneAnalyzer",
    "ArbitraryPolynomialChaos",
    "generate_mock_lindstedt_series",
    "RustJITCompiler", 
    "bender_wu_quartic"
]