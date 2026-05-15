// src/lib.rs

use pyo3::prelude::*;

pub mod math;
pub mod physics;
pub mod solvers;

use physics::lifting::{LiftDiagnosisPy};
use solvers::picard::{PicardConfigPy, PicardSolverPy, PropagationResultPy, KsCr3bpResultPy};

#[pymodule]
fn asp_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Expose solver configurations and results
    m.add_class::<PicardConfigPy>()?;
    m.add_class::<PicardSolverPy>()?;
    m.add_class::<PropagationResultPy>()?;
    m.add_class::<LiftDiagnosisPy>()?;
    m.add_class::<KsCr3bpResultPy>()?;

    // Expose fast physics bindings
    m.add_function(wrap_pyfunction!(solvers::picard::propagate_cr3bp_fast, m)?)?;
    m.add_function(wrap_pyfunction!(physics::ks_cr3bp::cr3bp_jacobi, m)?)?;
    m.add_function(wrap_pyfunction!(physics::lifting::diagnose_system, m)?)?;
    m.add_function(wrap_pyfunction!(physics::ks_cr3bp::ks_map_py, m)?)?;
    m.add_function(wrap_pyfunction!(physics::ks_cr3bp::ks_inverse_py, m)?)?;
    m.add_function(wrap_pyfunction!(physics::lifting::lc_map_py, m)?)?;
    m.add_function(wrap_pyfunction!(physics::lifting::lc_inverse_py, m)?)?;

    m.add("__version__", "0.1.0")?;
    
    Ok(())
}