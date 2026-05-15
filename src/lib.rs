// src/lib.rs

use pyo3::prelude::*;

pub mod math;
pub mod physics;
pub mod solvers;
pub mod borel_pade;
pub mod resurgent_ps; // <--- NEW MODULE

use physics::lifting::LiftDiagnosisPy;
use solvers::picard::{PicardConfigPy, PicardSolverPy, PropagationResultPy, KsCr3bpResultPy};

#[pyfunction]
fn borel_pade_laplace_py(coeffs: Vec<f64>, z: f64, n_gl: usize) -> f64 {
    borel_pade::borel_pade_laplace(&coeffs, z, n_gl)
}

#[pyfunction]
fn extract_singularities_py(coeffs: Vec<f64>) -> Vec<f64> {
    borel_pade::extract_singularities(&coeffs)
}

/// Python binding for the Resurgent Pseudoinverse
#[pyfunction]
#[pyo3(name = "resurgent_pseudoinverse")]
fn resurgent_pseudoinverse_py<'py>(
    py: Python<'py>,
    a_func: PyObject,
    eps_min: f64,
    eps_max: f64,
    n_cheb: usize,
) -> PyResult<(&'py numpy::PyArray2<f64>, &'py numpy::PyArray2<f64>)> {
    
    // Wrap the Python callable into a Rust closure
    let func = |eps: f64| -> nalgebra::DMatrix<f64> {
        Python::with_gil(|py2| {
            let args = (eps,);
            let result = a_func.call1(py2, args).unwrap();
            let pyarray: &numpy::PyArray2<f64> = result.extract(py2).unwrap();
            
            // Safe copy from numpy to nalgebra DMatrix
            let arr = pyarray.as_array();
            let mut mat = nalgebra::DMatrix::zeros(arr.nrows(), arr.ncols());
            for i in 0..arr.nrows() {
                for j in 0..arr.ncols() {
                    mat[(i, j)] = arr[[i, j]];
                }
            }
            mat
        })
    };

    let res = resurgent_ps::resurgent_pseudoinverse(func, eps_min, eps_max, n_cheb);

    // Safe copy from nalgebra DMatrix back to numpy
    let r_minus_1_py = numpy::PyArray2::from_shape_fn(py, (res.r_minus_1.nrows(), res.r_minus_1.ncols()), |(i, j)| res.r_minus_1[(i, j)]);
    let r_0_py = numpy::PyArray2::from_shape_fn(py, (res.r_0.nrows(), res.r_0.ncols()), |(i, j)| res.r_0[(i, j)]);

    Ok((r_minus_1_py, r_0_py))
}

#[pymodule]
fn asp_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PicardConfigPy>()?;
    m.add_class::<PicardSolverPy>()?;
    m.add_class::<PropagationResultPy>()?;
    m.add_class::<LiftDiagnosisPy>()?;
    m.add_class::<KsCr3bpResultPy>()?;

    m.add_function(wrap_pyfunction!(solvers::picard::propagate_cr3bp_fast, m)?)?;
    m.add_function(wrap_pyfunction!(physics::ks_cr3bp::cr3bp_jacobi, m)?)?;
    m.add_function(wrap_pyfunction!(physics::lifting::diagnose_system, m)?)?;
    m.add_function(wrap_pyfunction!(physics::ks_cr3bp::ks_map_py, m)?)?;
    m.add_function(wrap_pyfunction!(physics::ks_cr3bp::ks_inverse_py, m)?)?;
    m.add_function(wrap_pyfunction!(physics::lifting::lc_map_py, m)?)?;
    m.add_function(wrap_pyfunction!(physics::lifting::lc_inverse_py, m)?)?;
    
    m.add_function(wrap_pyfunction!(borel_pade_laplace_py, m)?)?;
    m.add_function(wrap_pyfunction!(extract_singularities_py, m)?)?;
    
    // NEW BINDING
    m.add_function(wrap_pyfunction!(resurgent_pseudoinverse_py, m)?)?;

    m.add("__version__", "0.1.0")?;
    
    Ok(())
}