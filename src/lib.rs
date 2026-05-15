// src/lib.rs
//
// PyO3 FFI layer for the ASP Computational Engine.

#![allow(non_local_definitions)]

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use ndarray::Array1;
use numpy::ToPyArray;

pub mod math;
pub mod physics;
pub mod solvers;
pub mod resurgent_ps;
pub mod picard_spec; 

use math::borel_pade;
use solvers::picard::{PicardConfig, build_trajectory_result};

// ---------------------------------------------------------------------------
// PyO3 structs
// ---------------------------------------------------------------------------

#[pyclass]
pub struct PropagationResultPy {
    #[pyo3(get)] pub n_segments:      usize,
    #[pyo3(get)] pub final_state:     Vec<f64>,
    #[pyo3(get)] pub nk_bound_max:    f64,
    #[pyo3(get)] pub jacobi_error:    f64,
    #[pyo3(get)] pub last_seg_coeffs: Vec<f64>,
    #[pyo3(get)] pub coeff_rows:      usize,
    #[pyo3(get)] pub coeff_cols:      usize,
}

#[pyclass]
pub struct PicardConfigPy {
    #[pyo3(get, set)] pub n_cheb:       usize,
    #[pyo3(get, set)] pub tol:          f64,
    #[pyo3(get, set)] pub max_iter:     usize,
    #[pyo3(get, set)] pub ds_initial:   f64,
    #[pyo3(get, set)] pub certify:      bool,
    #[pyo3(get, set)] pub max_segments: usize,
}

#[pymethods]
impl PicardConfigPy {
    #[new]
    fn new() -> Self {
        let d = PicardConfig::default();
        PicardConfigPy {
            n_cheb: d.n_cheb,
            tol: d.tol,
            max_iter: d.max_iter,
            ds_initial: d.ds_initial,
            certify: d.certify,
            max_segments: d.max_segments,
        }
    }
}

#[pyclass]
pub struct PicardSolverPy;

#[pyclass]
pub struct LiftDiagnosisPy {
    #[pyo3(get)] pub bernstein_rho:   f64,
    #[pyo3(get)] pub apply_fibration: bool,
    #[pyo3(get)] pub suggested_ds:    f64,
}

#[pyclass]
pub struct KsCr3bpResultPy {
    #[pyo3(get)] pub n_segments:         usize,
    #[pyo3(get)] pub jacobi_error:       f64,
    #[pyo3(get)] pub final_cartesian:    Vec<f64>,
    #[pyo3(get)] pub nk_bound_max:       f64,
    #[pyo3(get)] pub bernstein_rho_mean: f64,
}

// ---------------------------------------------------------------------------
// Chebyshev primitives
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "chebyshev_fit")]
fn chebyshev_fit_py<'py>(
    py: Python<'py>,
    vals: numpy::PyReadonlyArray1<'py, f64>,
) -> &'py numpy::PyArray1<f64> {
    let arr = ndarray::Array1::from_vec(vals.as_slice().unwrap().to_vec());
    let coeffs = math::chebyshev::values_to_coeffs(&arr);
    numpy::PyArray1::from_slice(py, coeffs.as_slice().unwrap())
}

#[pyfunction]
#[pyo3(name = "chebyshev_integrate")]
fn chebyshev_integrate_py<'py>(
    py: Python<'py>,
    c: numpy::PyReadonlyArray1<'py, f64>,
    u_init: f64,
    delta_s: f64,
) -> &'py numpy::PyArray1<f64> {
    let c_arr = ndarray::Array1::from_vec(c.as_slice().unwrap().to_vec());
    let d = math::chebyshev::integrate_coeffs(&c_arr, u_init, delta_s / 2.0);
    numpy::PyArray1::from_slice(py, d.as_slice().unwrap())
}

// ---------------------------------------------------------------------------
// Astrodynamics wrappers
// ---------------------------------------------------------------------------

#[pyfunction]
fn propagate_cr3bp_fast(
    x0: Vec<f64>,
    t_final: f64,
    mu: f64,
    n_cheb: Option<usize>,
    tol: Option<f64>,
) -> PyResult<PropagationResultPy> {
    let mut cfg = PicardConfig::default();
    if let Some(n) = n_cheb { cfg.n_cheb = n; }
    if let Some(t) = tol   { cfg.tol = t; }
    let x0_arr = Array1::from_vec(x0);
    let (traj, cj_err) = solvers::picard::propagate_cr3bp(&x0_arr, t_final, mu, &cfg);
    build_trajectory_result(traj, cj_err)
}

#[pyfunction]
fn cr3bp_jacobi(state: Vec<f64>, mu: f64) -> f64 {
    solvers::picard::jacobi_constant(&Array1::from_vec(state), mu)
}

#[pyfunction]
#[pyo3(name = "estimate_s_from_t")]
fn estimate_s_from_t_py(t_final: f64, x0: Vec<f64>, mu: f64) -> PyResult<f64> {
    if x0.len() != 6 { return Err(PyValueError::new_err("x0 must have length 6")); }
    let x0_arr = [x0[0], x0[1], x0[2], x0[3], x0[4], x0[5]];
    Ok(physics::ks_cr3bp::estimate_s_from_t(t_final, &x0_arr, mu))
}

#[pyfunction]
#[pyo3(name = "propagate_ks_cr3bp")]
fn propagate_ks_cr3bp_py(
    x0: Vec<f64>,
    s_final: f64,
    mu: f64,
    n_cheb: Option<usize>,
    tol: Option<f64>,
) -> PyResult<KsCr3bpResultPy> {
    if x0.len() != 6 { return Err(PyValueError::new_err("x0 must have length 6")); }
    let mut cfg = PicardConfig::default();
    if let Some(n) = n_cheb { cfg.n_cheb = n; }
    if let Some(t) = tol { cfg.tol = t; }
    
    let x0_arr = [x0[0], x0[1], x0[2], x0[3], x0[4], x0[5]];
    let res = physics::ks_cr3bp::propagate_ks_cr3bp(&x0_arr, s_final, mu, &cfg);
    
    Ok(KsCr3bpResultPy {
        n_segments: res.n_segments,
        jacobi_error: res.jacobi_error,
        final_cartesian: res.final_cartesian.to_vec(),
        nk_bound_max: res.nk_bound_max,
        bernstein_rho_mean: res.bernstein_rho_mean,
    })
}

#[pyfunction]
fn diagnose_system(
    rhs: PyObject,
    x0: Vec<f64>,
    t0: f64,
    dt_pilot: f64,
    ndim: usize,
) -> PyResult<LiftDiagnosisPy> {
    let system = physics::lifting::PythonOde { py_obj: rhs, ndim };
    let x0_arr = Array1::from_vec(x0);
    let d = physics::lifting::diagnose(&system, &x0_arr, t0, dt_pilot, 16, 3.0);
    Ok(LiftDiagnosisPy {
        bernstein_rho:   d.bernstein_rho,
        apply_fibration: d.apply_fibration,
        suggested_ds:    d.suggested_ds,
    })
}

#[pyfunction]
fn ks_map_py(u: Vec<f64>) -> PyResult<Vec<f64>> {
    if u.len() != 4 { return Err(PyValueError::new_err("u must have length 4")); }
    Ok(physics::lifting::ks_map(&[u[0], u[1], u[2], u[3]]).to_vec())
}

#[pyfunction]
fn ks_inverse_py(r: Vec<f64>) -> PyResult<Vec<f64>> {
    if r.len() != 3 { return Err(PyValueError::new_err("r must have length 3")); }
    match physics::lifting::ks_inverse(&[r[0], r[1], r[2]]) {
        Some(u) => Ok(u.to_vec()),
        None    => Err(PyValueError::new_err("|r|=0: inverse KS map undefined")),
    }
}

#[pyfunction]
fn lc_map_py(u: Vec<f64>) -> PyResult<Vec<f64>> {
    if u.len() != 2 { return Err(PyValueError::new_err("u must have length 2")); }
    Ok(physics::lifting::lc_map(&[u[0], u[1]]).to_vec())
}

#[pyfunction]
fn lc_inverse_py(r: Vec<f64>) -> PyResult<Vec<f64>> {
    if r.len() != 2 { return Err(PyValueError::new_err("r must have length 2")); }
    match physics::lifting::lc_inverse(&[r[0], r[1]]) {
        Some(u) => Ok(u.to_vec()),
        None    => Err(PyValueError::new_err("|r|=0: inverse LC map undefined")),
    }
}

// ---------------------------------------------------------------------------
// Borel-Padé & Resurgence wrappers
// ---------------------------------------------------------------------------

#[pyfunction]
fn borel_pade_laplace_py(coeffs: Vec<f64>, z: f64, n_gl: usize) -> f64 {
    borel_pade::borel_pade_laplace(&coeffs, z, n_gl)
}

#[pyfunction]
fn extract_singularities_py(coeffs: Vec<f64>) -> Vec<f64> {
    borel_pade::extract_singularities(&coeffs)
}

#[pyfunction]
fn median_resummation_py(coeffs: Vec<f64>, z: f64, epsilon: f64, n_gl: usize) -> f64 {
    borel_pade::median_resummation(&coeffs, z, epsilon, n_gl)
}

#[pyfunction]
fn auto_median_resummation_py(coeffs: Vec<f64>, z: f64, n_gl: usize) -> f64 {
    borel_pade::auto_median_resummation(&coeffs, z, n_gl)
}

#[pyfunction]
#[pyo3(name = "resurgent_pseudoinverse")]
fn resurgent_pseudoinverse_py(
    py: Python<'_>,
    a_func: PyObject,
    eps_min: f64,
    eps_max: f64,
    n_cheb: usize,
) -> PyResult<(Py<numpy::PyArray2<f64>>, Py<numpy::PyArray2<f64>>)> {
    let func = |eps: f64| -> nalgebra::DMatrix<f64> {
        let result = a_func.as_ref(py).call1((eps,)).unwrap();
        let pyarray: numpy::PyReadonlyArray2<f64> = result.extract().unwrap();
        let arr = pyarray.as_array();
        let mut mat = nalgebra::DMatrix::zeros(arr.nrows(), arr.ncols());
        for i in 0..arr.nrows() {
            for j in 0..arr.ncols() { mat[(i, j)] = arr[[i, j]]; }
        }
        mat
    };

    let res = resurgent_ps::resurgent_pseudoinverse(func, eps_min, eps_max, n_cheb);

    let to_py = |m: &nalgebra::DMatrix<f64>| -> Py<numpy::PyArray2<f64>> {
        let (nr, nc) = (m.nrows(), m.ncols());
        let arr = ndarray::Array2::from_shape_fn((nr, nc), |(i, j)| m[(i, j)]);
        arr.to_pyarray(py).to_owned()
    };

    Ok((to_py(&res.r_minus_1), to_py(&res.r_0)))
}

// ---------------------------------------------------------------------------
// JIT propagation (C-ABI function pointers from codegen.py)
// ---------------------------------------------------------------------------

type CAbiFunc = unsafe extern "C" fn(
    x_ptr:   *const f64,
    t_ptr:   *const f64,
    out_ptr: *mut  f64,
    ndim:    usize,
    npts:    usize,
);

#[pyfunction]
#[pyo3(name = "propagate_custom_c_abi",
       signature = (x0, t_final, rhs_ptr_val, jac_ptr_val=None, uk_ptr_val=None,
                    n_cheb=None, tol=None, certify=None, propagate_stm=None))]
fn propagate_custom_c_abi_py(
    x0:           Vec<f64>,
    t_final:      f64,
    rhs_ptr_val:  usize,
    jac_ptr_val:  Option<usize>,
    uk_ptr_val:   Option<usize>,
    n_cheb:       Option<usize>,
    tol:          Option<f64>,
    certify:      Option<bool>,
    propagate_stm: Option<bool>,
) -> PyResult<PropagationResultPy> {
    let mut config = PicardConfig::default();
    if let Some(n) = n_cheb  { config.n_cheb = n; }
    if let Some(t) = tol     { config.tol = t; }
    if let Some(c) = certify { config.certify = c; }

    let ndim   = x0.len();
    let do_stm = propagate_stm.unwrap_or(false);

    let rhs_c_fn: CAbiFunc = unsafe { std::mem::transmute(rhs_ptr_val as *const ()) };
    let rhs_closure = move |x: &ndarray::Array2<f64>, t: &ndarray::Array1<f64>| {
        let npts = x.ncols();
        let mut out = ndarray::Array2::<f64>::zeros((ndim, npts));
        unsafe { rhs_c_fn(x.as_ptr(), t.as_ptr(), out.as_mut_ptr(), ndim, npts); }
        out
    };

    type DynRhs = Box<dyn Fn(&ndarray::Array2<f64>, &ndarray::Array1<f64>) -> ndarray::Array2<f64>>;

    let jac_boxed: Option<DynRhs> = jac_ptr_val.map(|ptr| {
        let f: CAbiFunc = unsafe { std::mem::transmute(ptr as *const ()) };
        let b: DynRhs = Box::new(move |x: &ndarray::Array2<f64>, t: &ndarray::Array1<f64>| {
            let npts = x.ncols();
            let mut out = ndarray::Array2::<f64>::zeros((ndim * ndim, npts));
            unsafe { f(x.as_ptr(), t.as_ptr(), out.as_mut_ptr(), ndim, npts); }
            out
        });
        b
    });

    let uk_boxed: Option<DynRhs> = uk_ptr_val.map(|ptr| {
        let f: CAbiFunc = unsafe { std::mem::transmute(ptr as *const ()) };
        let b: DynRhs = Box::new(move |x: &ndarray::Array2<f64>, t: &ndarray::Array1<f64>| {
            let npts = x.ncols();
            let mut out = ndarray::Array2::<f64>::zeros((ndim, npts));
            unsafe { f(x.as_ptr(), t.as_ptr(), out.as_mut_ptr(), ndim, npts); }
            out
        });
        b
    });

    let traj = if do_stm {
        let jac = jac_boxed.as_deref()
            .ok_or_else(|| PyValueError::new_err("Jacobian required for STM propagation"))?;
        let mut y0 = Vec::with_capacity(ndim + ndim * ndim);
        y0.extend_from_slice(&x0);
        for r in 0..ndim {
            for c in 0..ndim { y0.push(if r == c { 1.0 } else { 0.0 }); }
        }
        let y0_arr = ndarray::Array1::from_vec(y0);
        let stm_rhs = |y: &ndarray::Array2<f64>, t: &ndarray::Array1<f64>| {
            solvers::stm::stm_rhs_batch(&rhs_closure, &jac, y, t, ndim)
        };
        solvers::picard::propagate_custom(&stm_rhs, None, None, &y0_arr, 0.0, t_final, &config, false)
    } else {
        let x0_arr = ndarray::Array1::from_vec(x0);
        solvers::picard::propagate_custom(
            &rhs_closure,
            jac_boxed.as_deref(),
            uk_boxed.as_deref(),
            &x0_arr, 0.0, t_final, &config, false,
        )
    };

    build_trajectory_result(traj, 0.0)
}

// ---------------------------------------------------------------------------
// Udwadia-Kalaba wrappers
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "compute_uk_force")]
fn compute_uk_force_py<'py>(
    py: Python<'py>,
    m_arr: numpy::PyReadonlyArray2<'py, f64>,
    a_arr: numpy::PyReadonlyArray2<'py, f64>,
    b_arr: numpy::PyReadonlyArray1<'py, f64>,
    q_arr: numpy::PyReadonlyArray1<'py, f64>,
    tol: f64,
) -> &'py numpy::PyArray1<f64> {
    let m = to_nalgebra_matrix2(m_arr);
    let a = to_nalgebra_matrix2(a_arr);
    let b = nalgebra::DVector::from_column_slice(b_arr.as_slice().unwrap());
    let q = nalgebra::DVector::from_column_slice(q_arr.as_slice().unwrap());
    let fc = physics::constrain_uk::compute_uk_force(&m, &a, &b, &q, tol);
    numpy::PyArray1::from_slice(py, fc.as_slice())
}

#[pyfunction]
#[pyo3(name = "compute_uk_force_identity")]
fn compute_uk_force_identity_py<'py>(
    py: Python<'py>,
    a_arr: numpy::PyReadonlyArray2<'py, f64>,
    b_arr: numpy::PyReadonlyArray1<'py, f64>,
    q_arr: numpy::PyReadonlyArray1<'py, f64>,
    tol: f64,
) -> &'py numpy::PyArray1<f64> {
    let a = to_nalgebra_matrix2(a_arr);
    let b = nalgebra::DVector::from_column_slice(b_arr.as_slice().unwrap());
    let q = nalgebra::DVector::from_column_slice(q_arr.as_slice().unwrap());
    let fc = physics::constrain_uk::compute_uk_force_identity(&a, &b, &q, tol);
    numpy::PyArray1::from_slice(py, fc.as_slice())
}

fn to_nalgebra_matrix2(arr: numpy::PyReadonlyArray2<f64>) -> nalgebra::DMatrix<f64> {
    let v = arr.as_array();
    nalgebra::DMatrix::from_row_slice(v.nrows(), v.ncols(), v.as_slice().unwrap())
}

#[pyfunction]
#[pyo3(name = "evaluate_clenshaw")]
fn evaluate_clenshaw_py<'py>(
    py: Python<'py>,
    coeffs: numpy::PyReadonlyArray2<'py, f64>,
    tau: f64,
) -> &'py numpy::PyArray1<f64> {
    let c_view = coeffs.as_array();
    let ndim   = c_view.nrows();
    let tau_c  = tau.clamp(-1.0, 1.0);
    let mut out = ndarray::Array1::<f64>::zeros(ndim);
    for i in 0..ndim {
        let row = ndarray::Array1::from_vec(c_view.row(i).to_owned().to_vec());
        out[i] = math::chebyshev::clenshaw(&row, tau_c);
    }
    numpy::PyArray1::from_slice(py, out.as_slice().unwrap())
}

#[pyfunction]
#[pyo3(name = "compute_apc_collocation")]
fn compute_apc_collocation_py(
    moments: Vec<f64>,
    n_points: usize,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    math::apc::compute_apc_collocation(&moments, n_points)
        .map_err(PyValueError::new_err)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn _asp_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PicardConfigPy>()?;
    m.add_class::<PicardSolverPy>()?;
    m.add_class::<PropagationResultPy>()?;
    m.add_class::<LiftDiagnosisPy>()?;
    m.add_class::<KsCr3bpResultPy>()?;

    m.add_function(wrap_pyfunction!(chebyshev_fit_py, m)?)?;
    m.add_function(wrap_pyfunction!(chebyshev_integrate_py, m)?)?;

    m.add_function(wrap_pyfunction!(propagate_cr3bp_fast, m)?)?;
    m.add_function(wrap_pyfunction!(cr3bp_jacobi, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_s_from_t_py, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_ks_cr3bp_py, m)?)?;
    
    m.add_function(wrap_pyfunction!(diagnose_system, m)?)?;
    m.add_function(wrap_pyfunction!(ks_map_py, m)?)?;
    m.add_function(wrap_pyfunction!(ks_inverse_py, m)?)?;
    m.add_function(wrap_pyfunction!(lc_map_py, m)?)?;
    m.add_function(wrap_pyfunction!(lc_inverse_py, m)?)?;

    m.add_function(wrap_pyfunction!(borel_pade_laplace_py, m)?)?;
    m.add_function(wrap_pyfunction!(extract_singularities_py, m)?)?;
    m.add_function(wrap_pyfunction!(resurgent_pseudoinverse_py, m)?)?;
    m.add_function(wrap_pyfunction!(median_resummation_py, m)?)?;
    m.add_function(wrap_pyfunction!(auto_median_resummation_py, m)?)?;

    m.add_function(wrap_pyfunction!(propagate_custom_c_abi_py, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_clenshaw_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_uk_force_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_uk_force_identity_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_apc_collocation_py, m)?)?;

    m.add("__version__", "0.1.0")?;
    Ok(())
}