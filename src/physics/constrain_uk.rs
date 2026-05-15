// src/physics/constrain_uk.rs
//
// Udwadia-Kalaba Exact Constraint Projection
// ======================================================================

use nalgebra::{DMatrix, DVector, SymmetricEigen};

pub fn fractional_mass_matrices(m: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
    let eigen = SymmetricEigen::new(m.clone());
    let n = m.nrows();
    
    let mut d_half = DMatrix::zeros(n, n);
    let mut d_inv_half = DMatrix::zeros(n, n);
    
    for i in 0..n {
        let val = eigen.eigenvalues[i].max(0.0).sqrt();
        d_half[(i, i)] = val;
        d_inv_half[(i, i)] = if val > 1e-14 { 1.0 / val } else { 0.0 };
    }
    
    let v = &eigen.eigenvectors;
    let v_t = v.transpose();
    
    let m_half = v * &d_half * &v_t;
    let m_inv_half = v * &d_inv_half * &v_t;
    
    (m_half, m_inv_half)
}

pub fn compute_uk_force(
    m: &DMatrix<f64>,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    q_force: &DVector<f64>,
    tol: f64,
) -> DVector<f64> {
    let (m_half, m_inv_half) = fractional_mass_matrices(m);
    
    let a_w = a * &m_inv_half;
    
    let svd = a_w.svd(true, true);
    let a_w_pinv = svd.pseudo_inverse(tol).expect("SVD pseudo-inverse failed");
    
    let m_inv_q = &m_inv_half * &m_inv_half * q_force;
    let diff = b - a * m_inv_q;
    
    &m_half * a_w_pinv * diff
}

pub fn compute_uk_force_identity(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    q_force: &DVector<f64>,
    tol: f64,
) -> DVector<f64> {
    let svd = a.clone().svd(true, true);
    let a_pinv = svd.pseudo_inverse(tol).expect("SVD pseudo-inverse failed");
    
    let diff = b - a * q_force;
    a_pinv * diff
}