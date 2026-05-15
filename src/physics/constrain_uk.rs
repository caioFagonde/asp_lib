// src/physics/constrain_uk.rs
//
// Udwadia-Kalaba Exact Constraint Projection
// ======================================================================
// Implements the UK constraint force equation:
// F_c = M^{1/2} (A M^{-1/2})^+ (b - A M^{-1} Q)
//
// This guarantees exact constraint satisfaction (A q_ddot = b) at every
// point in the continuous Chebyshev domain without Lagrange multipliers.

use nalgebra::{DMatrix, DVector, SymmetricEigen};

/// Computes M^{1/2} and M^{-1/2} for a symmetric positive-definite mass matrix M.
/// Uses the spectral decomposition M = V D V^T.
pub fn fractional_mass_matrices(m: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
    // Note: We clone `m` because SymmetricEigen consumes it
    let eigen = SymmetricEigen::new(m.clone());
    let n = m.nrows();
    
    let mut d_half = DMatrix::zeros(n, n);
    let mut d_inv_half = DMatrix::zeros(n, n);
    
    for i in 0..n {
        // Guard against negative eigenvalues due to numerical noise
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

/// Computes the Udwadia-Kalaba constraint force for a general mass matrix.
///
/// * `m` - Mass matrix (n x n), symmetric positive-definite
/// * `a` - Constraint matrix (m x n)
/// * `b` - Constraint vector (m)
/// * `q_force` - Unconstrained applied forces Q (n)
/// * `tol` - SVD pseudo-inverse tolerance
pub fn compute_uk_force(
    m: &DMatrix<f64>,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    q_force: &DVector<f64>,
    tol: f64,
) -> DVector<f64> {
    let (m_half, m_inv_half) = fractional_mass_matrices(m);
    
    // A_w = A * M^{-1/2}
    let a_w = a * &m_inv_half;
    
    // A_w^+
    let svd = a_w.svd(true, true);
    let a_w_pinv = svd.pseudo_inverse(tol).expect("SVD pseudo-inverse failed");
    
    // diff = b - A * M^{-1} * Q
    // M^{-1} Q = M^{-1/2} * M^{-1/2} * Q
    let m_inv_q = &m_inv_half * &m_inv_half * q_force;
    let diff = b - a * m_inv_q;
    
    // F_c = M^{1/2} * A_w^+ * diff
    &m_half * a_w_pinv * diff
}

/// Optimized path: Computes the Udwadia-Kalaba constraint force when M = I.
/// F_c = A^+ (b - A Q)
pub fn compute_uk_force_identity(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    q_force: &DVector<f64>,
    tol: f64,
) -> DVector<f64> {
    let svd = a.svd(true, true);
    let a_pinv = svd.pseudo_inverse(tol).expect("SVD pseudo-inverse failed");
    
    let diff = b - a * q_force;
    a_pinv * diff
}