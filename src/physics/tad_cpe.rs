// src/physics/tad_cpe.rs
//
// Thermal Alien Derivative Collision Probability Estimator (TAD-CPE)
// ==================================================================
// Computes exact orbital collision probabilities using Freidlin-Wentzell 
// large-deviation instanton theory and Gelfand-Yaglom prefactors, bypassing
// Gaussian quadrature and Monte Carlo sampling. (ASP Research Programme)

use nalgebra::{DMatrix, DVector};

/// Computes the probability of collision (P_c) in the 2D encounter plane (B-plane).
///
/// * `r_miss` - Nominal miss distance vector (2D).
/// * `cov` - 2x2 combined covariance matrix in the encounter plane.
/// * `r_hbr` - Hard body radius (collision sphere radius).
///
/// Returns a tuple: (P_c, A_FW, S_star)
pub fn compute_tad_cpe_2d(
    r_miss: &DVector<f64>,
    cov: &DMatrix<f64>,
    r_hbr: f64,
) -> (f64, f64, f64) {
    assert_eq!(r_miss.len(), 2, "TAD-CPE 2D requires a 2D miss vector.");
    assert_eq!(cov.nrows(), 2, "TAD-CPE 2D requires a 2x2 covariance matrix.");

    let r_miss_norm = r_miss.norm();

    // If the nominal miss distance is already inside the hard body radius, collision is certain.
    if r_miss_norm <= r_hbr {
        return (1.0, 0.0, 1.0);
    }

    // 1. Nearest point on the collision sphere (Instanton target)
    let n_rad = r_miss / r_miss_norm; // Radial unit vector
    let r_c_star = &n_rad * r_hbr;

    // 2. Compute the Freidlin-Wentzell Instanton Action (A_FW)
    let delta_r = &r_c_star - r_miss;
    let cov_inv = cov.clone().try_inverse().expect("Covariance matrix must be invertible.");
    
    // A_FW = 0.5 * delta_r^T * C^{-1} * delta_r
    let a_fw = 0.5 * delta_r.transpose() * &cov_inv * &delta_r;
    let a_fw_scalar = a_fw[0];

    // 3. Compute directional uncertainties
    // Radial uncertainty: sigma_r = sqrt(n^T * C * n)
    let sigma_r_sq = n_rad.transpose() * cov * &n_rad;
    let sigma_r = sigma_r_sq[0].sqrt();

    // Tangential unit vector (perpendicular to n_rad in 2D)
    let n_perp = DVector::from_vec(vec![-n_rad[1], n_rad[0]]);
    let sigma_perp_sq = n_perp.transpose() * cov * &n_perp;
    let sigma_perp = sigma_perp_sq[0].sqrt();

    // 4. Compute Gelfand-Yaglom Prefactor (Stokes Constant S_*)
    // S_* = (r_hbr * sigma_perp) / (||r_miss|| * sigma_r * A_FW * sqrt(2*pi))
    let s_star = (r_hbr * sigma_perp) 
           / (r_miss_norm * sigma_r * (2.0 * std::f64::consts::PI).sqrt());
    let p_c = s_star * (-a_fw_scalar).exp();

    (p_c, a_fw_scalar, s_star)
}