// src/resurgent_ps.rs
//
// The Resurgent Moore-Penrose Pseudoinverse
// ======================================================================
// Implements the spectral extraction of the Hadamard finite part (R_0)
// and the alien derivative/pole residue (R_{-1}) of a singular matrix
// function A(eps) as eps -> 0 (Thesis Paper 31).

use nalgebra::DMatrix;
use ndarray::Array1;
use crate::math::chebyshev::{lobatto_nodes, values_to_coeffs, clenshaw};

/// Computes the Chebyshev coefficients of the derivative P'(x) given the
/// coefficients of P(x).
/// If P(x) = sum_{k=0}^N c_k T_k(x), then P'(x) = sum_{k=0}^{N-1} c'_k T_k(x).
fn cheb_deriv_coeffs(c: &[f64]) -> Vec<f64> {
    let n = c.len() - 1;
    let mut d = vec![0.0; n];
    if n == 0 { return d; }
    if n == 1 { 
        d[0] = c[1]; 
        return d; 
    }
    
    // c'_{k} = c'_{k+2} + 2(k+1)c_{k+1}
    d[n - 1] = 2.0 * n as f64 * c[n];
    for k in (1..n - 1).rev() {
        let next_d = if k + 2 < n { d[k + 2] } else { 0.0 };
        d[k] = next_d + 2.0 * (k + 1) as f64 * c[k + 1];
    }
    
    // c'_0 = c'_2 / 2 + c_1
    let d2 = if 2 < n { d[2] } else { 0.0 };
    d[0] = d2 / 2.0 + c[1];
    d
}

pub struct ResurgentPinvResult {
    pub r_minus_1: DMatrix<f64>, // The pole residue (UK Constraint Force)
    pub r_0: DMatrix<f64>,       // The Hadamard finite part (Resurgent P-inv)
}

/// Computes the Resurgent Pseudoinverse of a parameterized matrix function A(eps).
/// 
/// We avoid evaluating exactly at eps = 0 (the singularity). Instead, we evaluate
/// F(eps) = eps * A^+(eps) on a Chebyshev grid mapped to [eps_min, eps_max].
/// We then spectrally extrapolate F(0) and F'(0) to recover R_{-1} and R_0.
pub fn resurgent_pseudoinverse<F>(
    a_func: F,
    eps_min: f64,
    eps_max: f64,
    n_cheb: usize,
) -> ResurgentPinvResult
where
    F: Fn(f64) -> DMatrix<f64>,
{
    let nodes = lobatto_nodes(n_cheb);
    let mut f_matrices = Vec::with_capacity(n_cheb + 1);

    let mut nrows = 0;
    let mut ncols = 0;

    // 1. Evaluate F(eps) = eps * A^+(eps) at Lobatto nodes
    for &tau in nodes.iter() {
        // Map tau in [-1, 1] to eps in [eps_min, eps_max]
        let eps = 0.5 * (eps_max - eps_min) * tau + 0.5 * (eps_max + eps_min);
        
        let a = a_func(eps);
        if nrows == 0 {
            nrows = a.nrows();
            ncols = a.ncols();
        }
        
        // Standard SVD pseudo-inverse at a safe, non-singular distance
        let svd = a.svd(true, true);
        let a_pinv = svd.pseudo_inverse(1e-14).expect("SVD pseudo-inverse failed");
        
        let f_eps = a_pinv * eps;
        f_matrices.push(f_eps);
    }

    // A^+ has dimensions (ncols x nrows) relative to A
    let mut r_minus_1 = DMatrix::<f64>::zeros(ncols, nrows);
    let mut r_0 = DMatrix::<f64>::zeros(ncols, nrows);

    // 2. Extrapolate to eps = 0
    // eps = 0 maps to tau_0 in the Chebyshev domain
    let tau_0 = -(eps_max + eps_min) / (eps_max - eps_min);
    let dtau_deps = 2.0 / (eps_max - eps_min);

    // 3. Spectral coefficient extraction per matrix element
    for i in 0..ncols {
        for j in 0..nrows {
            let mut f_vals = Array1::zeros(n_cheb + 1);
            for k in 0..=n_cheb {
                f_vals[k] = f_matrices[k][(i, j)];
            }

            let c = values_to_coeffs(&f_vals);
            let c_slice = c.as_slice().unwrap();

            // R_{-1} = F(0)
            let val = clenshaw(&c, tau_0);
            r_minus_1[(i, j)] = val;

            // R_0 = dF/deps(0) = dtau/deps * dF/dtau(tau_0)
            let c_deriv = cheb_deriv_coeffs(c_slice);
            let deriv_val = clenshaw(&Array1::from_vec(c_deriv), tau_0);
            r_0[(i, j)] = deriv_val * dtau_deps;
        }
    }

    ResurgentPinvResult { r_minus_1, r_0 }
}