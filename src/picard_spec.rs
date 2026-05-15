// src/picard_spec.rs

use ndarray::{Array1, ArrayView1};
use rustdct::DctPlanner;

/// Computes the Chebyshev coefficients from function values evaluated at CGL nodes.
/// Uses a Type-I Discrete Cosine Transform (DCT-I).
pub fn chebyshev_fit(vals: ArrayView1<f64>) -> Array1<f64> {
    let n = vals.len();
    let mut planner = DctPlanner::new();
    let dct = planner.plan_dct1(n);

    let mut buffer = vals.to_vec();
    dct.process_dct1(&mut buffer);

    // Normalize the DCT output to standard Chebyshev coefficients
    let n_float = (n - 1) as f64;
    for (k, val) in buffer.iter_mut().enumerate() {
        if k == 0 || k == n - 1 {
            *val /= 2.0 * n_float;   // endpoint: divide by 2N
        } else {
            *val /= n_float;          // interior: divide by N
        }
    }
    
    Array1::from_vec(buffer)
}

/// Integrates a Chebyshev series in coefficient space.
/// Represents the Picard integral operator: u(sigma) = u_init + (delta_s / 2) * int_{-1}^{sigma} F(s) ds
pub fn chebyshev_integrate(
    c: ArrayView1<f64>,
    u_init: f64,
    delta_s: f64,
) -> Array1<f64> {
    let n = c.len();
    if n == 0 {
        return Array1::from_vec(vec![u_init]);
    }
    
    // The integrated series has degree N + 1
    let mut d = Array1::<f64>::zeros(n + 1);

    if n == 1 {
        d[1] = c[0];
    } else {
        // Standard Chebyshev integration recurrence
        d[1] = c[0] - c[2] / 2.0;
        for k in 2..n {
            let c_k_plus_1 = if k + 1 < n { c[k + 1] } else { 0.0 };
            d[k] = (c[k - 1] - c_k_plus_1) / (2.0 * k as f64);
        }
        d[n] = c[n - 1] / (2.0 * n as f64);
    }

    // Determine d_0 such that the integral evaluates to 0 at sigma = -1
    // I(-1) = sum_{k=0}^N d_k (-1)^k = 0 => d_0 = -sum_{k=1}^N d_k (-1)^k
    let mut sum_alt = 0.0;
    for k in 1..=n {
        if k % 2 == 0 {
            sum_alt += d[k];
        } else {
            sum_alt -= d[k];
        }
    }
    d[0] = -sum_alt;

    // Apply the physical time scaling factor
    let scale = delta_s / 2.0;
    for k in 0..=n {
        d[k] *= scale;
    }
    
    // Add the initial condition to the constant term
    d[0] += u_init;

    d
}