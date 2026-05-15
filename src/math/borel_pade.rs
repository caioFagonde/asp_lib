// src/borel_pade.rs
//
// The Borel-Padé-Laplace Engine
// ======================================================================
// Implements the rigorous singularity extraction and analytic continuation
// of factorially divergent perturbation series (Thesis Chapter 2 & 3).

use nalgebra::{ComplexField, DMatrix, DVector, SymmetricEigen};
use num_complex::Complex;

/// Generates Gauss-Laguerre quadrature nodes and weights via the Golub-Welsch algorithm.
/// Computes the eigenvalues and eigenvectors of the symmetric tridiagonal matrix.
pub fn gauss_laguerre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut diag = DVector::<f64>::zeros(n);
    let mut off_diag = DVector::<f64>::zeros(n - 1);
    
    for i in 0..n {
        diag[i] = (2 * i + 1) as f64;
        if i < n - 1 {
            off_diag[i] = (i + 1) as f64;
        }
    }
    
    let mut mat = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        mat[(i, i)] = diag[i];
        if i < n - 1 {
            mat[(i, i + 1)] = off_diag[i];
            mat[(i + 1, i)] = off_diag[i];
        }
    }
    
    let eigen = SymmetricEigen::new(mat);
    let mut nodes_weights: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let x = eigen.eigenvalues[i];
            let v = eigen.eigenvectors[(0, i)];
            let w = v * v; // Weight is the square of the first component of the normalized eigenvector
            (x, w)
        })
        .collect();
        
    nodes_weights.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    let nodes = nodes_weights.iter().map(|nw| nw.0).collect();
    let weights = nodes_weights.iter().map(|nw| nw.1).collect();
    
    (nodes, weights)
}

/// Computes the robust Padé approximant [L/M] using SVD on the Toeplitz matrix.
/// Returns the coefficients of the numerator P and denominator Q.
pub fn robust_pade(coeffs: &[Complex<f64>]) -> (Vec<Complex<f64>>, Vec<Complex<f64>>) {
    let n = coeffs.len() - 1;
    let m = n / 2;
    let l = n - m;

    // Build the Toeplitz matrix for the system C * q = 0
    let mut t = DMatrix::<Complex<f64>>::zeros(m, m + 1);
    for i in 0..m {
        for j in 0..=m {
            let idx = l as isize + 1 + i as isize - j as isize;
            if idx >= 0 && (idx as usize) < coeffs.len() {
                t[(i, j)] = coeffs[idx as usize];
            }
        }
    }

    // Compute SVD to find the null space (the denominator coefficients q)
    let svd = t.svd(true, true);
    let v_t = svd.v_t.expect("SVD failed to compute V^T");
    
    // The last row of V^T (which is V^H) corresponds to the smallest singular value.
    // The null vector is the conjugate of this row.
    let mut q = Vec::with_capacity(m + 1);
    for j in 0..=m {
        q.push(v_t[(m, j)].conj());
    }

    // Compute numerator coefficients p_k = sum_{j=0}^k c_{k-j} q_j
    let mut p = Vec::with_capacity(l + 1);
    for k in 0..=l {
        let mut sum = Complex::new(0.0, 0.0);
        for j in 0..=k.min(m) {
            sum += coeffs[k - j] * q[j];
        }
        p.push(sum);
    }

    (p, q)
}

/// Extracts the roots of a polynomial given its coefficients via the Companion Matrix.
pub fn polynomial_roots(coeffs: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let mut d = coeffs.len() - 1;
    while d > 0 && coeffs[d].norm() < 1e-14 {
        d -= 1;
    }
    if d == 0 { return vec![]; }
    
    let lead = coeffs[d];
    let mut comp = DMatrix::<Complex<f64>>::zeros(d, d);
    
    for i in 1..d {
        comp[(i, i - 1)] = Complex::new(1.0, 0.0);
    }
    for i in 0..d {
        comp[(i, d - 1)] = -coeffs[i] / lead;
    }
    
    comp.complex_eigenvalues().into_iter().cloned().collect()
}

/// Evaluates a rational function P(z)/Q(z)
pub fn evaluate_rational(p: &[Complex<f64>], q: &[Complex<f64>], z: Complex<f64>) -> Complex<f64> {
    let mut num = Complex::new(0.0, 0.0);
    let mut den = Complex::new(0.0, 0.0);
    let mut z_pow = Complex::new(1.0, 0.0);
    
    for &c in p { num += c * z_pow; z_pow *= z; }
    z_pow = Complex::new(1.0, 0.0);
    for &c in q { den += c * z_pow; z_pow *= z; }
    
    num / den
}

/// Executes the full Borel-Padé-Laplace resummation.
/// Expects raw perturbative coefficients a_n. Internally converts to Borel coefficients b_n = a_n / n!.
pub fn borel_pade_laplace(raw_coeffs: &[f64], z_val: f64, n_gl: usize) -> f64 {
    let mut borel_coeffs = Vec::with_capacity(raw_coeffs.len());
    let mut fact = 1.0;
    
    for (n, &c) in raw_coeffs.iter().enumerate() {
        if n > 0 { fact *= n as f64; }
        borel_coeffs.push(Complex::new(c / fact, 0.0));
    }

    let (p, q) = robust_pade(&borel_coeffs);
    let (nodes, weights) = gauss_laguerre(n_gl);
    
    let z = Complex::new(z_val, 0.0);
    let mut sum = Complex::new(0.0, 0.0);
    
    // Laplace Integral: int_0^infty e^(-x) R(z*x) * z dx
    for (x, w) in nodes.into_iter().zip(weights.into_iter()) {
        let zeta = z * x; 
        let r = evaluate_rational(&p, &q, zeta);
        sum += r * w * z;
    }
    
    sum.re
}

/// Extracts physical Borel singularities (instanton actions) from the raw coefficients.
pub fn extract_singularities(raw_coeffs: &[f64]) -> Vec<f64> {
    let mut borel_coeffs = Vec::with_capacity(raw_coeffs.len());
    let mut fact = 1.0;
    
    for (n, &c) in raw_coeffs.iter().enumerate() {
        if n > 0 { fact *= n as f64; }
        borel_coeffs.push(Complex::new(c / fact, 0.0));
    }

    let (_, q) = robust_pade(&borel_coeffs);
    let roots = polynomial_roots(&q);
    
    let mut physical_poles = Vec::new();
    for root in roots {
        // Filter for poles that lie primarily on the real axis (the physical singularities)
        if root.im.abs() < 1e-3 {
            physical_poles.push(root.re);
        }
    }
    
    physical_poles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    physical_poles
}

// src/math/borel_pade.rs (Additions)

/// Computes the Lateral Borel Sum $\mathcal{S}^{\pm\epsilon}$.
/// Rotates the Laplace integration contour by angle `epsilon` in the complex plane
/// to avoid singularities lying directly on the Stokes line.
pub fn lateral_borel_sum(
    raw_coeffs: &[f64], 
    z_val: f64, 
    epsilon: f64, 
    n_gl: usize
) -> Complex<f64> {
    let mut borel_coeffs = Vec::with_capacity(raw_coeffs.len());
    let mut fact = 1.0;
    
    for (n, &c) in raw_coeffs.iter().enumerate() {
        if n > 0 { fact *= n as f64; }
        borel_coeffs.push(Complex::new(c / fact, 0.0));
    }

    let (p, q) = robust_pade(&borel_coeffs);
    let (nodes, weights) = gauss_laguerre(n_gl);
    
    // Complexify the coupling parameter: z_eff = z * e^{i * epsilon}
    let z_eff = Complex::new(0.0, epsilon).exp() * z_val;
    let mut sum = Complex::new(0.0, 0.0);
    
    // Directional Laplace Integral: \int_0^\infty e^{-x} R(z_eff * x) * z_eff dx
    for (x, w) in nodes.into_iter().zip(weights.into_iter()) {
        let zeta = z_eff * x; 
        let r = evaluate_rational(&p, &q, zeta);
        sum += r * w * z_eff;
    }
    
    sum
}

/// Executes the Measure operator (\mathcal{M}).
/// Computes the Median Resummation by taking the arithmetic mean of the 
/// upper and lower lateral Borel sums, explicitly canceling the imaginary 
/// ambiguities (instanton tunneling rates) to project onto the physical real line.
pub fn median_resummation(
    raw_coeffs: &[f64], 
    z_val: f64, 
    epsilon: f64, 
    n_gl: usize
) -> f64 {
    let s_plus = lateral_borel_sum(raw_coeffs, z_val, epsilon, n_gl);
    let s_minus = lateral_borel_sum(raw_coeffs, z_val, -epsilon, n_gl);
    
    // The Cauchy Principal Value is the real part of the average.
    // By definition of the Stokes automorphism, the imaginary parts are equal and opposite.
    let m_val = (s_plus + s_minus) / 2.0;
    
    m_val.re
}