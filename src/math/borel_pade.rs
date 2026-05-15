// src/math/borel_pade.rs
//
// The Borel-Padé-Laplace Engine
// ======================================================================

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use num_complex::Complex;

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
            let w = v * v;
            (x, w)
        })
        .collect();
        
    nodes_weights.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    let nodes = nodes_weights.iter().map(|nw| nw.0).collect();
    let weights = nodes_weights.iter().map(|nw| nw.1).collect();
    
    (nodes, weights)
}

pub fn robust_pade(coeffs: &[Complex<f64>]) -> (Vec<Complex<f64>>, Vec<Complex<f64>>) {
    let n = coeffs.len() - 1;
    let m = n / 2;
    let l = n - m;

    // Pad to (m+1) x (m+1) to ensure full V^T matrix from SVD
    let mut t = DMatrix::<Complex<f64>>::zeros(m + 1, m + 1);
    for i in 0..m {
        for j in 0..=m {
            let idx = l as isize + 1 + i as isize - j as isize;
            if idx >= 0 && (idx as usize) < coeffs.len() {
                t[(i, j)] = coeffs[idx as usize];
            }
        }
    }

    let svd = t.svd(true, true);
    let v_t = svd.v_t.expect("SVD failed to compute V^T");
    
    let mut q = Vec::with_capacity(m + 1);
    for j in 0..=m {
        q.push(v_t[(m, j)].conj());
    }

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

/// Robust Durand-Kerner complex polynomial root finder
pub fn polynomial_roots(coeffs: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let mut d = coeffs.len() - 1;
    while d > 0 && coeffs[d].norm() < 1e-14 {
        d -= 1;
    }
    if d == 0 { return vec![]; }
    
    let mut roots = Vec::with_capacity(d);
    let radius = 1.0;
    let center = Complex::new(0.0, 0.0);
    for i in 0..d {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (d as f64) + 0.1;
        roots.push(center + Complex::new(radius * angle.cos(), radius * angle.sin()));
    }
    
    let max_iter = 2000;
    for _ in 0..max_iter {
        let mut max_diff = 0.0f64;
        let mut next_roots = roots.clone();
        for i in 0..d {
            let z = roots[i];
            let mut p_val = Complex::new(0.0, 0.0);
            let mut z_pow = Complex::new(1.0, 0.0);
            for &c in &coeffs[0..=d] {
                p_val += c * z_pow;
                z_pow *= z;
            }
            
            let mut denominator = coeffs[d];
            for j in 0..d {
                if i != j {
                    denominator *= z - roots[j];
                }
            }
            
            if denominator.norm() > 1e-14 {
                let diff = p_val / denominator;
                next_roots[i] -= diff;
                max_diff = max_diff.max(diff.norm());
            }
        }
        roots = next_roots;
        if max_diff < 1e-12 {
            break;
        }
    }
    roots
}

pub fn evaluate_rational(p: &[Complex<f64>], q: &[Complex<f64>], z: Complex<f64>) -> Complex<f64> {
    let mut num = Complex::new(0.0, 0.0);
    let mut den = Complex::new(0.0, 0.0);
    let mut z_pow = Complex::new(1.0, 0.0);
    
    for &c in p { num += c * z_pow; z_pow *= z; }
    z_pow = Complex::new(1.0, 0.0);
    for &c in q { den += c * z_pow; z_pow *= z; }
    
    num / den
}

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
    
    for (x, w) in nodes.into_iter().zip(weights.into_iter()) {
        let zeta = z * x; 
        let r = evaluate_rational(&p, &q, zeta);
        sum += r * w * z;
    }
    
    sum.re
}

pub fn filter_froissart_doublets(
    p_roots: &[Complex<f64>],
    q_roots: &[Complex<f64>],
    delta: f64,
) -> Vec<Complex<f64>> {
    let mut cancelled_q = vec![false; q_roots.len()];
    for &p_root in p_roots {
        if let Some(idx) = q_roots.iter().enumerate()
            .filter(|(i, _)| !cancelled_q[*i])
            .min_by(|(_, a), (_, b)| {
                (**a - p_root).norm().partial_cmp(&(**b - p_root).norm()).unwrap()
            })
            .map(|(i, _)| i)
        {
            if (q_roots[idx] - p_root).norm() < delta {
                cancelled_q[idx] = true;
            }
        }
    }
    q_roots.iter().enumerate()
        .filter(|(i, _)| !cancelled_q[*i])
        .map(|(_, &r)| r)
        .collect()
}

pub fn extract_singularities(raw_coeffs: &[f64]) -> Vec<f64> {
    let mut borel_coeffs = Vec::with_capacity(raw_coeffs.len());
    let mut fact = 1.0;
    
    for (n, &c) in raw_coeffs.iter().enumerate() {
        if n > 0 { fact *= n as f64; }
        borel_coeffs.push(Complex::new(c / fact, 0.0));
    }
    let (p, q) = robust_pade(&borel_coeffs);
    let p_roots = polynomial_roots(&p);
    let q_roots_raw = polynomial_roots(&q);
    
    let q_roots_clean = filter_froissart_doublets(&p_roots, &q_roots_raw, 1e-3);
    
    let mut physical_poles = Vec::new();
    for root in q_roots_clean {
        if root.im.abs() < 1e-3 {
            physical_poles.push(root.re);
        }
    }
    physical_poles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    physical_poles
}

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
    
    let z_eff = Complex::new(0.0, epsilon).exp() * z_val;
    let mut sum = Complex::new(0.0, 0.0);
    
    for (x, w) in nodes.into_iter().zip(weights.into_iter()) {
        let zeta = z_eff * x; 
        let r = evaluate_rational(&p, &q, zeta);
        sum += r * w * z_eff;
    }
    
    sum
}

pub fn median_resummation(
    raw_coeffs: &[f64], 
    z_val: f64, 
    epsilon: f64, 
    n_gl: usize
) -> f64 {
    let s_plus = lateral_borel_sum(raw_coeffs, z_val, epsilon, n_gl);
    let s_minus = lateral_borel_sum(raw_coeffs, z_val, -epsilon, n_gl);
    
    let m_val = (s_plus + s_minus) / 2.0;
    
    m_val.re
}

pub fn auto_median_resummation(raw_coeffs: &[f64], z_val: f64, n_gl: usize) -> f64 {
    let mut borel_coeffs = Vec::with_capacity(raw_coeffs.len());
    let mut fact = 1.0;
    for (n, &c) in raw_coeffs.iter().enumerate() {
        if n > 0 { fact *= n as f64; }
        borel_coeffs.push(Complex::new(c / fact, 0.0));
    }

    let (_, q) = robust_pade(&borel_coeffs);
    let q_roots = polynomial_roots(&q);
    
    let mut min_mag = f64::MAX;
    let mut dominant_angle = 0.0;
    for root in q_roots {
        let mag = root.norm();
        if mag > 1e-10 && mag < min_mag {
            min_mag = mag;
            dominant_angle = root.im.atan2(root.re);
        }
    }
    if min_mag == f64::MAX { dominant_angle = 0.0; }
    
    let eps = 1e-5;
    let s_plus = lateral_borel_sum(raw_coeffs, z_val, dominant_angle + eps, n_gl);
    let s_minus = lateral_borel_sum(raw_coeffs, z_val, dominant_angle - eps, n_gl);
    
    ((s_plus + s_minus) / 2.0).re
}