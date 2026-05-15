// src/math/chebyshev.rs

use ndarray::{Array1, Array2};
use rustdct::DctPlanner;
use std::sync::Mutex;
use std::collections::HashMap;

// Cache DCT planners to avoid re-allocating them on every segment
lazy_static::lazy_static! {
    static ref DCT_PLANNER: Mutex<DctPlanner<f64>> = Mutex::new(DctPlanner::new());
}

pub fn lobatto_nodes(n: usize) -> Array1<f64> {
    let pi = std::f64::consts::PI;
    Array1::from_shape_fn(n + 1, |j| (pi * j as f64 / n as f64).cos())
}

/// O(N log N) Fast Chebyshev Transform using DCT-I
pub fn values_to_coeffs(fvals: &Array1<f64>) -> Array1<f64> {
    let n = fvals.len() - 1;
    if n == 0 { return fvals.clone(); }
    
    let mut buffer = fvals.to_vec();
    {
        let mut planner = DCT_PLANNER.lock().unwrap();
        let dct = planner.plan_dct1(n + 1);
        dct.process_dct1(&mut buffer);
    }

    let n_float = n as f64;
    for (k, val) in buffer.iter_mut().enumerate() {
        if k == 0 || k == n {
            *val /= 2.0 * n_float; // Halve endpoints to match synthesis convention
        } else {
            *val /= n_float;
        }
    }
    
    Array1::from_vec(buffer)
}

pub fn clenshaw(c: &Array1<f64>, x: f64) -> f64 {
    let n = c.len() - 1;
    if n == 0 { return c[0]; }
    let mut b2 = 0.0f64;
    let mut b1 = 0.0f64;
    for k in (1..=n).rev() {
        let b0 = c[k] + 2.0 * x * b1 - b2;
        b2 = b1;
        b1 = b0;
    }
    c[0] + x * b1 - b2
}

pub fn coeffs_to_values(c: &Array1<f64>) -> Array1<f64> {
    let n = c.len() - 1;
    let nodes = lobatto_nodes(n);
    Array1::from_shape_fn(n + 1, |j| clenshaw(c, nodes[j]))
}

pub fn integrate_coeffs(c: &Array1<f64>, ic: f64, alpha: f64) -> Array1<f64> {
    let n = c.len() - 1;
    let a = |k: usize| -> f64 {
        let v = c[k];
        if k == 0 || k == n { 2.0 * v } else { v }
    };
    let a_ext = |k: usize| -> f64 {
        if k <= n { a(k) } else { 0.0 }
    };
    let mut d = Array1::zeros(n + 2);
    for k in 1..=n + 1 {
        let cm = a_ext(k - 1);
        let cp = a_ext(k + 1); 
        d[k] = (cm - cp) / (2.0 * k as f64) * alpha;
    }
    let sum: f64 = (1..=n + 1)
        .map(|k| if k % 2 == 0 { d[k] } else { -d[k] })
        .sum();
    d[0] = ic - sum;
    d
}

pub struct ChebVec {
    pub c: Array2<f64>,
    pub n: usize,
    pub ndim: usize,
}

impl ChebVec {
    pub fn zeros(ndim: usize, n: usize) -> Self {
        ChebVec { c: Array2::zeros((ndim, n + 1)), n, ndim }
    }

    pub fn from_value_matrix(fmat: &Array2<f64>) -> Self {
        let ndim = fmat.nrows();
        let n = fmat.ncols() - 1;
        let mut c = Array2::zeros((ndim, n + 1));
        for i in 0..ndim {
            let row = Array1::from_vec(fmat.row(i).to_vec());
            c.row_mut(i).assign(&values_to_coeffs(&row));
        }
        ChebVec { c, n, ndim }
    }

    pub fn to_value_matrix(&self) -> Array2<f64> {
        let mut out = Array2::zeros((self.ndim, self.n + 1));
        for i in 0..self.ndim {
            let row = Array1::from_vec(self.c.row(i).to_vec());
            out.row_mut(i).assign(&coeffs_to_values(&row));
        }
        out
    }

    pub fn estimate_bernstein_rho(&self) -> f64 {
        let n = self.n;
        if n < 8 { return 2.0; }
        let k_start = n / 2;
        let k_end = n;
        let npts = (k_end - k_start + 1) as f64;
        let log_mags: Vec<f64> = (k_start..=k_end).map(|k| {
            let mx = (0..self.ndim).map(|i| self.c[[i, k]].abs())
                .fold(0.0f64, f64::max);
            if mx > 1e-300 { mx.ln() } else { -690.0 }
        }).collect();
        let k_vals: Vec<f64> = (k_start..=k_end).map(|k| k as f64).collect();
        let km = k_vals.iter().sum::<f64>() / npts;
        let vm = log_mags.iter().sum::<f64>() / npts;
        let num: f64 = k_vals.iter().zip(log_mags.iter()).map(|(k,v)|(k-km)*(v-vm)).sum();
        let den: f64 = k_vals.iter().map(|k|(k-km).powi(2)).sum();
        if den < 1e-10 { return 2.0; }
        let slope = num / den;
        (-slope).exp().max(1.01)
    }
}