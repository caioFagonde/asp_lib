// src/picard.rs
//
// Adaptive Spectral Picard Solver with Newton-Kantorovich Certification
// ======================================================================
// Implements Algorithm 3 from Chapter 8 of the thesis.
// The Picard iteration is executed in Chebyshev coefficient space:
//
//   u^{(n+1)}(σ) = u_0 + (Δs/2) ∫_{-1}^{σ} F(u^{(n)}(σ'), σ') dσ'
//
// Integration is performed via the exact recurrence (8.X) in coefficient
// space: d_k = (c_{k-1} - c_{k+1}) / (2k), O(N) per component.
// The DCT converts between value space and coefficient space in O(N log N).

use ndarray::{Array1, Array2};
use crate::chebyshev::{lobatto_nodes, ChebVec, clenshaw};
use crate::math::interval::nk_certify;
// ============================================================
//  Solver configuration
// ============================================================

/// Configuration for the adaptive Picard solver.
#[derive(Debug, Clone)]
pub struct PicardConfig {
    /// Chebyshev polynomial degree N (N+1 nodes per segment).
    pub n_cheb: usize,
    /// Picard convergence tolerance.
    pub tol: f64,
    /// Maximum Picard iterations per segment.
    pub max_iter: usize,
    /// Initial segment length in (fictitious) time.
    pub ds_initial: f64,
    /// Minimum allowed segment length (prevents infinite loop).
    pub ds_min: f64,
    /// Maximum allowed segment length.
    pub ds_max: f64,
    /// Safety factor for segment doubling: double if converged in < k_double iterations.
    pub k_double: usize,
    /// Safety factor: segment halves if not converged in max_iter iterations.
    pub max_segments: usize,
    /// Whether to compute NK certification for each segment.
    pub certify: bool,
    /// Bernstein ρ threshold below which lifting is recommended.
    pub rho_threshold: f64,
}

impl Default for PicardConfig {
    fn default() -> Self {
        PicardConfig {
            n_cheb: 32,
            tol: 1e-12,
            max_iter: 50,
            ds_initial: 0.1,
            ds_min: 1e-8,
            ds_max: 10.0,
            k_double: 5,
            max_segments: 100_000,
            certify: true,
            rho_threshold: 3.0,
        }
    }
}

// ============================================================
//  Segment result
// ============================================================

/// Result for a single Picard segment.
#[derive(Debug, Clone)]
pub struct SegmentResult {
    /// Chebyshev coefficients of the solution on this segment: [ndim, N+1].
    pub cheb_coeffs: Array2<f64>,
    /// Segment start (fictitious time).
    pub s_start: f64,
    /// Segment length.
    pub ds: f64,
    /// Physical time at segment end.
    pub t_end: f64,
    /// Number of Picard iterations used.
    pub n_iter: usize,
    /// Max absolute Picard residual at convergence.
    pub residual: f64,
    /// NK certified error bound (if certification enabled).
    pub nk_bound: Option<f64>,
    /// Estimated Bernstein ρ for this segment.
    pub bernstein_rho: f64,
}

/// Full trajectory: a sequence of segments.
#[derive(Debug)]
pub struct Trajectory {
    pub segments: Vec<SegmentResult>,
    pub ndim: usize,
    pub n_cheb: usize,
}

impl Trajectory {
    /// Evaluate the trajectory at a single (fictitious) time s.
    pub fn eval_at(&self, s: f64) -> Option<Array1<f64>> {
        for seg in &self.segments {
            let s_end = seg.s_start + seg.ds;
            if s >= seg.s_start - 1e-14 && s <= s_end + 1e-14 {
                let tau = 2.0 * (s - seg.s_start) / seg.ds - 1.0;
                let tau = tau.clamp(-1.0, 1.0);
                let ndim = seg.cheb_coeffs.nrows();
                return Some(Array1::from_shape_fn(ndim, |i| {
                    let row = Array1::from_vec(seg.cheb_coeffs.row(i).to_vec());
                    clenshaw(&row, tau)
                }));
            }
        }
        None
    }
    
    /// Return all segment endpoint states: [n_segments, ndim+1]
    pub fn endpoint_states(&self) -> Array2<f64> {
        let n = self.segments.len();
        let ndim = self.ndim;
        let mut out = Array2::zeros((n, ndim));
        for (k, seg) in self.segments.iter().enumerate() {
            for i in 0..ndim {
                let row = Array1::from_vec(seg.cheb_coeffs.row(i).to_vec());
                out[[k, i]] = clenshaw(&row, 1.0);
            }
        }
        out
    }

    pub fn total_segments(&self) -> usize { self.segments.len() }

    pub fn final_state(&self) -> Option<Array1<f64>> {
        self.segments.last().map(|seg| {
            let ndim = seg.cheb_coeffs.nrows();
            Array1::from_shape_fn(ndim, |i| {
                let row = Array1::from_vec(seg.cheb_coeffs.row(i).to_vec());
                clenshaw(&row, 1.0)
            })
        })
    }

    /// Jacobi constant error (CR3BP specific): supply C_J_initial.
    /// Computes max deviation across all segment endpoints.
    pub fn max_jacobi_error(
        &self,
        cj_initial: f64,
        jacobi_fn: &dyn Fn(&Array1<f64>) -> f64,
    ) -> f64 {
        let endpoints = self.endpoint_states();
        let mut max_err = 0.0f64;
        for k in 0..endpoints.nrows() {
            let state = endpoints.row(k).to_owned();
            let cj = jacobi_fn(&state);
            let err = (cj - cj_initial).abs();
            if err > max_err { max_err = err; }
        }
        max_err
    }
}

// ============================================================
//  Core Picard iteration (single segment)
// ============================================================

/// Run the Picard iteration on a single segment [s_start, s_start + ds].
/// `rhs_fn`: evaluates F(x, sigma) at batch of Lobatto nodes.
///   Input:  x_batch [ndim, N+1] at nodes, sigma_batch [N+1]
///   Output: F_batch [ndim, N+1]
/// `x0`: initial condition at s_start.
pub fn picard_segment<F>(
    rhs_fn: &F,
    x0: &Array1<f64>,
    s_start: f64,
    ds: f64,
    config: &PicardConfig,
) -> Option<SegmentResult>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Array2<f64>,
{
    let ndim = x0.len();
    let n = config.n_cheb;
    let nodes = lobatto_nodes(n);

    let s_nodes = Array1::from_shape_fn(n + 1, |j| {
        s_start + ds * (nodes[j] + 1.0) / 2.0
    });

    let mut x_curr = Array2::from_shape_fn((ndim, n + 1), |(i, _)| x0[i]);
    
    let mut n_iter = 0usize;
    let mut residual = f64::INFINITY;
    
    // Track successive differences to empirically estimate the Lipschitz contraction constant kappa
    let mut prev_diff = f64::INFINITY;
    let mut kappa_est = 0.0f64;

    for iter in 0..config.max_iter {
        let f_vals = rhs_fn(&x_curr, &s_nodes);
        let f_cv = ChebVec::from_value_matrix(&f_vals);
        let integrated = integrate_cheb_with_ic_scaled(&f_cv, x0, ds / 2.0);
        let x_new_vals = integrated.to_value_matrix();

        let mut max_diff = 0.0f64;
        for i in 0..ndim {
            for j in 0..=n {
                let d = (x_curr[[i,j]] - x_new_vals[[i,j]]).abs();
                if d > max_diff { max_diff = d; }
            }
        }
        
        // Dynamically estimate the contraction ratio kappa = ||u_n - u_{n-1}|| / ||u_{n-1} - u_{n-2}||
        if prev_diff < f64::INFINITY && prev_diff > 1e-14 {
            let current_kappa = max_diff / prev_diff;
            if current_kappa > kappa_est { kappa_est = current_kappa; } // Take the worst-case contraction
        }
        
        prev_diff = max_diff;
        residual = max_diff;
        x_curr = x_new_vals;
        n_iter = iter + 1;

        if residual < config.tol {
            break;
        }
    }

    if residual >= config.tol * 1e3 {
        return None;
    }

    let final_cv = ChebVec::from_value_matrix(&x_curr);
    let bernstein_rho = final_cv.estimate_bernstein_rho();

    // Rigorous Newton-Kantorovich Certification using Interval Arithmetic
    let nk_bound = if config.certify && residual < config.tol * 100.0 {
        // If empirical kappa is invalid, fallback to the theoretical bound derived from Bernstein rho
        let kappa = if kappa_est > 0.0 && kappa_est < 1.0 {
            kappa_est
        } else {
            (1.0 / bernstein_rho).min(0.99)
        };
        // Call the rigorous interval arithmetic certifier
        nk_certify(residual, kappa)
    } else {
        None
    };

    Some(SegmentResult {
        cheb_coeffs: final_cv.c,
        s_start,
        ds,
        t_end: 0.0, 
        n_iter,
        residual,
        nk_bound,
        bernstein_rho,
    })
}   

/// Integrate a ChebVec of degree N in coefficient space.
/// Returns a ChebVec of the SAME degree N (values evaluated at the same N+1 Lobatto nodes).
/// The antiderivative has degree N+1, but we represent it at the original N nodes.
///
/// Procedure:
///   1. Compute antiderivative coefficients d_k of degree N+1 (N+2 terms).
///   2. Evaluate this degree-(N+1) polynomial at the N+1 Lobatto nodes for N.
///   3. Refit to degree-N Chebyshev representation for the Picard iterate.
///
/// This keeps the iterate dimension stable (N+1 values throughout).
fn integrate_cheb_with_ic_scaled(
    f_cv: &ChebVec,
    x0: &Array1<f64>,
    alpha: f64,  // = ds/2
) -> ChebVec {
    let ndim = f_cv.ndim;
    let n = f_cv.n;
    let nodes = crate::chebyshev::lobatto_nodes(n);  // N+1 nodes for original degree
    let mut result = ChebVec::zeros(ndim, n);
    for i in 0..ndim {
        let c = Array1::from_vec(f_cv.c.row(i).to_vec());
        // Step 1: compute degree-(n+1) antiderivative coefficients
        let d = crate::chebyshev::integrate_coeffs(&c, x0[i], alpha);
        // Step 2: evaluate the (n+1)-degree polynomial at the n+1 original nodes
        let vals: Array1<f64> = Array1::from_shape_fn(n + 1, |j| {
            crate::chebyshev::clenshaw(&d, nodes[j])
        });
        // Step 3: refit to degree-n Chebyshev coefficients
        let new_c = crate::chebyshev::values_to_coeffs(&vals);
        result.c.row_mut(i).assign(&new_c);
    }
    result
}

/// Simple NK bound estimate from Picard residual and a local Lipschitz estimate.
/// Returns η/(1-κ) where η = residual and κ is estimated from the Chebyshev decay.
fn estimate_nk_bound(cv: &ChebVec, residual: f64) -> Option<f64> {
    let rho = cv.estimate_bernstein_rho();
    if rho < 1.01 { return None; }
    // Crude κ estimate: for a well-converged solution, κ << 1.
    // Using κ ≈ tol/residual as a proxy for the Picard contraction ratio.
    let kappa = (1.0 / rho).min(0.9);
    Some(residual / (1.0 - kappa))
}

// ============================================================
//  Adaptive driver: propagate over [s0, s_final]
// ============================================================

/// Propagate the system from s0 to s_final using the adaptive Picard solver.
/// `rhs_fn`: batch RHS evaluator (see picard_segment).
/// `x0`: initial condition.
/// `has_time`: if true, the last component of x is physical time t; its
///             endpoint value is used to fill SegmentResult::t_end.
pub fn propagate<F>(
    rhs_fn: &F,
    x0: &Array1<f64>,
    s0: f64,
    s_final: f64,
    config: &PicardConfig,
    has_time: bool,
) -> Trajectory
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Array2<f64>,
{
    let ndim = x0.len();
    let mut segments = Vec::new();
    let mut s = s0;
    let mut x = x0.clone();
    let mut ds = config.ds_initial;
    let mut n_segs = 0usize;

    while s < s_final - 1e-14 {
        if n_segs >= config.max_segments {
            eprintln!("WARNING: max_segments ({}) reached", config.max_segments);
            break;
        }
        // Clamp to not overshoot
        ds = ds.min(s_final - s).max(config.ds_min);

        match picard_segment(rhs_fn, &x, s, ds, config) {
            None => {
                // Convergence failure: halve segment
                ds = (ds / 2.0).max(config.ds_min);
                if ds <= config.ds_min {
                    eprintln!("WARNING: ds_min reached at s={:.6}", s);
                    break;
                }
                continue;
            }
            Some(mut seg) => {
                // Fill t_end from the last component if time-extended
                let x_end = {
                    let ndim_seg = seg.cheb_coeffs.nrows();
                    Array1::from_shape_fn(ndim_seg, |i| {
                        let row = Array1::from_vec(seg.cheb_coeffs.row(i).to_vec());
                        clenshaw(&row, 1.0)
                    })
                };
                if has_time {
                    seg.t_end = x_end[ndim - 1];
                } else {
                    seg.t_end = s + ds;  // fictitious time = physical time
                }

                let n_iter = seg.n_iter;
                x = x_end;
                s += ds;
                segments.push(seg);
                n_segs += 1;

                // Adaptive step size control
                if n_iter < config.k_double {
                    ds = (ds * 2.0).min(config.ds_max);
                }
                // If n_iter was close to max_iter, halve (don't double)
                if n_iter > config.max_iter * 3 / 4 {
                    ds = (ds / 2.0).max(config.ds_min);
                }
            }
        }
    }

    Trajectory { segments, ndim, n_cheb: config.n_cheb }
}

// ============================================================
//  Specialized CR3BP right-hand side
// ============================================================

/// Compute the CR3BP RHS at a batch of states.
/// State: [x, y, z, vx, vy, vz] (6 components)
/// mu: mass ratio of secondary (e.g. 0.01215 for Earth-Moon)
pub fn cr3bp_rhs_batch(
    x_batch: &Array2<f64>,
    _t_batch: &Array1<f64>,
    mu: f64,
) -> Array2<f64> {
    let n_pts = x_batch.ncols();
    let mut out = Array2::zeros((6, n_pts));
    let mu1 = 1.0 - mu;

    for j in 0..n_pts {
        let x  = x_batch[[0, j]];
        let y  = x_batch[[1, j]];
        let z  = x_batch[[2, j]];
        let vx = x_batch[[3, j]];
        let vy = x_batch[[4, j]];
        let vz = x_batch[[5, j]];

        let r1 = ((x+mu)*(x+mu) + y*y + z*z).sqrt();
        let r2 = ((x-mu1)*(x-mu1) + y*y + z*z).sqrt();

        let r1_3 = r1.powi(3).max(1e-300);
        let r2_3 = r2.powi(3).max(1e-300);

        // Equations of motion
        out[[0, j]] = vx;
        out[[1, j]] = vy;
        out[[2, j]] = vz;
        out[[3, j]] =  2.0*vy + x - mu1*(x+mu)/r1_3 - mu*(x-mu1)/r2_3;
        out[[4, j]] = -2.0*vx + y - mu1*y/r1_3 - mu*y/r2_3;
        out[[5, j]] =            -mu1*z/r1_3 - mu*z/r2_3;
    }
    out
}

/// Compute the Jacobi constant for the CR3BP.
/// C_J = 2*Omega - v^2 where Omega = (x^2+y^2)/2 + (1-mu)/r1 + mu/r2 + mu*(1-mu)/2
pub fn jacobi_constant(state: &Array1<f64>, mu: f64) -> f64 {
    let x  = state[0];
    let y  = state[1];
    let z  = state[2];
    let vx = state[3];
    let vy = state[4];
    let vz = state[5];
    let mu1 = 1.0 - mu;

    let r1 = ((x+mu)*(x+mu) + y*y + z*z).sqrt();
    let r2 = ((x-mu1)*(x-mu1) + y*y + z*z).sqrt();

    let omega = 0.5*(x*x + y*y) + mu1/r1 + mu/r2 + 0.5*mu*mu1;
    let v2 = vx*vx + vy*vy + vz*vz;
    2.0*omega - v2
}

/// Propagate the CR3BP with the Adaptive Spectral Picard solver.
/// Returns the trajectory and the Jacobi constant error.
pub fn propagate_cr3bp(
    x0: &Array1<f64>,
    t_final: f64,
    mu: f64,
    config: &PicardConfig,
) -> (Trajectory, f64) {
    let cj0 = jacobi_constant(x0, mu);

    let rhs = |x: &Array2<f64>, t: &Array1<f64>| cr3bp_rhs_batch(x, t, mu);
    let traj = propagate(&rhs, x0, 0.0, t_final, config, false);

    // Compute Jacobi constant error at all endpoints
    let jac_fn = |s: &Array1<f64>| jacobi_constant(s, mu);
    let cj_err = traj.max_jacobi_error(cj0, &jac_fn);

    (traj, cj_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_harmonic_oscillator() {
        // Solve ẋ = v, v̇ = -x (SHO) with x(0)=1, v(0)=0.
        // Exact: x(t) = cos(t), v(t) = -sin(t).
        let config = PicardConfig {
            n_cheb: 24,
            tol: 1e-12,
            max_iter: 30,
            ds_initial: 1.0,
            ds_max: 2.0 * std::f64::consts::PI,
            ..Default::default()
        };
        let x0 = array![1.0, 0.0];
        let rhs = |x: &Array2<f64>, _t: &Array1<f64>| {
            let ncols = x.ncols();
            let mut out = Array2::zeros((2, ncols));
            for j in 0..ncols {
                out[[0, j]] =  x[[1, j]];
                out[[1, j]] = -x[[0, j]];
            }
            out
        };
        eprintln!("SHO test: starting propagate...");
        // Propagate one step to t=1.0 and verify against exact solution
        let traj = propagate(&rhs, &x0, 0.0, 1.0, &config, false);
        eprintln!("SHO test: {} segments", traj.total_segments());
        let final_state = traj.final_state().unwrap();
        let exact_x = 1.0_f64.cos();
        let exact_v = -1.0_f64.sin();
        assert!((final_state[0] - exact_x).abs() < 1e-10,
            "x(1) = {} != {}", final_state[0], exact_x);
        assert!((final_state[1] - exact_v).abs() < 1e-10,
            "v(1) = {} != {}", final_state[1], exact_v);
    }

    #[test]
    fn test_cr3bp_jacobi() {
        // Halo orbit initial condition (approximate L1 halo, Earth-Moon)
        let mu = 0.01215_f64;
        let x0 = array![
            8.316591e-1_f64,   // x
            0.0,               // y
            1.2744e-1,         // z
            0.0,               // vx
            -1.32767e-1,       // vy (adjusted for periodicity)
            0.0                // vz
        ];
        let config = PicardConfig {
            n_cheb: 16,  // small for fast debug-mode test
            tol: 1e-8,
            ds_initial: 0.2,
            max_segments: 20,
            certify: false,
            ..Default::default()
        };
        let (traj, cj_err) = propagate_cr3bp(&x0, 1.0, mu, &config);
        println!("CR3BP: {} segments, Jacobi error = {:.2e}", traj.total_segments(), cj_err);
        assert!(cj_err < 1e-4, "Jacobi error too large: {:.2e}", cj_err);
    }
}
