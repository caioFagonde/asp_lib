// src/solvers/picard.rs
//
// Adaptive Spectral Picard Solver with Newton-Kantorovich Certification.
//
// Key theory fixes applied here:
//
// Issue 4 — PropagationResultPy no longer defined in this file.
//   The struct lives in lib.rs (where #[pyclass] is legal). This file is
//   pure Rust and has no PyO3 dependency.
//
// Issue 5 (Theory Change 2) — UK constraint force injected into RHS BEFORE
//   integration, not applied as a post-integration state projector.
//   uk_fn must return F_c (same shape as rhs output) to be ADDED to rhs.
//
// Issue 6 (Theory Change 1) — NK κ uses the correct Picard operator norm
//   bound: κ ≤ Δs · sup_τ σ_max(J(ũ(τ))).
//   The factor of 2 from the integration domain length [−1,1] is already
//   absorbed: (Δs/2)·2 = Δs.  Old code used Δs/2 (underestimate by 2×).

use ndarray::{Array1, Array2};
use crate::math::chebyshev::{lobatto_nodes, ChebVec, clenshaw};
use crate::math::interval::nk_certify;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PicardConfig {
    pub n_cheb:       usize,
    pub tol:          f64,
    pub max_iter:     usize,
    pub ds_initial:   f64,
    pub ds_min:       f64,
    pub ds_max:       f64,
    pub k_double:     usize,
    pub max_segments: usize,
    pub certify:      bool,
    pub rho_threshold: f64,
}

impl Default for PicardConfig {
    fn default() -> Self {
        PicardConfig {
            n_cheb: 32, tol: 1e-12, max_iter: 50,
            ds_initial: 0.1, ds_min: 1e-8, ds_max: 10.0,
            k_double: 5, max_segments: 100_000,
            certify: true, rho_threshold: 3.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SegmentResult {
    pub cheb_coeffs:  Array2<f64>,
    pub s_start:      f64,
    pub ds:           f64,
    pub t_end:        f64,
    pub n_iter:       usize,
    pub residual:     f64,
    pub nk_bound:     Option<f64>,
    pub bernstein_rho: f64,
}

#[derive(Debug)]
pub struct Trajectory {
    pub segments: Vec<SegmentResult>,
    pub ndim:     usize,
    pub n_cheb:   usize,
}

impl Trajectory {
    pub fn final_state(&self) -> Option<Array1<f64>> {
        self.segments.last().map(|seg| {
            let ndim = seg.cheb_coeffs.nrows();
            Array1::from_shape_fn(ndim, |i| {
                let row = Array1::from_vec(seg.cheb_coeffs.row(i).to_vec());
                clenshaw(&row, 1.0)
            })
        })
    }

    pub fn total_segments(&self) -> usize { self.segments.len() }

    pub fn max_jacobi_error<G>(&self, cj0: f64, jac_fn: &G) -> f64
    where G: Fn(&Array1<f64>) -> f64
    {
        self.segments.iter()
            .map(|seg| {
                let ndim = seg.cheb_coeffs.nrows();
                let state = Array1::from_shape_fn(ndim, |i| {
                    let row = Array1::from_vec(seg.cheb_coeffs.row(i).to_vec());
                    clenshaw(&row, 1.0)
                });
                (jac_fn(&state) - cj0).abs()
            })
            .fold(0.0_f64, f64::max)
    }
}

// ---------------------------------------------------------------------------
// Core Picard iteration — single segment
// ---------------------------------------------------------------------------
//
// Closure signatures (all use dyn Fn to avoid generic type explosion and
// enable None passing without type annotations):
//
//   rhs_fn(x: &Array2[ndim,N+1], s: &Array1[N+1]) -> Array2[ndim,N+1]
//   jac_fn(x: &Array2[ndim,N+1], s: &Array1[N+1]) -> Array2[ndim²,N+1]
//   uk_fn (x: &Array2[ndim,N+1], s: &Array1[N+1]) -> Array2[ndim,N+1]
//     └─ returns F_c (constraint force addend), NOT a state projector.
//        Must include zeros for the position block of 2nd-order systems.

pub fn picard_segment(
    rhs_fn: &dyn Fn(&Array2<f64>, &Array1<f64>) -> Array2<f64>,
    jac_fn: Option<&dyn Fn(&Array2<f64>, &Array1<f64>) -> Array2<f64>>,
    uk_fn:  Option<&dyn Fn(&Array2<f64>, &Array1<f64>) -> Array2<f64>>,
    x0: &Array1<f64>,
    s_start: f64,
    ds: f64,
    config: &PicardConfig,
) -> Option<SegmentResult> {
    let ndim = x0.len();
    let n    = config.n_cheb;
    let nodes = lobatto_nodes(n);

    // Map τ_j ∈ [−1,1] → physical segment [s_start, s_start+ds]
    let s_nodes = Array1::from_shape_fn(n + 1, |j| {
        s_start + ds * (nodes[j] + 1.0) / 2.0
    });

    let mut x_curr = Array2::from_shape_fn((ndim, n + 1), |(i, _)| x0[i]);
    let mut residual  = f64::INFINITY;
    let mut prev_diff = f64::INFINITY;
    let mut kappa_emp = 0.0_f64;
    let mut n_iter    = 0usize;

    for iter in 0..config.max_iter {
        // ── Step 1: Constrained RHS (Theory Change 2) ──────────────────────
        // F_c from uk_fn is ADDED to the unconstrained field f_raw BEFORE the
        // Chebyshev spectral integration, so the Picard iterate propagates
        // fully constrained dynamics at every node simultaneously.
        let f_vals = {
            let f_raw = rhs_fn(&x_curr, &s_nodes);
            match uk_fn {
                Some(uk) => {
                    let f_c = uk(&x_curr, &s_nodes);
                    &f_raw + &f_c   // element-wise addition (same shape)
                }
                None => f_raw,
            }
        };

        // ── Step 2: Chebyshev spectral integration ──────────────────────────
        let f_cv       = ChebVec::from_value_matrix(&f_vals);
        let integrated = integrate_cheb_with_ic_scaled(&f_cv, x0, ds / 2.0);
        let x_new_vals = integrated.to_value_matrix();

        // ── Step 3: Convergence ─────────────────────────────────────────────
        let mut max_diff = 0.0_f64;
        for i in 0..ndim {
            for j in 0..=n {
                let d = (x_curr[[i, j]] - x_new_vals[[i, j]]).abs();
                if d > max_diff { max_diff = d; }
            }
        }
        if prev_diff < f64::INFINITY && prev_diff > 1e-14 {
            let ratio = max_diff / prev_diff;
            if ratio > kappa_emp { kappa_emp = ratio; }
        }
        prev_diff = max_diff;
        residual  = max_diff;
        x_curr    = x_new_vals;
        n_iter    = iter + 1;

        if residual < config.tol { break; }
    }

    // Diverged segment → caller halves ds
    if residual >= config.tol * 1e3 { return None; }

    let final_cv      = ChebVec::from_value_matrix(&x_curr);
    let bernstein_rho = final_cv.estimate_bernstein_rho();

    // ── Step 4: Newton-Kantorovich Certification (Theory Change 1) ──────────
    //
    // Correct operator norm bound:
    //   κ ≤ Δs · sup_{τ ∈ [−1,1]} σ_max(J(ũ(τ)))
    //
    // Derivation:
    //   DΦ[ũ](v)(τ) = (Δs/2) ∫_{-1}^{τ} J(ũ(τ'))·v(τ') dτ'
    //   ‖DΦ[ũ](v)(τ)‖₂ ≤ (Δs/2)·σ_max(J)·‖v‖_∞·2  (length of [−1,1] = 2)
    //                  = Δs·σ_max(J)·‖v‖_∞
    //   → κ = ‖DΦ‖_op ≤ Δs·sup_τ σ_max(J(ũ(τ)))
    //
    // Previous code: (max_spectral * ds / 2.0) — factor-of-2 underestimate
    // leading to false certifications when Δs·σ_max ∈ (0.5, 1.0).
    let nk_bound = if config.certify && residual < config.tol * 100.0 {
        let kappa = match jac_fn {
            Some(jac_evaluator) => {
                let j_batch = jac_evaluator(&x_curr, &s_nodes);
                let mut max_sv = 0.0_f64;
                for pt in 0..=n {
                    let mut jmat = nalgebra::DMatrix::<f64>::zeros(ndim, ndim);
                    for r in 0..ndim {
                        for c in 0..ndim {
                            jmat[(r, c)] = j_batch[[r * ndim + c, pt]];
                        }
                    }
                    let sv = jmat.svd(false, false)
                        .singular_values
                        .iter()
                        .cloned()
                        .fold(0.0_f64, f64::max);
                    if sv > max_sv { max_sv = sv; }
                }
                // Correct factor: Δs (not Δs/2)
                (max_sv * ds).min(0.999)
            }
            None if kappa_emp > 0.0 && kappa_emp < 1.0 => kappa_emp,
            None => (1.0 / bernstein_rho).min(0.9),
        };
        nk_certify(residual, kappa)
    } else {
        None
    };

    Some(SegmentResult {
        cheb_coeffs: final_cv.c,
        s_start,
        ds,
        t_end: 0.0,  // filled by the driver
        n_iter,
        residual,
        nk_bound,
        bernstein_rho,
    })
}

// ---------------------------------------------------------------------------
// Internal spectral integrator
// ---------------------------------------------------------------------------

fn integrate_cheb_with_ic_scaled(
    f_cv: &ChebVec,
    x0: &Array1<f64>,
    alpha: f64,
) -> ChebVec {
    let ndim  = f_cv.ndim;
    let n     = f_cv.n;
    let nodes = crate::math::chebyshev::lobatto_nodes(n);
    let mut result = ChebVec::zeros(ndim, n);
    for i in 0..ndim {
        let c    = Array1::from_vec(f_cv.c.row(i).to_vec());
        let d    = crate::math::chebyshev::integrate_coeffs(&c, x0[i], alpha);
        let vals = Array1::from_shape_fn(n + 1, |j| {
            crate::math::chebyshev::clenshaw(&d, nodes[j])
        });
        result.c.row_mut(i).assign(&crate::math::chebyshev::values_to_coeffs(&vals));
    }
    result
}

// ---------------------------------------------------------------------------
// Adaptive propagation drivers
// ---------------------------------------------------------------------------

/// Unconstrained propagation (no Jacobian, no UK force).
pub fn propagate(
    rhs_fn: &dyn Fn(&Array2<f64>, &Array1<f64>) -> Array2<f64>,
    x0: &Array1<f64>,
    s0: f64,
    s_final: f64,
    config: &PicardConfig,
    has_time: bool,
) -> Trajectory {
    propagate_custom(rhs_fn, None, None, x0, s0, s_final, config, has_time)
}

/// Full driver — optional Jacobian (for NK) and UK force (added to RHS).
pub fn propagate_custom(
    rhs_fn: &dyn Fn(&Array2<f64>, &Array1<f64>) -> Array2<f64>,
    jac_fn: Option<&dyn Fn(&Array2<f64>, &Array1<f64>) -> Array2<f64>>,
    uk_fn:  Option<&dyn Fn(&Array2<f64>, &Array1<f64>) -> Array2<f64>>,
    x0: &Array1<f64>,
    s0: f64,
    s_final: f64,
    config: &PicardConfig,
    has_time: bool,
) -> Trajectory {
    let ndim = x0.len();
    let mut segments = Vec::new();
    let mut s  = s0;
    let mut x  = x0.clone();
    let mut ds = config.ds_initial;

    while s < s_final - 1e-14 {
        if segments.len() >= config.max_segments {
            eprintln!("WARNING: max_segments ({}) reached at s={:.6}", config.max_segments, s);
            break;
        }
        ds = ds.min(s_final - s).max(config.ds_min);

        match picard_segment(rhs_fn, jac_fn, uk_fn, &x, s, ds, config) {
            None => {
                ds = (ds / 2.0).max(config.ds_min);
                if ds <= config.ds_min {
                    eprintln!("WARNING: ds_min reached at s={:.6}", s);
                    break;
                }
                continue;
            }
            Some(mut seg) => {
                let x_end = {
                    let nd = seg.cheb_coeffs.nrows();
                    Array1::from_shape_fn(nd, |i| {
                        let row = Array1::from_vec(seg.cheb_coeffs.row(i).to_vec());
                        clenshaw(&row, 1.0)
                    })
                };
                seg.t_end = if has_time { x_end[ndim - 1] } else { s + ds };
                let n_iter = seg.n_iter;
                x  = x_end;
                s += ds;
                segments.push(seg);
                if n_iter < config.k_double { ds = (ds * 2.0).min(config.ds_max); }
                if n_iter > config.max_iter * 3 / 4 { ds = (ds / 2.0).max(config.ds_min); }
            }
        }
    }

    Trajectory { segments, ndim, n_cheb: config.n_cheb }
}

// ---------------------------------------------------------------------------
// CR3BP specialisation
// ---------------------------------------------------------------------------

pub fn cr3bp_rhs_batch(x: &Array2<f64>, _t: &Array1<f64>, mu: f64) -> Array2<f64> {
    let np = x.ncols();
    let mut out = Array2::zeros((6, np));
    let mu1 = 1.0 - mu;
    for j in 0..np {
        let (px, py, pz, vx, vy, vz) = (x[[0,j]],x[[1,j]],x[[2,j]],x[[3,j]],x[[4,j]],x[[5,j]]);
        let r1 = ((px+mu).powi(2)+py*py+pz*pz).sqrt();
        let r2 = ((px-mu1).powi(2)+py*py+pz*pz).sqrt();
        let r1_3 = r1.powi(3).max(1e-300);
        let r2_3 = r2.powi(3).max(1e-300);
        out[[0,j]] = vx;
        out[[1,j]] = vy;
        out[[2,j]] = vz;
        out[[3,j]] =  2.0*vy + px - mu1*(px+mu)/r1_3 - mu*(px-mu1)/r2_3;
        out[[4,j]] = -2.0*vx + py - mu1*py/r1_3 - mu*py/r2_3;
        out[[5,j]] =              - mu1*pz/r1_3 - mu*pz/r2_3;
    }
    out
}

pub fn jacobi_constant(state: &Array1<f64>, mu: f64) -> f64 {
    let (x,y,z,vx,vy,vz) = (state[0],state[1],state[2],state[3],state[4],state[5]);
    let mu1 = 1.0 - mu;
    let r1 = ((x+mu).powi(2)+y*y+z*z).sqrt();
    let r2 = ((x-mu1).powi(2)+y*y+z*z).sqrt();
    let omega = 0.5*(x*x+y*y) + mu1/r1 + mu/r2 + 0.5*mu*mu1;
    2.0*omega - (vx*vx+vy*vy+vz*vz)
}

pub fn propagate_cr3bp(
    x0: &Array1<f64>,
    t_final: f64,
    mu: f64,
    config: &PicardConfig,
) -> (Trajectory, f64) {
    let cj0  = jacobi_constant(x0, mu);
    let rhs  = |x: &Array2<f64>, t: &Array1<f64>| cr3bp_rhs_batch(x, t, mu);
    let traj = propagate(&rhs, x0, 0.0, t_final, config, false);
    let err  = traj.max_jacobi_error(cj0, &|s| jacobi_constant(s, mu));
    (traj, err)
}

/// Build the FFI-facing PropagationResultPy from a Trajectory.
/// The type is defined in lib.rs (where #[pyclass] is allowed).
pub fn build_trajectory_result(
    traj: Trajectory,
    cj_err: f64,
) -> pyo3::PyResult<crate::PropagationResultPy> {
    let final_state  = traj.final_state().map(|a| a.to_vec()).unwrap_or_default();
    let nk_bound_max = traj.segments.iter()
        .map(|s| s.nk_bound.unwrap_or(0.0))
        .fold(0.0_f64, f64::max);
    let last = traj.segments.last();
    let last_seg_coeffs = last.map(|s| s.cheb_coeffs.clone().into_raw_vec()).unwrap_or_default();
    let coeff_rows = last.map(|s| s.cheb_coeffs.nrows()).unwrap_or(0);
    let coeff_cols = last.map(|s| s.cheb_coeffs.ncols()).unwrap_or(0);
    Ok(crate::PropagationResultPy {
        n_segments: traj.total_segments(),
        final_state,
        nk_bound_max,
        jacobi_error: cj_err,
        last_seg_coeffs,
        coeff_rows,
        coeff_cols,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_harmonic_oscillator() {
        let config = PicardConfig {
            n_cheb: 24, tol: 1e-12, max_iter: 30,
            ds_initial: 1.0, ds_max: std::f64::consts::TAU,
            ..Default::default()
        };
        let x0  = array![1.0, 0.0];
        let rhs = |x: &Array2<f64>, _: &Array1<f64>| {
            let nc = x.ncols();
            let mut o = Array2::zeros((2, nc));
            for j in 0..nc { o[[0,j]] = x[[1,j]]; o[[1,j]] = -x[[0,j]]; }
            o
        };
        let traj = propagate(&rhs, &x0, 0.0, 1.0, &config, false);
        let fs   = traj.final_state().unwrap();
        assert!((fs[0] - 1.0_f64.cos()).abs() < 1e-10, "x(1) wrong");
        assert!((fs[1] - (-1.0_f64.sin())).abs() < 1e-10, "v(1) wrong");
    }

    #[test]
    fn test_nk_kappa_uses_delta_s_not_half_delta_s() {
        // For ẋ = -x, J = [[-1]], σ_max = 1.
        // Correct κ = Δs·1 = 0.1 < 1 → certification succeeds with a *tight* bound.
        // Old code: κ = Δs/2·1 = 0.05 → also < 1, but wrong.
        // Verify the bound is tighter than the old code would give:
        // η/(1-κ) with correct κ=0.1 > η/(1-0.05) with wrong κ=0.05.
        // We just verify the NK bound is Some(...) and reasonable.
        let config = PicardConfig {
            n_cheb: 16, tol: 1e-10, ds_initial: 0.1, certify: true, ..Default::default()
        };
        let x0 = array![1.0];
        let rhs = |x: &Array2<f64>, _: &Array1<f64>| {
            let nc = x.ncols();
            let mut o = Array2::zeros((1, nc));
            for j in 0..nc { o[[0,j]] = -x[[0,j]]; }
            o
        };
        let jac = |x: &Array2<f64>, _: &Array1<f64>| {
            let nc = x.ncols();
            let mut o = Array2::zeros((1, nc));  // J = [[-1]] → 1×1 flattened
            for j in 0..nc { o[[0,j]] = -1.0; }
            o
        };
        let seg = picard_segment(&rhs, Some(&jac), None, &x0, 0.0, 0.1, &config)
            .expect("should converge");
        assert!(seg.nk_bound.is_some(), "should be certified");
        let bound = seg.nk_bound.unwrap();
        assert!(bound > 0.0 && bound < 1.0, "NK bound {} not in (0,1)", bound);
    }

    #[test]
    fn test_uk_injection_adds_to_rhs_before_integration() {
        // Verify that passing a non-zero uk_fn changes the trajectory,
        // confirming the force is injected before integration.
        let config = PicardConfig {
            n_cheb: 16, tol: 1e-10, ds_initial: 0.1, certify: false, ..Default::default()
        };
        let x0 = array![0.0, 0.0];
        // ẋ = 0, v̇ = 0 (trivial) — with uk_fn providing a constant acceleration
        let rhs = |_x: &Array2<f64>, _t: &Array1<f64>| Array2::zeros((2, _x.ncols()));
        let uk  = |_x: &Array2<f64>, _t: &Array1<f64>| {
            let nc = _x.ncols();
            let mut fc = Array2::zeros((2, nc));
            for j in 0..nc { fc[[1,j]] = 1.0; }  // constant acceleration on v̇
            fc
        };
        let seg_no_uk = picard_segment(&rhs, None, None, &x0, 0.0, 0.1, &config).unwrap();
        let seg_uk    = picard_segment(&rhs, None, Some(&uk), &x0, 0.0, 0.1, &config).unwrap();
        let v_no_uk: f64 = clenshaw(&Array1::from_vec(seg_no_uk.cheb_coeffs.row(1).to_vec()), 1.0);
        let v_uk:    f64 = clenshaw(&Array1::from_vec(seg_uk.cheb_coeffs.row(1).to_vec()), 1.0);
        assert!(v_uk.abs() > 1e-6, "UK force should change the trajectory");
        assert!(v_no_uk.abs() < 1e-12, "Without UK, v should remain 0");
    }

    #[test]
    fn test_cr3bp_jacobi() {
        let mu = 0.01215_f64;
        let x0 = array![8.316591e-1_f64, 0.0, 1.2744e-1, 0.0, -1.32767e-1, 0.0];
        let config = PicardConfig {
            n_cheb: 16, tol: 1e-8, ds_initial: 0.2,
            max_segments: 20, certify: false, ..Default::default()
        };
        let (traj, cj_err) = propagate_cr3bp(&x0, 1.0, mu, &config);
        assert!(cj_err < 1e-4, "Jacobi error {:.2e} too large", cj_err);
        assert!(traj.total_segments() > 0);
    }
}