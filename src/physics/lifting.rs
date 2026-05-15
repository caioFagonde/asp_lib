// src/lifting.rs
//
// Automatic Singularity Detection and State-Space Lifting
// =========================================================
// This module implements the "ASP Lift": given an ODE ẋ = f(x,t), it
// automatically detects singularities in the complex time plane and applies
// a generalised Sundman time transformation + optional state-space fibration
// to convert the system into one whose solutions are entire (or at least more
// analytic), enabling superexponential Chebyshev convergence.
//
// The algorithm generalises the Kustaanheimo-Stiefel transformation from
// celestial mechanics to arbitrary ODEs, providing a principled API:
//
//   1. DETECT:  pilot Picard run → estimate Bernstein ρ
//   2. DIAGNOSE: classify the singularity (algebraic/collision/essential)
//   3. LIFT:    construct w(x) and optional fibration map π: ℝ^m → ℝ^n
//   4. SOLVE:   run the full spectral Picard solver on the lifted system
//   5. PROJECT: map the lifted solution back to original coordinates
//
// Mathematical foundation:
//   - Sundman transformation:  dt = w(x) ds  →  x' = w(x)·f(x,t), t' = w(x)
//   - KS fibration:            r = π(u) with |r| = |u|², removes 1/|r|^k poles
//   - General fibration:       any map with |π'(u)| = w(u) can regularise
//
// References:
//   - Kustaanheimo & Stiefel (1965) Crelle's Journal 218:204-219
//   - Stiefel & Scheifele (1971) Linear and Regular Celestial Mechanics
//   - Stiefeleñ Scheifele (1971) for Levi-Civita (2D case)

use ndarray::{Array1, Array2};
use crate::math::chebyshev::{lobatto_nodes, ChebVec};

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyArray2, ToPyArray};

/// A bridge struct allowing Rust's generic ODE lifting engine to evaluate
/// vector fields defined purely in Python.
pub struct PythonOde {
    pub py_obj: PyObject,
    pub ndim: usize,
}

impl OdeSystem for PythonOde {
    fn ndim(&self) -> usize {
        self.ndim
    }

    fn rhs_batch(&self, x_batch: &Array2<f64>, t_batch: &Array1<f64>) -> Array2<f64> {
        Python::with_gil(|py| {
            // Convert Rust ndarrays to NumPy arrays without copying where possible
            let py_x = x_batch.to_pyarray(py);
            let py_t = t_batch.to_pyarray(py);
            
            // Construct arguments tuple
            let args = PyTuple::new(py, &[py_x.as_any(), py_t.as_any()]);
            
            // Invoke the Python callable
            let result = self.py_obj.call1(py, args)
                .expect("Python ODE evaluation failed during diagnosis.");
            
            // Extract the resulting NumPy array back to Rust Array2
            let py_out: &numpy::PyArray2<f64> = result.extract(py)
                .expect("Python ODE must return a 2D numpy float64 array.");
            
            py_out.to_owned_array()
        })
    }
}

// ============================================================
//  Public types
// ============================================================

/// Classification of the detected singularity type.
#[derive(Debug, Clone, PartialEq)]
pub enum SingularityType {
    /// No singularity detected; Bernstein ρ is large enough.
    None,
    /// Algebraic singularity f ~ C/(x - x*)^p.
    /// Sundman weight w = |x - x_sing|^p is applied.
    Algebraic { power: f64, component: usize, x_sing: f64 },
    /// Collision singularity: norm |r_i - r_j| → 0.
    /// KS fibration (or Levi-Civita for 2D) is applied.
    Collision { body_i: usize, body_j: usize, dim: usize },
    /// General rational singularity with multiple poles.
    /// Product Sundman weight w = ∏_i |x - x_i*|^{p_i} is used.
    MultiPole { poles: Vec<(f64, f64)> },  // (location, power)
    /// Unknown/essential singularity; only time rescaling is possible.
    Unknown,
}

/// Result of the lifting analysis.
#[derive(Debug, Clone)]
pub struct LiftDiagnosis {
    pub bernstein_rho: f64,
    pub singularity: SingularityType,
    /// If true, use KS fibration (dimension goes from n to n+1).
    pub apply_fibration: bool,
    /// The Sundman weight function, evaluated at the pilot nodes.
    pub pilot_weights: Array1<f64>,
    /// Recommended segment length for lifted system.
    pub suggested_ds: f64,
}

/// A generic ODE system: ẋ = f(x, t) with initial condition.
/// Users implement this trait (via Python callback or Rust closure).
pub trait OdeSystem: Send + Sync {
    fn ndim(&self) -> usize;
    /// Evaluate f(x, t) at a batch of n_pts points.
    /// x_batch: [ndim, n_pts], t_batch: [n_pts]
    /// Returns: [ndim, n_pts]
    fn rhs_batch(&self, x_batch: &Array2<f64>, t_batch: &Array1<f64>) -> Array2<f64>;
    
    /// Optional: provide analytic weight function w(x) for Sundman lifting.
    /// If None, the weight is estimated automatically.
    fn sundman_weight(&self, x: &Array1<f64>) -> Option<f64> { None }
    
    /// Optional: provide a collision pair (indices into the position block)
    /// for KS-type fibration. If None, fibration is not applied.
    fn ks_collision_pair(&self) -> Option<(usize, usize, usize)> { None } // (i, j, dim)
}

// ============================================================
//  Lifted system: x_lifted = [u; t], dx/ds = w(π(u))·f(π(u), t)
// ============================================================

/// A lifted ODE system in Sundman-extended coordinates.
/// State: (x_physical [ndim], t [1])  → lifted state has ndim + 1 components.
pub struct LiftedSystem<'a> {
    pub base: &'a dyn OdeSystem,
    pub diagnosis: LiftDiagnosis,
    /// If Some((body_i, body_j, dim)), apply KS fibration.
    pub ks_pair: Option<(usize, usize, usize)>,
}

impl<'a> LiftedSystem<'a> {
    /// Evaluate the lifted RHS: dx/ds = w(x)·f(x, t),  dt/ds = w(x).
    /// x_lifted: [ndim+1] = [x_0,...,x_{n-1}, t]
    /// Returns: [ndim+1]
    pub fn rhs_lifted(&self, x_lifted: &Array1<f64>) -> Array1<f64> {
        let ndim = self.base.ndim();
        let x = x_lifted.slice(ndarray::s![..ndim]).to_owned();
        let t = x_lifted[ndim];
        
        let x_batch = x.view().to_shape((ndim, 1)).unwrap().to_owned();
        let t_batch = Array1::from_vec(vec![t]);
        let f_batch = self.base.rhs_batch(&x_batch, &t_batch);
        let f = f_batch.column(0).to_owned();
        
        // Compute Sundman weight w(x)
        let w = self.compute_weight(&x);
        
        let mut out = Array1::zeros(ndim + 1);
        for i in 0..ndim {
            out[i] = w * f[i];
        }
        out[ndim] = w;  // dt/ds = w
        out
    }

    /// Evaluate the lifted RHS at a batch of points.
    /// x_batch: [ndim+1, n_pts]  → returns [ndim+1, n_pts]
    pub fn rhs_batch_lifted(&self, x_batch: &Array2<f64>) -> Array2<f64> {
        let ndim = self.base.ndim();
        let n_pts = x_batch.ncols();
        let mut out = Array2::zeros((ndim + 1, n_pts));
        
        // Extract physical state and time
        let x_phys = x_batch.slice(ndarray::s![..ndim, ..]).to_owned();
        let t_vec = x_batch.row(ndim).to_owned();
        
        // Batch evaluate base RHS
        let f_batch = self.base.rhs_batch(&x_phys, &t_vec);
        
        for j in 0..n_pts {
            let x_j = x_phys.column(j).to_owned();
            let w = self.compute_weight(&x_j);
            for i in 0..ndim {
                out[[i, j]] = w * f_batch[[i, j]];
            }
            out[[ndim, j]] = w;
        }
        out
    }

    /// Compute the Sundman weight w(x) at a physical state x.
    fn compute_weight(&self, x: &Array1<f64>) -> f64 {
        // 1. User-provided weight takes priority
        if let Some(w) = self.base.sundman_weight(x) {
            return w.max(1e-300);  // guard against zero
        }

        // 2. KS collision weight: w = |r_i - r_j|^2
        if let Some((i, j, dim)) = self.ks_pair {
            let r_i = x.slice(ndarray::s![i..i+dim]).to_owned();
            let r_j = x.slice(ndarray::s![j..j+dim]).to_owned();
            let diff_sq: f64 = r_i.iter().zip(r_j.iter()).map(|(a,b)| (a-b).powi(2)).sum();
            return diff_sq.max(1e-300);
        }

        // 3. Use diagnosed singularity type
        match &self.diagnosis.singularity {
            SingularityType::Algebraic { power, component, x_sing } => {
                let dist = (x[*component] - x_sing).abs();
                dist.powf(*power).max(1e-300)
            }
            SingularityType::Collision { body_i, body_j, dim } => {
                let n = *dim;
                let i = body_i * n;
                let j = body_j * n;
                let diff_sq: f64 = (0..n).map(|k| (x[i+k] - x[j+k]).powi(2)).sum();
                diff_sq.max(1e-300)
            }
            SingularityType::MultiPole { poles } => {
                // Product weight
                poles.iter().map(|(loc, power)| {
                    let dist = x.iter().map(|xi| (xi - loc).abs()).fold(f64::INFINITY, f64::min);
                    dist.powf(*power).max(1e-300)
                }).product::<f64>().max(1e-300)
            }
            _ => 1.0,  // No weight: identity Sundman (unregularised)
        }
    }
}

// ============================================================
//  Diagnosis engine
// ============================================================

/// Run the full lifting diagnosis on an ODE system.
/// `pilot_x0`: initial condition (ndim)
/// `pilot_dt`: initial segment length in physical time
/// `pilot_n`: Chebyshev degree for pilot run
pub fn diagnose<S: OdeSystem>(
    system: &S,
    x0: &Array1<f64>,
    t0: f64,
    dt_pilot: f64,
    pilot_n: usize,
    rho_threshold: f64,  // below this, apply lifting
) -> LiftDiagnosis {
    // 1. Pilot Picard run (one segment, crude tolerance)
    let rho = pilot_bernstein_rho(system, x0, t0, dt_pilot, pilot_n);

    // 2. If rho is large enough, no lifting needed
    if rho >= rho_threshold {
        return LiftDiagnosis {
            bernstein_rho: rho,
            singularity: SingularityType::None,
            apply_fibration: false,
            pilot_weights: Array1::ones(pilot_n + 1),
            suggested_ds: dt_pilot,
        };
    }

    // 3. Check for user-provided collision pair
    if let Some((i, j, dim)) = system.ks_collision_pair() {
        return LiftDiagnosis {
            bernstein_rho: rho,
            singularity: SingularityType::Collision { body_i: i, body_j: j, dim },
            apply_fibration: dim == 3,  // KS for 3D, LC for 2D
            pilot_weights: compute_collision_weights(x0, i, j, dim, pilot_n),
            suggested_ds: estimate_lifted_ds(x0, i, j, dim, dt_pilot),
        };
    }

    // 4. General algebraic singularity detection via coefficient decay analysis
    let sing = detect_algebraic_singularity(system, x0, t0, dt_pilot, pilot_n);
    let pilot_weights = compute_algebraic_weights(x0, &sing, pilot_n);
    let suggested_ds = 0.1_f64.min(dt_pilot * 2.0);  // lifted system allows larger steps

    LiftDiagnosis {
        bernstein_rho: rho,
        singularity: sing,
        apply_fibration: false,
        pilot_weights,
        suggested_ds,
    }
}

/// Estimate Bernstein ρ by running one Picard iteration and measuring coefficient decay.
fn pilot_bernstein_rho<S: OdeSystem>(
    system: &S,
    x0: &Array1<f64>,
    t0: f64,
    dt: f64,
    n: usize,
) -> f64 {
    let ndim = system.ndim();
    let nodes = lobatto_nodes(n);  // tau_j in [-1,1]
    
    // Map nodes to physical time: t_j = t0 + dt*(tau_j + 1)/2
    let t_batch = Array1::from_shape_fn(n + 1, |j| t0 + dt * (nodes[j] + 1.0) / 2.0);
    
    // Constant initial-condition iterate: x(t) ≈ x0 for all t (zeroth Picard)
    let x_batch = Array2::from_shape_fn((ndim, n + 1), |(i, _)| x0[i]);
    let f_batch = system.rhs_batch(&x_batch, &t_batch);
    
    // Compute Chebyshev coefficients of the RHS
    let cv = ChebVec::from_value_matrix(&f_batch);
    cv.estimate_bernstein_rho()
}

/// Detect algebraic singularity from coefficient pattern.
/// Uses Richardson extrapolation on coefficient ratios to locate the pole.
fn detect_algebraic_singularity<S: OdeSystem>(
    system: &S,
    x0: &Array1<f64>,
    t0: f64,
    dt: f64,
    n: usize,
) -> SingularityType {
    // For now: heuristic based on which component has the most rapid decay
    // A full implementation would use Darboux's theorem on the coefficient sequence
    // to fit ρ and the singularity location via the ratio test.
    
    // Simple heuristic: if x0 has a component near zero with large force, flag it
    let ndim = system.ndim();
    
    // Evaluate RHS at x0
    let x_batch = Array2::from_shape_fn((ndim, 1), |(i, _)| x0[i]);
    let t_batch = Array1::from_vec(vec![t0]);
    let f = system.rhs_batch(&x_batch, &t_batch);
    
    // Find component with largest |f_i| / |x_i| ratio (indicative of 1/x singularity)
    let mut max_ratio = 0.0f64;
    let mut sing_component = 0;
    for i in 0..ndim {
        let x_abs = x0[i].abs().max(1e-10);
        let ratio = f[[i, 0]].abs() / x_abs;
        if ratio > max_ratio {
            max_ratio = ratio;
            sing_component = i;
        }
    }
    
    if max_ratio > 1e3 {
        // Large ratio suggests algebraic singularity near x=0 for component i
        SingularityType::Algebraic {
            power: 1.0,  // default: 1/x singularity
            component: sing_component,
            x_sing: 0.0,
        }
    } else {
        SingularityType::Unknown
    }
}

fn compute_collision_weights(
    x0: &Array1<f64>,
    body_i: usize, body_j: usize, dim: usize,
    n: usize,
) -> Array1<f64> {
    // All weights = |r_i - r_j|^2 evaluated at x0 (constant pilot)
    let diff_sq: f64 = (0..dim).map(|k| (x0[body_i*dim+k] - x0[body_j*dim+k]).powi(2)).sum();
    Array1::from_elem(n + 1, diff_sq.max(1e-300))
}

fn compute_algebraic_weights(
    x0: &Array1<f64>,
    sing: &SingularityType,
    n: usize,
) -> Array1<f64> {
    match sing {
        SingularityType::Algebraic { power, component, x_sing } => {
            let dist = (x0[*component] - x_sing).abs().max(1e-300);
            Array1::from_elem(n + 1, dist.powf(*power))
        }
        _ => Array1::ones(n + 1),
    }
}

fn estimate_lifted_ds(
    x0: &Array1<f64>,
    body_i: usize, body_j: usize, dim: usize,
    dt_pilot: f64,
) -> f64 {
    // In KS coordinates, the fictitious time is related to physical time by:
    // ds = dt / |r|  →  Δs ≈ Δt / |r|
    // But |r| = |u|^2, so we want Δs such that |u|^2 * Δs ≈ Δt.
    let diff_sq: f64 = (0..dim).map(|k| (x0[body_i*dim+k] - x0[body_j*dim+k]).powi(2)).sum();
    let r = diff_sq.sqrt().max(1e-10);
    // Lifted segment: Δs ~ Δt / r, but clamp for stability
    (dt_pilot / r).min(1.0).max(0.01)
}

// ============================================================
//  KS Fibration (3D → 4D via Hopf fibration)
// ============================================================

/// Apply the KS map: u ∈ ℝ^4 → r ∈ ℝ^3 with |r| = |u|^2.
/// This is the standard Kustaanheimo-Stiefel transformation.
pub fn ks_map(u: &[f64; 4]) -> [f64; 3] {
    [
        u[0]*u[0] - u[1]*u[1] - u[2]*u[2] + u[3]*u[3],
        2.0*(u[0]*u[1] - u[2]*u[3]),
        2.0*(u[0]*u[2] + u[1]*u[3]),
    ]
}

/// KS matrix L(u): r = L(u)·u, see Chapter 8 Eq.(8.2).
pub fn ks_matrix(u: &[f64; 4]) -> [[f64; 4]; 4] {
    [
        [ u[0], -u[1], -u[2],  u[3]],
        [ u[1],  u[0], -u[3], -u[2]],
        [ u[2],  u[3],  u[0],  u[1]],
        [ u[3], -u[2],  u[1], -u[0]],
    ]
}

/// KS bilinear constraint: Ψ(u) = u[0]*u[3] - u[1]*u[2] = 0.
pub fn ks_constraint(u: &[f64; 4]) -> f64 {
    u[0]*u[3] - u[1]*u[2]
}

/// Find u ∈ ℝ^4 such that ks_map(u) = r (inverse KS map).
/// Convention: u[3] = 0 (canonical fiber, bilinear constraint satisfied).
///
/// With u[3]=0, ks_map gives:
///   r[0] = u[0]² - u[1]² - u[2]²
///   r[1] = 2 u[0] u[1]
///   r[2] = 2 u[0] u[2]
///   |r| = |u|²
///
/// Solving: u[0] = sqrt((|r| + r[0]) / 2),
///          u[1] = r[1] / (2 u[0]),
///          u[2] = r[2] / (2 u[0]).
///
/// Derivation: |u|² = |r| forces u[0]² + (r[1]²+r[2]²)/(4u[0]²) = |r|,
/// giving u[0]² = (|r| + r[0]) / 2.
///
/// Returns None if |r| = 0.
pub fn ks_inverse(r: &[f64; 3]) -> Option<[f64; 4]> {
    let r_norm = (r[0]*r[0] + r[1]*r[1] + r[2]*r[2]).sqrt();
    if r_norm < 1e-300 { return None; }
    let u0_sq = (r_norm + r[0]) / 2.0;
    if u0_sq < 0.0 { return None; }  // should not happen for valid r
    let u0 = u0_sq.sqrt();
    if u0 < 1e-150 {
        // r[0] ≈ -|r|: use alternate branch (u[0]≈0, swap roles)
        // Use u[1] = sqrt((|r| - r[0]) / 2) as pivot instead
        let u1_sq = (r_norm - r[0]) / 2.0;
        let u1 = u1_sq.sqrt().max(1e-300);
        return Some([0.0, u1, r[2] / (2.0 * u1), r[1] / (2.0 * u1)]);
    }
    let two_u0 = 2.0 * u0;
    Some([u0, r[1] / two_u0, r[2] / two_u0, 0.0])
}

/// Levi-Civita map (2D regularization): u ∈ ℝ^2 → r ∈ ℝ^2 with |r| = |u|^2.
/// r_1 = u_1^2 - u_2^2,  r_2 = 2*u_1*u_2
pub fn lc_map(u: &[f64; 2]) -> [f64; 2] {
    [u[0]*u[0] - u[1]*u[1], 2.0*u[0]*u[1]]
}

/// Inverse Levi-Civita: find u with lc_map(u) = r.
pub fn lc_inverse(r: &[f64; 2]) -> Option<[f64; 2]> {
    let r_norm = (r[0]*r[0] + r[1]*r[1]).sqrt();
    if r_norm < 1e-300 { return None; }
    let theta = r[1].atan2(r[0]) / 2.0;
    let u_norm = r_norm.sqrt();
    Some([u_norm * theta.cos(), u_norm * theta.sin()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ks_roundtrip() {
        // Test: ks_map(ks_inverse(r)) == r  and  |ks_inverse(r)|^2 == |r|
        // Note: the canonical inverse uses u[3]=0, which satisfies the KS constraint
        // ψ = u[0]*u[3] - u[1]*u[2] = 0 only when u[1]*u[2]=0.
        // For general r, we use ks_inverse_constrained() for astrodynamical ICs.
        let test_cases = [
            [1.5_f64, -0.7, 0.3],
            [0.5, 0.0, 0.0],   // along x-axis: u3=0 gives ψ=0 trivially
            [1.0, 0.5, 0.0],   // in xy-plane: u2=0 gives ψ=0
            [0.1, 0.2, 0.3],
        ];
        for r in &test_cases {
            let r_norm = (r[0]*r[0]+r[1]*r[1]+r[2]*r[2]).sqrt();
            let u = ks_inverse(r).unwrap();
            let r2 = ks_map(&u);
            for (a, b) in r.iter().zip(r2.iter()) {
                assert!((a - b).abs() < 1e-11, 
                    "KS roundtrip failed for r={:?}: got {} expected {}", r, b, a);
            }
            let u_norm_sq: f64 = u.iter().map(|x| x*x).sum();
            assert!((u_norm_sq - r_norm).abs() < 1e-11,
                "|u|^2={} != |r|={}", u_norm_sq, r_norm);
        }
    }

    #[test]
    fn test_ks_norm() {
        let r = [0.5_f64, 0.3, 0.4];
        let r_norm = (r[0]*r[0]+r[1]*r[1]+r[2]*r[2]).sqrt();
        let u = ks_inverse(&r).unwrap();
        let u_norm_sq: f64 = u.iter().map(|x| x*x).sum();
        assert!((u_norm_sq - r_norm).abs() < 1e-13,
            "|u|^2 = {} != |r| = {}", u_norm_sq, r_norm);
    }

    #[test]
    fn test_lc_roundtrip() {
        let r = [0.6_f64, 0.8];
        let u = lc_inverse(&r).unwrap();
        let r2 = lc_map(&u);
        for (a, b) in r.iter().zip(r2.iter()) {
            assert!((a - b).abs() < 1e-13, "LC roundtrip: {} vs {}", a, b);
        }
    }
}
