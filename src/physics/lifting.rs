// src/physics/lifting.rs
//
// Automatic Singularity Detection and State-Space Lifting
// =========================================================

use ndarray::{Array1, Array2};
use crate::math::chebyshev::{lobatto_nodes, ChebVec};

use pyo3::prelude::*;
use numpy::ToPyArray;

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
            let py_x = x_batch.to_pyarray(py);
            let py_t = t_batch.to_pyarray(py);
            
            let result = self.py_obj.as_ref(py).call1((py_x, py_t))
                .expect("Python ODE evaluation failed during diagnosis.");
            
            let py_out: numpy::PyReadonlyArray2<f64> = result.extract()
                .expect("Python ODE must return a 2D numpy float64 array.");
            
            py_out.as_array().to_owned()
        })
    }
}

// ============================================================
//  Public types
// ============================================================

#[derive(Debug, Clone, PartialEq)]
pub enum SingularityType {
    None,
    Algebraic { power: f64, component: usize, x_sing: f64 },
    Collision { body_i: usize, body_j: usize, dim: usize },
    MultiPole { poles: Vec<(f64, f64)> },
    Unknown,
}

#[derive(Debug, Clone)]
pub struct LiftDiagnosis {
    pub bernstein_rho: f64,
    pub singularity: SingularityType,
    pub apply_fibration: bool,
    pub pilot_weights: Array1<f64>,
    pub suggested_ds: f64,
}

pub trait OdeSystem: Send + Sync {
    fn ndim(&self) -> usize;
    fn rhs_batch(&self, x_batch: &Array2<f64>, t_batch: &Array1<f64>) -> Array2<f64>;
    fn sundman_weight(&self, _x: &Array1<f64>) -> Option<f64> { None }
    fn ks_collision_pair(&self) -> Option<(usize, usize, usize)> { None } 
}

// ============================================================
//  Lifted system
// ============================================================

pub struct LiftedSystem<'a> {
    pub base: &'a dyn OdeSystem,
    pub diagnosis: LiftDiagnosis,
    pub ks_pair: Option<(usize, usize, usize)>,
}

impl<'a> LiftedSystem<'a> {
    pub fn rhs_lifted(&self, x_lifted: &Array1<f64>) -> Array1<f64> {
        let ndim = self.base.ndim();
        let x = x_lifted.slice(ndarray::s![..ndim]).to_owned();
        let t = x_lifted[ndim];
        
        let x_batch = x.view().to_shape((ndim, 1)).unwrap().to_owned();
        let t_batch = Array1::from_vec(vec![t]);
        let f_batch = self.base.rhs_batch(&x_batch, &t_batch);
        let f = f_batch.column(0).to_owned();
        
        let w = self.compute_weight(&x);
        
        let mut out = Array1::zeros(ndim + 1);
        for i in 0..ndim {
            out[i] = w * f[i];
        }
        out[ndim] = w;
        out
    }

    pub fn rhs_batch_lifted(&self, x_batch: &Array2<f64>) -> Array2<f64> {
        let ndim = self.base.ndim();
        let n_pts = x_batch.ncols();
        let mut out = Array2::zeros((ndim + 1, n_pts));
        
        let x_phys = x_batch.slice(ndarray::s![..ndim, ..]).to_owned();
        let t_vec = x_batch.row(ndim).to_owned();
        
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

    fn compute_weight(&self, x: &Array1<f64>) -> f64 {
        if let Some(w) = self.base.sundman_weight(x) {
            return w.max(1e-300);
        }
        if let Some((i, j, dim)) = self.ks_pair {
            let r_i = x.slice(ndarray::s![i..i+dim]).to_owned();
            let r_j = x.slice(ndarray::s![j..j+dim]).to_owned();
            let diff_sq: f64 = r_i.iter().zip(r_j.iter()).map(|(a,b)| (a-b).powi(2)).sum();
            return diff_sq.max(1e-300);
        }
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
                poles.iter().map(|(loc, power)| {
                    let dist = x.iter().map(|xi| (xi - loc).abs()).fold(f64::INFINITY, f64::min);
                    dist.powf(*power).max(1e-300)
                }).product::<f64>().max(1e-300)
            }
            _ => 1.0, 
        }
    }
}

// ============================================================
//  Diagnosis engine
// ============================================================

pub fn diagnose<S: OdeSystem>(
    system: &S,
    x0: &Array1<f64>,
    t0: f64,
    dt_pilot: f64,
    pilot_n: usize,
    rho_threshold: f64, 
) -> LiftDiagnosis {
    let rho = pilot_bernstein_rho(system, x0, t0, dt_pilot, pilot_n);

    if rho >= rho_threshold {
        return LiftDiagnosis {
            bernstein_rho: rho,
            singularity: SingularityType::None,
            apply_fibration: false,
            pilot_weights: Array1::ones(pilot_n + 1),
            suggested_ds: dt_pilot,
        };
    }

    if let Some((i, j, dim)) = system.ks_collision_pair() {
        return LiftDiagnosis {
            bernstein_rho: rho,
            singularity: SingularityType::Collision { body_i: i, body_j: j, dim },
            apply_fibration: dim == 3,
            pilot_weights: compute_collision_weights(x0, i, j, dim, pilot_n),
            suggested_ds: estimate_lifted_ds(x0, i, j, dim, dt_pilot),
        };
    }

    let sing = detect_algebraic_singularity(system, x0, t0, dt_pilot, pilot_n);
    let pilot_weights = compute_algebraic_weights(x0, &sing, pilot_n);
    let suggested_ds = 0.1_f64.min(dt_pilot * 2.0); 

    LiftDiagnosis {
        bernstein_rho: rho,
        singularity: sing,
        apply_fibration: false,
        pilot_weights,
        suggested_ds,
    }
}

fn pilot_bernstein_rho<S: OdeSystem>(
    system: &S,
    x0: &Array1<f64>,
    t0: f64,
    dt: f64,
    n: usize,
) -> f64 {
    let ndim = system.ndim();
    let nodes = lobatto_nodes(n);
    let t_batch = Array1::from_shape_fn(n + 1, |j| t0 + dt * (nodes[j] + 1.0) / 2.0);
    let x_batch = Array2::from_shape_fn((ndim, n + 1), |(i, _)| x0[i]);
    let f_batch = system.rhs_batch(&x_batch, &t_batch);
    
    let cv = ChebVec::from_value_matrix(&f_batch);
    cv.estimate_bernstein_rho()
}

fn detect_algebraic_singularity<S: OdeSystem>(
    system: &S,
    x0: &Array1<f64>,
    t0: f64,
    _dt: f64,
    _n: usize,
) -> SingularityType {
    let ndim = system.ndim();
    let x_batch = Array2::from_shape_fn((ndim, 1), |(i, _)| x0[i]);
    let t_batch = Array1::from_vec(vec![t0]);
    let f = system.rhs_batch(&x_batch, &t_batch);
    
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
        SingularityType::Algebraic {
            power: 1.0, 
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
    let diff_sq: f64 = (0..dim).map(|k| (x0[body_i*dim+k] - x0[body_j*dim+k]).powi(2)).sum();
    let r = diff_sq.sqrt().max(1e-10);
    (dt_pilot / r).min(1.0).max(0.01)
}

// ============================================================
//  KS Fibration (3D → 4D via Hopf fibration)
// ============================================================

pub fn ks_map(u: &[f64; 4]) -> [f64; 3] {
    [
        u[0]*u[0] - u[1]*u[1] - u[2]*u[2] + u[3]*u[3],
        2.0*(u[0]*u[1] - u[2]*u[3]),
        2.0*(u[0]*u[2] + u[1]*u[3]),
    ]
}

pub fn ks_matrix(u: &[f64; 4]) -> [[f64; 4]; 4] {
    [
        [ u[0], -u[1], -u[2],  u[3]],
        [ u[1],  u[0], -u[3], -u[2]],
        [ u[2],  u[3],  u[0],  u[1]],
        [ u[3], -u[2],  u[1], -u[0]],
    ]
}

pub fn ks_constraint(u: &[f64; 4]) -> f64 {
    u[0]*u[3] - u[1]*u[2]
}

pub fn ks_inverse(r: &[f64; 3]) -> Option<[f64; 4]> {
    let r_norm = (r[0]*r[0] + r[1]*r[1] + r[2]*r[2]).sqrt();
    if r_norm < 1e-300 { return None; }
    let u0_sq = (r_norm + r[0]) / 2.0;
    if u0_sq < 0.0 { return None; } 
    let u0 = u0_sq.sqrt();
    if u0 < 1e-150 {
        let u1_sq = (r_norm - r[0]) / 2.0;
        let u1 = u1_sq.sqrt().max(1e-300);
        let two_u1 = 2.0 * u1;
        return Some([r[1] / two_u1, u1, 0.0, r[2] / two_u1]);
    }
    let two_u0 = 2.0 * u0;
    Some([u0, r[1] / two_u0, r[2] / two_u0, 0.0])
}

pub fn lc_map(u: &[f64; 2]) -> [f64; 2] {
    [u[0]*u[0] - u[1]*u[1], 2.0*u[0]*u[1]]
}

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

    #[test]
    fn test_ks_roundtrip() {
        let test_cases = [
            [1.5_f64, -0.7, 0.3],
            [0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [0.1, 0.2, 0.3],
            [-1.0, 0.0, 0.0],
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