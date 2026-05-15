// src/solvers/stm.rs
//
// Spectral State Transition Matrix (STM) Propagation
// ==================================================
// Propagates the augmented state [x, vec(Phi)] simultaneously in the 
// Chebyshev spectral domain. Enables exact analytical differentiation 
// with respect to initial conditions for the SC-EKF and boundary value problems.

use ndarray::{Array1, Array2};

/// Evaluates the augmented vector field for STM propagation.
/// 
/// * `rhs_fn` - Closure evaluating the base vector field f(x, t).
/// * `jac_fn` - Closure evaluating the Jacobian matrix J(x, t) = df/dx.
/// * `y_batch` - Augmented state batch [n + n^2, npts].
/// * `t_batch` - Time batch [npts].
/// * `ndim` - Dimension of the base state x (n).
pub fn stm_rhs_batch<F, J>(
    rhs_fn: &F,
    jac_fn: &J,
    y_batch: &Array2<f64>,
    t_batch: &Array1<f64>,
    ndim: usize,
) -> Array2<f64>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Array2<f64>,
    J: Fn(&Array2<f64>, &Array1<f64>) -> Array2<f64>,
{
    let npts = y_batch.ncols();
    let augmented_dim = ndim + ndim * ndim;
    assert_eq!(y_batch.nrows(), augmented_dim, "Augmented state dimension mismatch.");

    let mut out = Array2::zeros((augmented_dim, npts));

    // 1. Extract base state x_batch
    let x_batch = y_batch.slice(ndarray::s![0..ndim, ..]).to_owned();

    // 2. Evaluate base vector field and Jacobian
    let f_vals = rhs_fn(&x_batch, t_batch);
    let j_vals = jac_fn(&x_batch, t_batch); // Shape: [ndim * ndim, npts]

    // 3. Compute augmented dynamics
    for pt in 0..npts {
        // Copy base state derivatives
        for i in 0..ndim {
            out[[i, pt]] = f_vals[[i, pt]];
        }

        // Reconstruct Jacobian matrix for this point
        let mut j_mat = nalgebra::DMatrix::<f64>::zeros(ndim, ndim);
        for r in 0..ndim {
            for c in 0..ndim {
                j_mat[(r, c)] = j_vals[[r * ndim + c, pt]];
            }
        }

        // Reconstruct STM matrix Phi for this point
        let mut phi_mat = nalgebra::DMatrix::<f64>::zeros(ndim, ndim);
        for r in 0..ndim {
            for c in 0..ndim {
                let idx = ndim + r * ndim + c;
                phi_mat[(r, c)] = y_batch[[idx, pt]];
            }
        }

        // Matrix multiplication: dPhi/dt = J(t) * Phi
        let dphi_mat = j_mat * phi_mat;

        // Flatten dPhi/dt back into the output array
        for r in 0..ndim {
            for c in 0..ndim {
                let idx = ndim + r * ndim + c;
                out[[idx, pt]] = dphi_mat[(r, c)];
            }
        }
    }

    out
}