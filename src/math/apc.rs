// src/math/apc.rs
//
// Arbitrary Polynomial Chaos (APC) Engine
// ======================================================================
// Constructs orthogonal polynomial bases and collocation nodes directly 
// from raw statistical moments, bypassing the need for continuous PDFs.

use nalgebra::{DMatrix, SymmetricEigen};

/// Computes the collocation nodes and weights for Arbitrary Polynomial Chaos
/// using the Hankel-Cholesky method to derive the Jacobi matrix.
/// 
/// * `moments` - The raw statistical moments [mu_0, mu_1, ..., mu_{2N-1}]
/// * `n_points` - The desired number of collocation points N
pub fn compute_apc_collocation(moments: &[f64], n_points: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    assert!(moments.len() >= 2 * n_points, "Requires 2N moments for N points.");
    let mut hankel = DMatrix::<f64>::zeros(n_points, n_points);
    for i in 0..n_points {
        for j in 0..n_points { hankel[(i, j)] = moments[i + j]; }
    }
    let cholesky = hankel.cholesky()
        .ok_or_else(|| "Hankel matrix is not positive definite. \
                        Moments are invalid or distribution is degenerate.".to_string())?;
    let l = cholesky.l();

    // 3. Extract the three-term recurrence coefficients (alpha, beta)
    let mut alpha = vec![0.0; n_points];
    let mut beta = vec![0.0; n_points];

    for j in 0..n_points {
        let l_j_j = l[(j, j)];
        let l_jp1_j = if j + 1 < n_points { l[(j + 1, j)] } else { 0.0 };
        
        let l_j_jm1 = if j > 0 { l[(j, j - 1)] } else { 0.0 };
        let l_jm1_jm1 = if j > 0 { l[(j - 1, j - 1)] } else { 1.0 }; // Prevent div by zero, term cancels anyway

        alpha[j] = l_jp1_j / l_j_j - if j > 0 { l_j_jm1 / l_jm1_jm1 } else { 0.0 };
        
        if j > 0 {
            beta[j] = (l_j_j / l_jm1_jm1).powi(2);
        }
    }

    // 4. Construct the symmetric tridiagonal Jacobi matrix T
    let mut t_mat = DMatrix::<f64>::zeros(n_points, n_points);
    for j in 0..n_points {
        t_mat[(j, j)] = alpha[j];
        if j < n_points - 1 {
            let off_diag = beta[j + 1].sqrt();
            t_mat[(j, j + 1)] = off_diag;
            t_mat[(j + 1, j)] = off_diag;
        }
    }

    // 5. Golub-Welsch Algorithm: Eigenvalues are nodes, eigenvectors give weights
    let eigen = SymmetricEigen::new(t_mat);
    let mut nodes_weights: Vec<(f64, f64)> = (0..n_points)
        .map(|i| {
            let node = eigen.eigenvalues[i];
            let v = eigen.eigenvectors[(0, i)];
            let weight = moments[0] * v * v;
            (node, weight)
        })
        .collect();

    // Sort by node value for consistency
    nodes_weights.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let nodes = nodes_weights.iter().map(|nw| nw.0).collect();
    let weights = nodes_weights.iter().map(|nw| nw.1).collect();

    Ok((nodes, weights))
}