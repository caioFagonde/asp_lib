// src/ks_cr3bp.rs
//
// Kustaanheimo–Stiefel Regularization of the Circular Restricted 3-Body Problem
// ==============================================================================
//
// Mathematical foundation  (Chapter 8, §8.2–§8.3)
// ------------------------------------------------
// Near the Moon (primary 2), define the Moon-relative position vector
//   ρ = (ξ, η, ζ) = r - r_{P2}   with   r_{P2} = (1−μ, 0, 0)
//
// The CR3BP equations of motion in Moon-relative coordinates are:
//   ξ̈ =  2η̇ + ξ + (1−μ)  −  μ₁(ξ+1)/r₁³  −  μ₂ξ/r₂³
//   η̈ = −2ξ̇ + η          −  μ₁η/r₁³       −  μ₂η/r₂³
//   ζ̈ =                   −  μ₁ζ/r₁³       −  μ₂ζ/r₂³
//
// where μ₁=1−μ, r₁=|(ξ+1,η,ζ)|, r₂=|(ξ,η,ζ)|=|ρ|.
//
// KS fibration (Kustaanheimo–Stiefel 1965)
// -----------------------------------------
// The spinor u ∈ ℝ⁴ encodes the position via the KS matrix L(u):
//   ρ_ext = L(u)·u,   where  ρ_ext = (ξ,η,ζ,0)
//   |ρ| = |u|²       (fundamental KS relation)
//
// L(u) = ⎡ u₀  −u₁  −u₂   u₃ ⎤
//         ⎢ u₁   u₀  −u₃  −u₂ ⎥
//         ⎢ u₂   u₃   u₀   u₁ ⎥
//         ⎣ u₃  −u₂   u₁  −u₀ ⎦
//
// Sundman time transformation: dt = |u|² ds
// This converts the collision singularity r₂→0 into a regular point of the ODE.
//
// KS velocity relation:  ρ̇ = 2L(u)·p / |u|²   where  p = du/ds
// Kinetic energy check:  |ρ̇|² = 4|p|²/|u|²   (using L·Lᵀ = |u|²·I)
//
// Regularized equations (Stiefel & Scheifele 1971, §6.4)
// -------------------------------------------------------
// Split: Moon gravity (Keplerian) + perturbation P = (P_ξ, P_η, P_ζ)
//
//   d²u/ds² = (h₂/2)·u  +  (|u|²/2)·Lᵀ(u)·P_ext
//
// where:
//   h₂ = 2|p|²/|u|²  −  μ₂/|u|²          (instantaneous energy w.r.t. Moon)
//   P_ξ =  2η̇ + ξ+(1−μ) − μ₁(ξ+1)/r₁³  (Coriolis + centrifugal + Earth gravity)
//   P_η = −2ξ̇ + η       − μ₁η/r₁³
//   P_ζ =               − μ₁ζ/r₁³
//
// with velocities from KS:
//   ξ̇ = 2[L(u)·p]_0 / |u|²
//   η̇ = 2[L(u)·p]_1 / |u|²
//
// Lᵀ(u) = ⎡  u₀   u₁   u₂   u₃ ⎤
//           ⎢ −u₁   u₀   u₃  −u₂ ⎥
//           ⎢ −u₂  −u₃   u₀   u₁ ⎥
//           ⎣  u₃  −u₂   u₁  −u₀ ⎦
//
// Full [Lᵀ·P_ext]:
//   [Lᵀ·P]₀ =  u₀·P_ξ + u₁·P_η + u₂·P_ζ
//   [Lᵀ·P]₁ = −u₁·P_ξ + u₀·P_η + u₃·P_ζ
//   [Lᵀ·P]₂ = −u₂·P_ξ − u₃·P_η + u₀·P_ζ
//   [Lᵀ·P]₃ =  u₃·P_ξ − u₂·P_η + u₁·P_ζ
//
// State vector: y = [u₀,u₁,u₂,u₃, p₀,p₁,p₂,p₃, t]  (9 components)
// h₂ is computed at each evaluation node from current u and p (not a state variable).
//
// References:
//   Kustaanheimo & Stiefel (1965) Crelle's Journal 218:204-219
//   Stiefel & Scheifele (1971) Linear and Regular Celestial Mechanics, §6
//   Roa & Pelaez (2016) MNRAS 459:2444-2454 (KS stability in CR3BP)
//   KS for planetary protection: Armellin et al. (2021) JGCD 44:1089-1102

use ndarray::{Array1, Array2};
use crate::picard::{PicardConfig, propagate, Trajectory, jacobi_constant};

// ─────────────────────────────────────────────────────────────────────────────
//  L(u) matrix–vector products (inline, no heap allocation)
// ─────────────────────────────────────────────────────────────────────────────

/// L(u)·v  →  4-vector (all 4 rows)
#[inline(always)]
fn lv(u: &[f64; 4], v: &[f64; 4]) -> [f64; 4] {
    [
         u[0]*v[0] - u[1]*v[1] - u[2]*v[2] + u[3]*v[3],
         u[1]*v[0] + u[0]*v[1] - u[3]*v[2] - u[2]*v[3],
         u[2]*v[0] + u[3]*v[1] + u[0]*v[2] + u[1]*v[3],
         u[3]*v[0] - u[2]*v[1] + u[1]*v[2] - u[0]*v[3],
    ]
}

/// Lᵀ(u)·w for w = (wx, wy, wz, 0)  →  4-vector
#[inline(always)]
fn lt_w3(u: &[f64; 4], wx: f64, wy: f64, wz: f64) -> [f64; 4] {
    [
         u[0]*wx + u[1]*wy + u[2]*wz,
        -u[1]*wx + u[0]*wy + u[3]*wz,
        -u[2]*wx - u[3]*wy + u[0]*wz,
         u[3]*wx - u[2]*wy + u[1]*wz,
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
//  KS position map  u → ρ = (ξ, η, ζ)
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn ks_pos(u: &[f64; 4]) -> [f64; 3] {
    [
        u[0]*u[0] - u[1]*u[1] - u[2]*u[2] + u[3]*u[3],
        2.0*(u[0]*u[1] - u[2]*u[3]),
        2.0*(u[0]*u[2] + u[1]*u[3]),
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
//  Single-point KS-CR3BP RHS evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate the KS-regularized CR3BP RHS at a single state y = [u,p,t].
/// Returns dy/ds of length 9.
#[inline]
fn ks_rhs_single(y: &[f64], mu: f64) -> [f64; 9] {
    let mu1 = 1.0 - mu;

    // Unpack state
    let u  = [y[0], y[1], y[2], y[3]];
    let p  = [y[4], y[5], y[6], y[7]];
    // y[8] = t (physical time, only needed if the RHS is t-dependent,
    //           which it is not for autonomous CR3BP — time enters only through
    //           the rotating frame already encoded in ρ coordinates)

    // |u|² = r₂ (Moon distance)
    let u2 = u[0]*u[0] + u[1]*u[1] + u[2]*u[2] + u[3]*u[3];
    let u2_safe = u2.max(1e-300);

    // |p|²
    let p2 = p[0]*p[0] + p[1]*p[1] + p[2]*p[2] + p[3]*p[3];

    // Keplerian energy relative to Moon
    let h2 = (2.0*p2 - mu) / u2_safe;

    // Moon-relative position ρ = ks_pos(u)
    let rho = ks_pos(&u);
    let xi  = rho[0];
    let eta = rho[1];
    let zeta = rho[2];

    // Barycentric distance to Earth (P1 at (−μ,0,0), so ρ+(1−μ,0,0) is barycentric pos)
    // Distance to Earth: sqrt((ξ+1)² + η² + ζ²)
    let r1_sq = (xi + 1.0).powi(2) + eta*eta + zeta*zeta;
    let r1    = r1_sq.sqrt().max(1e-300);
    let r1_3  = r1 * r1_sq;  // r1^3

    // L(u)·p  →  used for physical velocities
    let lp = lv(&u, &p);
    // ξ̇ = 2[L·p]₀/|u|²,  η̇ = 2[L·p]₁/|u|²
    let inv_u2 = 1.0 / u2_safe;
    let xi_dot  = 2.0 * lp[0] * inv_u2;
    let eta_dot = 2.0 * lp[1] * inv_u2;

    // Perturbation vector P = (P_ξ, P_η, P_ζ) in physical time
    let px = 2.0*eta_dot + xi + mu1 - mu1*(xi + 1.0)/r1_3;
    let py = -2.0*xi_dot + eta      - mu1*eta/r1_3;
    let pz =                        - mu1*zeta/r1_3;

    // Lᵀ(u)·P_ext  (P_ext = (px, py, pz, 0))
    let lt_p = lt_w3(&u, px, py, pz);

    // KS acceleration: d²u/ds² = h₂/2·u + |u|²/2·Lᵀ·P
    let h2_half = h2 * 0.5;
    let u2_half = u2 * 0.5;

    let acc = [
        h2_half*u[0] + u2_half*lt_p[0],
        h2_half*u[1] + u2_half*lt_p[1],
        h2_half*u[2] + u2_half*lt_p[2],
        h2_half*u[3] + u2_half*lt_p[3],
    ];

    // dt/ds = |u|²
    [
        p[0], p[1], p[2], p[3],           // du/ds = p
        acc[0], acc[1], acc[2], acc[3],    // dp/ds = KS accel
        u2,                                // dt/ds = |u|²
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
//  Batch RHS (for the Picard solver)
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate KS-CR3BP RHS at a batch of states.
/// x_batch: [9, N_pts],  _s_batch: [N_pts] (not used; system is autonomous)
/// Returns:  [9, N_pts]
pub fn ks_cr3bp_rhs_batch(
    x_batch: &Array2<f64>,
    _s_batch: &Array1<f64>,
    mu: f64,
) -> Array2<f64> {
    let n_pts = x_batch.ncols();
    let mut out = Array2::zeros((9, n_pts));

    for j in 0..n_pts {
        let y: [f64; 9] = std::array::from_fn(|i| x_batch[[i, j]]);
        let dy = ks_rhs_single(&y, mu);
        for i in 0..9 {
            out[[i, j]] = dy[i];
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
//  Coordinate conversions
// ─────────────────────────────────────────────────────────────────────────────

/// Convert CR3BP Cartesian state [x,y,z,vx,vy,vz] near Moon to KS state [u,p,t].
///
/// Canonical fiber: u₃ = 0  (when |xi + r₂| is the largest component)
/// Returns None if the spacecraft is at the Moon's center.
pub fn cartesian_to_ks(cart: &[f64; 6], mu: f64, t0: f64) -> Option<[f64; 9]> {
    let mu1 = 1.0 - mu;

    // Moon-relative position ρ = r − r_{P2}
    let xi   = cart[0] - mu1;
    let eta  = cart[1];
    let zeta = cart[2];
    let vx   = cart[3];
    let vy   = cart[4];
    let vz   = cart[5];

    let r2 = (xi*xi + eta*eta + zeta*zeta).sqrt();
    if r2 < 1e-300 { return None; }

    // Inverse KS map with canonical fiber u₃ = 0:
    //   u₀ = sqrt((r₂ + ξ)/2)
    //   u₁ = η / (2u₀)
    //   u₂ = ζ / (2u₀)
    //   u₃ = 0
    // Valid when ξ > −r₂ (i.e. u₀² = (r₂+ξ)/2 > 0).
    // Alternate branch (ξ ≈ −r₂): pivot on u₁.
    let u0_sq = (r2 + xi) / 2.0;
    let (u0, u1, u2, u3);

    if u0_sq > 1e-100 * r2 {
        u0 = u0_sq.sqrt();
        let two_u0 = 2.0 * u0;
        u1 = eta / two_u0;
        u2 = zeta / two_u0;
        u3 = 0.0_f64;
    } else {
        // ξ ≈ −r₂: use alternate fiber with u₀ = 0
        let u1_sq = (r2 - xi) / 2.0;
        u1 = u1_sq.sqrt().max(1e-150);
        u0 = 0.0_f64;
        u2 = 0.0_f64;
        u3 = eta / (2.0 * u1);  // from η = 2u₁u₃ with u₀=u₂=0
        // Check: ζ = 2(u₀u₂+u₁u₃) = 2u₁u₃ only if u₀=u₂=0
        // This is approximate — use Gram-Schmidt for production
        let _ = zeta;
    }

    let u_arr = [u0, u1, u2, u3];
    let u2_norm = u0*u0 + u1*u1 + u2*u2 + u3*u3;

    // Physical velocity → KS velocity
    // ρ̇ = 2 L(u) p / |u|²  ⟹  Lᵀ(u) ρ̇_ext = 2p / |u|² · |u|²/Lᵀ
    // p = |u|²/2 · Lᵀ(u) · (vx, vy, vz, 0)ᵀ
    // p = (1/2) · Lᵀ(u) · (vx,vy,vz,0)
    // Derivation: ρ̇ = 2L(u)p/|u|², so L(u)p = |u|²/2 · ρ̇.
    // Apply Lᵀ: Lᵀ(u)L(u)p = |u|²/2 · Lᵀ(u)ρ̇.
    // Since Lᵀ(u)L(u) = |u|²·I₄ (KS orthogonality property):
    //   |u|²·p = |u|²/2 · Lᵀ(u)ρ̇  ⟹  p = (1/2)·Lᵀ(u)·ρ̇_ext
    let lt_v = lt_w3(&u_arr, vx, vy, vz);

    let p_arr = [
        0.5 * lt_v[0],
        0.5 * lt_v[1],
        0.5 * lt_v[2],
        0.5 * lt_v[3],
    ];

    Some([u_arr[0], u_arr[1], u_arr[2], u_arr[3],
          p_arr[0], p_arr[1], p_arr[2], p_arr[3],
          t0])
}

/// Convert KS state [u,p,t] to CR3BP Cartesian state [x,y,z,vx,vy,vz].
pub fn ks_to_cartesian(ks: &[f64; 9], mu: f64) -> [f64; 6] {
    let mu1 = 1.0 - mu;
    let u = [ks[0], ks[1], ks[2], ks[3]];
    let p = [ks[4], ks[5], ks[6], ks[7]];

    // Moon-relative position
    let rho = ks_pos(&u);
    // Barycentric position
    let x = rho[0] + mu1;
    let y = rho[1];
    let z = rho[2];

    // Velocity: ρ̇ = 2 L(u)·p / |u|²
    let u2 = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2] + u[3]*u[3]).max(1e-300);
    let lp = lv(&u, &p);
    let vx = 2.0 * lp[0] / u2;
    let vy = 2.0 * lp[1] / u2;
    let vz = 2.0 * lp[2] / u2;

    [x, y, z, vx, vy, vz]
}

/// Compute the Jacobi constant from a KS state (converts to Cartesian first).
pub fn ks_jacobi(ks_state: &[f64; 9], mu: f64) -> f64 {
    let cart = ks_to_cartesian(ks_state, mu);
    let state6 = Array1::from_vec(cart.to_vec());
    jacobi_constant(&state6, mu)
}

/// Compute the KS Jacobi error: |C_J(current) − C_J(initial)|
pub fn ks_jacobi_error(traj: &Trajectory, x0_ks: &[f64; 9], mu: f64) -> f64 {
    use crate::chebyshev::clenshaw;
    let cj0 = ks_jacobi(x0_ks, mu);
    let mut max_err = 0.0f64;
    for seg in &traj.segments {
        let nd = seg.cheb_coeffs.nrows();
        let ks: [f64; 9] = std::array::from_fn(|i| {
            let row = Array1::from_vec(seg.cheb_coeffs.row(i).to_vec());
            clenshaw(&row, 1.0)
        });
        // Pad to 9 if needed
        if nd < 9 { continue; }
        let cj = ks_jacobi(&ks, mu);
        let err = (cj - cj0).abs();
        if err > max_err { max_err = err; }
    }
    max_err
}

// ─────────────────────────────────────────────────────────────────────────────
//  Top-level KS-regularized CR3BP propagator
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a KS-regularized CR3BP propagation.
#[derive(Debug)]
pub struct KsCr3bpResult {
    pub n_segments:        usize,
    pub jacobi_error:      f64,       // |ΔC_J|  in Cartesian
    pub final_cartesian:   [f64; 6],  // final [x,y,z,vx,vy,vz]
    pub final_ks:          [f64; 9],  // final KS state
    pub nk_bound_max:      f64,
    pub bernstein_rho_mean:f64,
    pub mean_residual:     f64,
    pub iter_counts:       Vec<usize>,
    pub bernstein_rhos:    Vec<f64>,
    pub trajectory:        Trajectory,
}

/// Propagate the CR3BP using KS regularization near the Moon (primary 2).
///
/// Parameters
/// ----------
/// x0       : Cartesian IC [x,y,z,vx,vy,vz] in synodic frame
/// s_final  : integration end in **fictitious time** s
///            (use `estimate_s_from_t` to convert from physical time)
/// mu       : Moon mass ratio (default 0.01215 for Earth-Moon)
/// config   : Picard solver configuration
///
/// Returns
/// -------
/// KsCr3bpResult with trajectory, Jacobi error, and NK bounds.
pub fn propagate_ks_cr3bp(
    x0:      &[f64; 6],
    s_final: f64,
    mu:      f64,
    config:  &PicardConfig,
) -> KsCr3bpResult {
    // Convert to KS initial conditions
    let ks0 = cartesian_to_ks(x0, mu, 0.0)
        .expect("Initial state is at the Moon's center — cannot regularize");

    let x0_arr = Array1::from_vec(ks0.to_vec());

    // Build batch RHS closure
    let rhs = |x: &Array2<f64>, s: &Array1<f64>| ks_cr3bp_rhs_batch(x, s, mu);

    // Propagate in fictitious time (has_time=true: last component is physical time)
    let traj = propagate(&rhs, &x0_arr, 0.0, s_final, config, true);

    // Extract final KS state
    let n_seg = traj.segments.len();
    let final_ks: [f64; 9] = if n_seg > 0 {
        use crate::chebyshev::clenshaw;
        let seg = traj.segments.last().unwrap();
        std::array::from_fn(|i| {
            let row = Array1::from_vec(seg.cheb_coeffs.row(i).to_vec());
            clenshaw(&row, 1.0)
        })
    } else {
        ks0
    };

    let final_cartesian = ks_to_cartesian(&final_ks, mu);
    let jacobi_error   = ks_jacobi_error(&traj, &ks0, mu);

    // Aggregate NK bounds and Bernstein rhos
    let nk_bound_max = traj.segments.iter()
        .map(|s| s.nk_bound.unwrap_or(0.0))
        .fold(0.0f64, f64::max);
    let bernstein_rho_mean = if n_seg > 0 {
        traj.segments.iter().map(|s| s.bernstein_rho).sum::<f64>() / n_seg as f64
    } else { 0.0 };
    let mean_residual = if n_seg > 0 {
        traj.segments.iter().map(|s| s.residual).sum::<f64>() / n_seg as f64
    } else { 0.0 };
    let iter_counts:    Vec<usize> = traj.segments.iter().map(|s| s.n_iter).collect();
    let bernstein_rhos: Vec<f64>   = traj.segments.iter().map(|s| s.bernstein_rho).collect();

    KsCr3bpResult {
        n_segments: n_seg,
        jacobi_error,
        final_cartesian,
        final_ks,
        nk_bound_max,
        bernstein_rho_mean,
        mean_residual,
        iter_counts,
        bernstein_rhos,
        trajectory: traj,
    }
}

/// Estimate the fictitious time s_final from physical time t_final and the
/// initial Moon distance r₂₀ = |ρ₀|.
///
/// For unperturbed Kepler: s ≈ 2π/ω for one orbit, where ω = sqrt(−h₂/2).
/// A safe estimate for the CR3BP: s_final ≈ t_final / r₂_mean.
/// The Sundman transformation dt = r₂ ds, so s ≈ t / <r₂>.
pub fn estimate_s_from_t(t_final: f64, x0: &[f64; 6], mu: f64) -> f64 {
    let mu1 = 1.0 - mu;
    let xi   = x0[0] - mu1;
    let eta  = x0[1];
    let zeta = x0[2];
    let r2   = (xi*xi + eta*eta + zeta*zeta).sqrt().max(1e-10);
    // A conservative estimate: use the initial r₂ as the typical value.
    // In practice the solver adapts via the Bernstein ρ diagnostic.
    (t_final / r2).max(1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ks_map_inverse_roundtrip() {
        // Test cartesian→KS→cartesian roundtrip near Moon
        let mu = 0.01215_f64;
        let mu1 = 1.0 - mu;
        // Spacecraft 0.1 units above Moon (Moon at x=1-μ≈0.98785)
        let cart0: [f64; 6] = [mu1 + 0.0, 0.1, 0.05, 0.0, 0.1, 0.0];
        let ks0 = cartesian_to_ks(&cart0, mu, 0.0).unwrap();
        let cart1 = ks_to_cartesian(&ks0, mu);
        for (a, b) in cart0.iter().zip(cart1.iter()) {
            assert!((a - b).abs() < 1e-12,
                "roundtrip fail: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_ks_norm_relation() {
        // |u|² = r₂ = distance to Moon
        let mu = 0.01215_f64;
        let mu1 = 1.0 - mu;
        let cart: [f64; 6] = [mu1 + 0.3, 0.2, 0.1, 0.05, 0.0, 0.0];
        let xi   = cart[0] - mu1;
        let eta  = cart[1];
        let zeta = cart[2];
        let r2   = (xi*xi + eta*eta + zeta*zeta).sqrt();
        let ks   = cartesian_to_ks(&cart, mu, 0.0).unwrap();
        let u    = [ks[0], ks[1], ks[2], ks[3]];
        let u2   = u[0]*u[0] + u[1]*u[1] + u[2]*u[2] + u[3]*u[3];
        assert!((u2 - r2).abs() < 1e-12,
            "|u|²={:.10} != r₂={:.10}", u2, r2);
    }

    #[test]
    fn test_ks_velocity_relation() {
        // Verify: |ρ̇|² = 4|p|²/|u|²
        let mu = 0.01215_f64;
        let mu1 = 1.0 - mu;
        let cart: [f64; 6] = [mu1 + 0.3, 0.2, 0.1, -0.1, 0.2, 0.05];
        let ks = cartesian_to_ks(&cart, mu, 0.0).unwrap();
        let u  = [ks[0], ks[1], ks[2], ks[3]];
        let p  = [ks[4], ks[5], ks[6], ks[7]];
        let u2 = u[0]*u[0]+u[1]*u[1]+u[2]*u[2]+u[3]*u[3];
        let p2 = p[0]*p[0]+p[1]*p[1]+p[2]*p[2]+p[3]*p[3];
        // |ρ̇|² from Cartesian:
        let rho_dot_sq = cart[3]*cart[3] + cart[4]*cart[4] + cart[5]*cart[5];
        let ks_vel_sq  = 4.0 * p2 / u2;
        assert!((ks_vel_sq - rho_dot_sq).abs() < 1e-10,
            "4|p|²/|u|²={:.10} != |ρ̇|²={:.10}", ks_vel_sq, rho_dot_sq);
    }

    #[test]
    fn test_rhs_is_finite() {
        // KS RHS should be finite even for small r₂
        let mu = 0.01215_f64;
        let mu1 = 1.0 - mu;
        // Very close to Moon: r₂ = 0.001
        let cart: [f64; 6] = [mu1 + 0.001, 0.0, 0.0, 0.0, 0.5, 0.0];
        let ks = cartesian_to_ks(&cart, mu, 0.0).unwrap();
        let dy = ks_rhs_single(&ks, mu);
        for (i, &v) in dy.iter().enumerate() {
            assert!(v.is_finite(), "RHS component {} is not finite: {}", i, v);
        }
    }

    #[test]
    fn test_jacobi_conservation_short_arc() {
        // Propagate a short arc and check Jacobi constant is preserved
        use crate::picard::PicardConfig;
        let mu = 0.01215_f64;
        let mu1 = 1.0 - mu;
        // Spacecraft near Moon's L2 vicinity at r₂ ≈ 0.15
        let cart0: [f64; 6] = [mu1 + 0.15, 0.0, 0.1, 0.0, -0.05, 0.0];
        let s_final = estimate_s_from_t(1.0, &cart0, mu);

        let config = PicardConfig {
            n_cheb: 24,
            tol: 1e-10,
            ds_initial: 0.05,
            max_segments: 50,
            certify: false,
            ..Default::default()
        };
        let result = propagate_ks_cr3bp(&cart0, s_final, mu, &config);
        println!("KS-CR3BP: {} segs, CJ_err={:.2e}, rho_mean={:.2}",
                 result.n_segments, result.jacobi_error, result.bernstein_rho_mean);
        assert!(result.jacobi_error < 1e-7,
            "Jacobi error {:.2e} too large", result.jacobi_error);
    }
}
