# python/asp/navigation/sc_ekf.py

import numpy as np
import asp._asp_core as asp_core

class SpectralChebyshevEKF:
    """
    Spectral-Chebyshev Extended Kalman Filter (SC-EKF).
    Replaces discrete-time numerical integration with exact analytical evaluation
    of globally continuous Chebyshev polynomials. Eliminates linearization truncation
    error during the time-update phase (ASP Paper 8).
    """
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self.stm_dim = state_dim * state_dim
        
        # Current filter estimates
        self.t_k = 0.0
        self.x_k = np.zeros(state_dim)
        self.P_k = np.eye(state_dim)
        
        # Continuous trajectory data (populated by ASP-UK-Res Solver)
        # Represents the augmented state [x, vec(Phi)]
        self.current_segment_coeffs = None
        self.segment_t_start = 0.0
        self.segment_t_end = 0.0

    def load_reference_segment(self, coeffs: np.ndarray, t_start: float, t_end: float):
        """
        Loads the Chebyshev coefficients representing the augmented state
        [x(t), vec(Phi(t, t_k))] computed by the rigorous Rust Picard solver.
        """
        expected_dim = self.state_dim + self.stm_dim
        if coeffs.shape[0] != expected_dim:
            raise ValueError(f"Expected {expected_dim} dimensions for augmented state, got {coeffs.shape[0]}")
            
        self.current_segment_coeffs = coeffs
        self.segment_t_start = t_start
        self.segment_t_end = t_end

    def _map_time_to_tau(self, t_obs: float) -> float:
        """Maps physical time t to the Chebyshev domain tau in [-1, 1]."""
        dt = self.segment_t_end - self.segment_t_start
        if dt <= 0:
            raise ValueError("Invalid segment time bounds.")
        tau = 2.0 * (t_obs - self.segment_t_start) / dt - 1.0
        return max(-1.0, min(1.0, tau))

    def predict(self, t_obs: float, Q_process_noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Continuous-Time Prediction (Time Update).
        Evaluates the exact analytical state and STM at the observation time
        using Clenshaw's algorithm. Zero numerical integration required here.
        """
        if self.current_segment_coeffs is None:
            raise RuntimeError("Reference segment not loaded. Run Picard solver first.")
            
        tau_obs = self._map_time_to_tau(t_obs)
        
        # Evaluate augmented state [x, vec(Phi)] via Rust FFI
        augmented_state = asp_core.evaluate_clenshaw(self.current_segment_coeffs, tau_obs)
        
        x_pred = augmented_state[:self.state_dim]
        phi_vec = augmented_state[self.state_dim:]
        Phi = phi_vec.reshape((self.state_dim, self.state_dim))
        
        # Propagate Covariance: P_{k+1|k} = \Phi P_{k|k} \Phi^T + Q
        P_pred = Phi @ self.P_k @ Phi.T + Q_process_noise
        
        return x_pred, P_pred

    def update(self, z_obs: np.ndarray, x_pred: np.ndarray, P_pred: np.ndarray, H: np.ndarray, R: np.ndarray):
        """
        Standard Measurement Update.
        """
        # Innovation covariance: S = H P H^T + R
        S = H @ P_pred @ H.T + R
        
        # Kalman Gain: K = P H^T S^{-1}
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # State update
        innovation = z_obs - (H @ x_pred)
        self.x_k = x_pred + K @ innovation
        
        # Covariance update: P = (I - K H) P
        I = np.eye(self.state_dim)
        self.P_k = (I - K @ H) @ P_pred
        self.t_k = self.segment_t_end # Advance filter epoch
        
        return self.x_k, self.P_k