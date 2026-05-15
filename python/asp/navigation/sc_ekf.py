import numpy as np
import asp._asp_core as asp_core
 
 
class SpectralChebyshevEKF:
    """
    Spectral-Chebyshev Extended Kalman Filter (SC-EKF).
    Replaces discrete-time numerical integration with exact analytical
    evaluation of globally continuous Chebyshev polynomials. Eliminates
    linearization truncation error during the time-update phase (Paper 8).
    """
 
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self.stm_dim   = state_dim * state_dim
        self.t_k       = 0.0
        self.x_k       = np.zeros(state_dim)
        self.P_k       = np.eye(state_dim)
        self.current_segment_coeffs = None
        self.segment_t_start = 0.0
        self.segment_t_end   = 0.0
 
    def load_reference_segment(
        self, coeffs: np.ndarray, t_start: float, t_end: float
    ):
        expected_dim = self.state_dim + self.stm_dim
        if coeffs.shape[0] != expected_dim:
            raise ValueError(
                f"Expected {expected_dim} augmented-state dimensions, got {coeffs.shape[0]}"
            )
        self.current_segment_coeffs = coeffs
        self.segment_t_start = t_start
        self.segment_t_end   = t_end
 
    def _map_time_to_tau(self, t_obs: float) -> float:
        dt = self.segment_t_end - self.segment_t_start
        if dt <= 0:
            raise ValueError("Invalid segment time bounds.")
        return max(-1.0, min(1.0, 2.0 * (t_obs - self.segment_t_start) / dt - 1.0))
 
    def predict(
        self,
        t_obs: float,
        Q_process_noise: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Continuous-time prediction. Evaluates exact state and STM analytically
        via Clenshaw's algorithm — zero numerical integration required.
        """
        if self.current_segment_coeffs is None:
            raise RuntimeError("Reference segment not loaded. Run Picard solver first.")
        tau_obs = self._map_time_to_tau(t_obs)
        augmented = asp_core.evaluate_clenshaw(self.current_segment_coeffs, tau_obs)
        x_pred  = augmented[: self.state_dim]
        phi_vec = augmented[self.state_dim :]
        Phi     = phi_vec.reshape((self.state_dim, self.state_dim))
        P_pred  = Phi @ self.P_k @ Phi.T + Q_process_noise
        return x_pred, P_pred
 
    def update(
        self,
        z_obs: np.ndarray,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Standard measurement update with numerically stable innovation solve.
 
        Issue 14 fix: replace np.linalg.inv(S) with np.linalg.solve(S, ...)
        to avoid inverting the innovation covariance directly. For a p×p matrix
        the Cholesky-based solve in linalg.solve is O(p³/3) vs O(p³) for a
        general inverse, and is numerically stable when S is near-singular.
        """
        S = H @ P_pred @ H.T + R   # p×p innovation covariance
 
        # K = P Hᵀ S⁻¹  →  Kᵀ = S⁻ᵀ H P = S⁻¹ H P  (S is symmetric)
        # Solve S·Kᵀ = H·P for Kᵀ, then transpose.
        K_T = np.linalg.solve(S, H @ P_pred)   # p×n
        K   = K_T.T                             # n×p
 
        innovation = z_obs - (H @ x_pred)
        self.x_k   = x_pred + K @ innovation
        I          = np.eye(self.state_dim)
        self.P_k   = (I - K @ H) @ P_pred
        self.t_k   = self.segment_t_end
        return self.x_k, self.P_k