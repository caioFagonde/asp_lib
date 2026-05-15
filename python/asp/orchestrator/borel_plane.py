# python/asp/orchestrator/borel_plane.py

import numpy as np
import asp._asp_core as asp_core

class BorelPlaneAnalyzer:
    """
    Analyzes the Borel plane of a factorially divergent perturbative series.
    Executes blind singularity detection via the Rust Borel-Padé engine to 
    extract topological instanton actions and non-perturbative bounds.
    """
    def __init__(self, coeffs: list[float]):
        """
        :param coeffs: The perturbative coefficients a_n (e.g., Bender-Wu or Lindstedt).
        """
        self.coeffs = coeffs

    def extract_instanton_actions(self) -> list[float]:
        """
        Extracts the Borel singularities (instanton actions) from the series.
        """
        return asp_core.extract_singularities_py(self.coeffs)

    def compute_nekhoroshev_bounds(self, K: float, K_c: float) -> dict:
        """
        Computes the Constructive Nekhoroshev Stability Bounds (Thesis Paper 18).
        
        :param K: The current coupling parameter (e.g., stochastic perturbation size).
        :param K_c: The critical coupling where KAM tori break (e.g., 0.971635).
        :return: Dictionary containing stability metrics.
        """
        poles = self.extract_instanton_actions()
        
        # The relevant physical instanton action is the nearest positive real pole
        physical_poles = [p for p in poles if p > 0]
        if not physical_poles:
            raise ValueError("No physical instanton actions found in the Borel plane.")
            
        S_inst = min(physical_poles)
        
        epsilon = K - K_c
        
        # If below the critical coupling, the system is KAM-bounded (infinite stability)
        if epsilon <= 0:
            return {
                "regime": "KAM-bounded",
                "S_inst": S_inst,
                "peierls_barrier": float('inf'),
                "t_nekhoroshev": float('inf'),
                "arnold_diffusion_rate": 0.0
            }
            
        # Above the critical coupling, Arnold diffusion dominates
        # Peierls' barrier P(omega) = exp(-S_inst / (K - K_c))
        peierls_barrier = np.exp(-S_inst / epsilon)
        
        # Nekhoroshev stability time T_Nek = exp(exp(S_inst / (K - K_c)))
        # Using float64, this double exponential will overflow for small epsilon, 
        # which accurately reflects the "super-exponential" stability of the standard map.
        try:
            t_nekhoroshev = np.exp(np.exp(S_inst / epsilon))
        except OverflowError:
            t_nekhoroshev = float('inf')
            
        # Arnold diffusion rate Gamma = pi * S_inst
        arnold_diffusion_rate = np.pi * S_inst
        
        return {
            "regime": "Arnold diffusion",
            "S_inst": S_inst,
            "peierls_barrier": peierls_barrier,
            "t_nekhoroshev": t_nekhoroshev,
            "arnold_diffusion_rate": arnold_diffusion_rate
        }