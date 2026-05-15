# python/asp/navigation/tad_cpe.py

import numpy as np
import asp._asp_core as asp_core

class ConjunctionAnalyzer:
    """
    Orchestrates exact orbital collision probabilities via the Thermal Alien Derivative.
    """
    
    @staticmethod
    def compute_probability_2d(r_miss: np.ndarray, covariance: np.ndarray, r_hbr: float) -> dict:
        """
        Computes the Probability of Collision (Pc) in the 2D B-plane.
        
        :param r_miss: 2D nominal miss distance vector [x, y].
        :param covariance: 2x2 combined relative covariance matrix.
        :param r_hbr: Hard body radius (combined object radius).
        :return: Dictionary containing Pc, Freidlin-Wentzell Action, and Stokes Constant.
        """
        if r_miss.shape != (2,) or covariance.shape != (2, 2):
            raise ValueError("TAD-CPE 2D requires 2D vectors and 2x2 matrices.")
            
        # Delegate the exact mathematical computation to the Rust core
        p_c, a_fw, s_star = asp_core.compute_tad_cpe_2d(
            r_miss.astype(np.float64), 
            covariance.astype(np.float64), 
            float(r_hbr)
        )
        
        return {
            "probability_of_collision": p_c,
            "freidlin_wentzell_action": a_fw,
            "stokes_constant": s_star,
            "regime": "Large Deviation" if a_fw > 1.0 else "High Risk"
        }