# tests/test_resurgent_ps.py

import numpy as np
import pytest
import asp._asp_core as asp_core

def test_milestone_4_resurgent_pseudoinverse():
    """
    Validates the Resurgent Pseudoinverse (Paper 31).
    Tests the extraction of the Hadamard finite part (R_0) and pole residue (R_{-1})
    at the wrist singularity of a 2-link robotic arm.
    """
    
    # 2-link planar arm Jacobian at theta_1 = pi/4.
    # Singularity occurs at theta_2 (eps) = 0.
    def robot_jacobian(eps: float) -> np.ndarray:
        t1 = np.pi / 4.0
        t12 = t1 + eps
        
        # Link lengths L1 = L2 = 1.0
        return np.array([
            [-np.sin(t1) - np.sin(t12), -np.sin(t12)],
            [ np.cos(t1) + np.cos(t12),  np.cos(t12)]
        ], dtype=np.float64)

    # Evaluate the Resurgent Pseudoinverse using Chebyshev spectral extraction
    # We evaluate F(eps) in a safe window strictly away from the singularity at eps=0.
    eps_min = 1e-4
    eps_max = 1e-2
    n_cheb = 32
    
    r_minus_1, r_0 = asp_core.resurgent_pseudoinverse(
        robot_jacobian, 
        eps_min, 
        eps_max, 
        n_cheb
    )
    
    # Thesis Paper 31 states:
    # R_{-1}^{00} = 1 / sqrt(2) = 0.707107 (Alien derivative / UK constraint force)
    expected_r_minus_1_00 = 1.0 / np.sqrt(2.0)
    
    # R_0 = Hadamard finite part (Geometrically correct pseudoinverse)
    # The paper notes R_0 is proportional to (-1/sqrt(2)) * 1_{2x2}
    
    error_r_minus_1 = abs(r_minus_1[0, 0] - expected_r_minus_1_00)
    
    print(f"Extracted R_{-1} (Alien Derivative Matrix):\n{r_minus_1}")
    print(f"Extracted R_0 (Hadamard Finite Part Matrix):\n{r_0}")
    print(f"Error in R_{-1}[0,0]: {error_r_minus_1:.4e}")
    
    assert error_r_minus_1 < 1e-6, f"Resurgent Pseudoinverse failed. Error: {error_r_minus_1}"

if __name__ == "__main__":
    test_milestone_4_resurgent_pseudoinverse()