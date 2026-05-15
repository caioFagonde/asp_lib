# python/asp/physics/floer.py

import numpy as np

def compute_conley_zehnder_index(monodromy_matrix: np.ndarray) -> int:
    """
    Computes the Conley-Zehnder index of a periodic orbit in the CR3BP (Thesis Paper 29).
    
    :param monodromy_matrix: The 6x6 State Transition Matrix evaluated at the period T.
    :return: The integer Conley-Zehnder index \mu_{CZ}.
    """
    if monodromy_matrix.shape != (6, 6):
        raise ValueError("Monodromy matrix must be 6x6 for CR3BP.")
        
    # Extract eigenvalues
    eigenvalues = np.linalg.eigvals(monodromy_matrix)
    
    # Filter for real eigenvalues (phi_j)
    real_eigs = [e.real for e in eigenvalues if abs(e.imag) < 1e-10]
    
    # CZ Index Formula:
    # \mu_{CZ} = (1/2) \sum [sign(phi_j - 1) + sign(phi_j + 1)] + #{j : phi_j < 0}
    
    sum_signs = 0
    negative_count = 0
    
    for phi in real_eigs:
        sum_signs += np.sign(phi - 1.0) + np.sign(phi + 1.0)
        if phi < 0:
            negative_count += 1
            
    mu_cz = int(0.5 * sum_signs) + negative_count
    return mu_cz

def arnold_floer_transfer_bound(mu_1: int, mu_2: int, action_1: float, action_2: float) -> float:
    """
    Computes the topological lower bound on transfer Delta V between two orbits (Theorem 4).
    If the CZ indices differ, the orbits cannot be continuously deformed.
    """
    if mu_1 == mu_2:
        return 0.0 # Can be deformed continuously
        
    return abs(action_1 - action_2)