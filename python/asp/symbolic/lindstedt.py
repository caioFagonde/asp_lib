# python/asp/symbolic/lindstedt.py

import numpy as np

def generate_mock_lindstedt_series(n_orders: int, K_c: float = 0.971635, S_inst: float = 6.28) -> list[float]:
    """
    Generates a mock Lindstedt norm series for the Standard Map golden-mean torus.
    
    In a full production environment, this module utilizes SymPy and FFT-based 
    polynomial algebras to exactly compute the cohomological equations:
    D(n) u_k^n = F_k^n
    
    For the architectural milestone, we generate a synthetic Gevrey-1 series 
    that mathematically guarantees a Borel pole at S_inst, mimicking the exact 
    output of the recursive Fourier scheme (Thesis Paper 18 & 36).
    """
    coeffs = []
    for k in range(n_orders):
        # Gevrey-1 growth: a_k ~ c^k * k! where c = 1 / S_inst
        # We add alternating signs or specific phases depending on the exact topology,
        # but for a dominant real positive pole, the coefficients grow factorially.
        c = 1.0 / S_inst
        val = (c ** k) * float(np.math.factorial(k))
        coeffs.append(val)
        
    return coeffs