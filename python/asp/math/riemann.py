# python/asp/math/riemann.py

import numpy as np
import asp._asp_core as asp_core

def compute_zeta_spectral_resolvent(known_zeros: list[float], n_moments: int = 20) -> list[float]:
    """
    Constructs the generating series Z(w) from the Riemann zeros (Thesis Paper 17).
    
    :param known_zeros: The imaginary parts of the Riemann zeros (gamma_k).
    :param n_moments: Number of spectral moments to compute.
    :return: The coefficients sigma_{m+1} of the spectral resolvent.
    """
    gammas = np.array(known_zeros, dtype=np.float64)
    
    sigma = []
    for m in range(1, n_moments + 1):
        # \sigma_m = \sum_k \gamma_k^{-2m}
        s_m = np.sum(gammas ** (-2 * m))
        sigma.append(s_m)
        
    return sigma

def extract_riemann_zeros(sigma_coeffs: list[float]) -> list[float]:
    """
    Uses the Borel-Padé engine to extract the Riemann zeros from the spectral moments.
    The Padé poles of Z(w) converge to gamma_k^2.
    """
    # The ASP core robust_pade expects the coefficients.
    # We use extract_singularities_py which finds the roots of the Padé denominator.
    poles = asp_core.extract_singularities_py(sigma_coeffs)
    
    # The poles are w = gamma_k^2. We take the square root to get gamma_k.
    extracted_zeros = [np.sqrt(p) for p in poles if p > 0]
    return sorted(extracted_zeros)