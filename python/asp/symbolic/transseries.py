# python/asp/symbolic/transseries.py

import numpy as np
import math
import asp._asp_core as asp_core

class Transseries:
    r"""
    Algebraic representation of a formal resurgent transseries:
    \Phi(z; \sigma) = \sum_{k=0}^\infty \sigma^k e^{-k A / z} \tilde{\phi}_k(z)
    """
    def __init__(self, perturbative_coeffs: list[float], instanton_action: float = None):
        self.phi_0 = perturbative_coeffs
        
        if instanton_action is None:
            poles = asp_core.extract_singularities_py(self.phi_0)
            physical_poles = [p for p in poles if p > 0]
            if not physical_poles:
                raise ValueError("No physical Borel singularity detected in the perturbative sector.")
            self.action = min(physical_poles)
        else:
            self.action = instanton_action

    def median_resummation(self, z: float, epsilon: float = 1e-5, n_gl: int = 150) -> float:
        r"""
        Executes the Measure operator (\mathcal{M}).
        Computes the Cauchy Principal Value of the ambiguous lateral Borel sums
        exactly on the Stokes line, projecting the ambiguous analytic continuation
        onto the deterministic physical reality.
        
        :param z: The physical coupling parameter.
        :param epsilon: The contour rotation angle (radians) to bypass the Stokes line.
        :param n_gl: Number of Gauss-Laguerre quadrature nodes.
        """
        return asp_core.median_resummation_py(self.phi_0, z, epsilon, n_gl)

    def extract_alien_derivative(self) -> float:
        r"""
        Executes the Extract operator (\Delta_\omega).
        Returns the Stokes constant S_1 associated with the leading singularity.
        """
        n = len(self.phi_0) - 1
        a_n = self.phi_0[n]
        
        # Asymptotic growth: a_n \sim (S_1 / 2\pi) * \Gamma(n) / A^n
        S_1_mag = abs(a_n) * 2.0 * np.pi * (self.action ** n) / math.factorial(n - 1)
        return S_1_mag