# python/asp/symbolic/bender_wu.py

import numpy as np

def bender_wu_quartic(N: int) -> list[float]:
    """
    Computes the exact perturbative energy coefficients a_n for the 
    quartic anharmonic oscillator V(x) = x^2/2 + g x^4 using the 
    Bender-Wu algebraic recursion (Thesis Chapter 2).
    
    The energy expansion is E(g) = sum_{n=0}^N a_n g^n.
    
    :param N: Number of coefficients to generate.
    :return: List of coefficients [a_0, a_1, ..., a_N]
    """
    if N < 1:
        return []

    # C[n][k] stores the coefficient of the k-th Fock state at the n-th order of perturbation
    # Max k needed at order n is 4*n
    C = np.zeros((N, 4 * N + 1), dtype=np.float64)
    a = np.zeros(N, dtype=np.float64)

    # Base case: unperturbed harmonic oscillator ground state
    a[0] = 0.5
    C[0][0] = 1.0

    # Matrix elements of x^4 in the harmonic oscillator Fock basis <k|x^4|m>
    def x4_matrix_element(k, m):
        if k == m:
            return 0.75 * (2 * m**2 + 2 * m + 1)
        elif abs(k - m) == 2:
            min_idx = min(k, m)
            return (2 * min_idx + 3) * np.sqrt((min_idx + 1) * (min_idx + 2)) / 4.0
        elif abs(k - m) == 4:
            min_idx = min(k, m)
            return np.sqrt((min_idx + 1) * (min_idx + 2) * (min_idx + 3) * (min_idx + 4)) / 4.0
        return 0.0

    for n in range(1, N):
        # 1. Compute the energy correction a_n = <0|x^4|psi^{(n-1)}>
        a[n] = sum(x4_matrix_element(0, m) * C[n-1][m] for m in range(4 * (n - 1) + 1))
        
        # 2. Compute the state corrections C_k^{(n)}
        for k in range(2, 4 * n + 1, 2): # Only even states contribute
            sum_matrix = sum(x4_matrix_element(k, m) * C[n-1][m] for m in range(4 * (n - 1) + 1))
            sum_energy = sum(a[j] * C[n-j][k] for j in range(1, n))
            
            C[n][k] = -(sum_matrix - sum_energy) / float(k)

    return a.tolist()