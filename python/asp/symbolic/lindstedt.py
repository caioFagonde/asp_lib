import math           # standard-library math.factorial — always available
import numpy as np
 
 
def generate_mock_lindstedt_series(
    n_orders: int,
    K_c: float = 0.971635,
    S_inst: float = 6.28,
) -> list[float]:
    """
    Generates a mock Lindstedt norm series for the Standard Map golden-mean torus.
 
    Produces a Gevrey-1 series guaranteed to have a Borel pole at S_inst,
    mimicking the exact output of the recursive Fourier scheme (Paper 18 & 36).
    """
    c = 1.0 / S_inst
    return [
        (c ** k) * float(math.factorial(k))   # Issue 13 fix: math.factorial
        for k in range(n_orders)
    ]