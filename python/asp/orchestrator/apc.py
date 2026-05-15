import numpy as np
import asp._asp_core as asp_core
from asp.orchestrator.cluster_api import ClusterDispatcher
 
 
class ArbitraryPolynomialChaos:
    """
    Arbitrary Polynomial Chaos (APC) engine for orbital uncertainty quantification.
    Constructs orthogonal polynomial bases from raw statistical moments using
    the Hankel-Cholesky / Golub-Welsch algorithm, then dispatches to the
    Go/Rust gRPC cluster for distributed trajectory propagation.
    """
 
    def __init__(self, moments: list[float], n_points: int):
        if len(moments) < 2 * n_points:
            raise ValueError(
                f"APC requires ≥ 2N moments for N collocation points. "
                f"Got {len(moments)}, need {2 * n_points}."
            )
        self.moments  = moments
        self.n_points = n_points
        self.mu_0     = moments[0]          # total probability mass
        if abs(self.mu_0) < 1e-300:
            raise ValueError("Zeroth moment μ₀ = 0 — distribution has zero mass.")
 
        nodes_raw, weights_raw = asp_core.compute_apc_collocation(
            self.moments, self.n_points
        )
        self.nodes   = list(nodes_raw)
        self.weights = list(weights_raw)    # Σwᵢ = μ₀
 
    def generate_collocation_states(
        self,
        nominal_state: list[float],
        uncertainty_idx: int,
    ) -> list[list[float]]:
        states = []
        for node in self.nodes:
            st = list(nominal_state)
            st[uncertainty_idx] += node
            states.append(st)
        return states
 
    def propagate_and_aggregate(
        self,
        nominal_state: list[float],
        uncertainty_idx: int,
        t_final: float,
        mu_cr3bp: float = 0.01215,
        host: str = "localhost",
    ) -> dict:
        """
        Dispatch collocation trajectories and compute output statistics.
 
        Correct normalised quadrature (Issue 12 fix):
            E[f]   = (1/μ₀) Σ wᵢ · f(xᵢ)
            E[f²]  = (1/μ₀) Σ wᵢ · f(xᵢ)²
            Var[f] = E[f²] − (E[f])²
        """
        states = self.generate_collocation_states(nominal_state, uncertainty_idx)
        dispatcher = ClusterDispatcher(host=host)
        results = dispatcher.propagate_batch(states, t_final, mu=mu_cr3bp)
 
        n_state       = len(nominal_state)
        expected_state = np.zeros(n_state)
        expected_sq    = np.zeros(n_state)
        all_certified  = True
        max_nk_bound   = 0.0
 
        for i, res in enumerate(results):
            f_i = np.array(res["final_state"], dtype=np.float64)
            w_i = self.weights[i]
            expected_state += w_i * f_i
            expected_sq    += w_i * (f_i ** 2)
            if not res["is_certified"]:
                all_certified = False
            max_nk_bound = max(max_nk_bound, res["nk_bound"])
 
        # Divide by μ₀ to obtain true expectations (fixes Issue 12)
        expected_state /= self.mu_0
        expected_sq    /= self.mu_0
 
        # Clamp numerical noise that could produce tiny negative values near 0
        variance = np.maximum(expected_sq - expected_state ** 2, 0.0)
 
        return {
            "expected_state": expected_state.tolist(),
            "variance":       variance.tolist(),
            "is_certified":   all_certified,
            "max_nk_bound":   max_nk_bound,
        }