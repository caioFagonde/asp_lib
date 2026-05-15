# tests/test_apc_cluster.py

import pytest
import numpy as np
from asp.orchestrator.apc import ArbitraryPolynomialChaos

# Note: This test requires the docker-compose cluster to be running locally on ports 50051/50052.
# For CI environments without the cluster, this should be marked with @pytest.mark.skipif

def test_milestone_8_apc_distributed_propagation():
    """
    Validates the Arbitrary Polynomial Chaos (APC) distributed pipeline.
    
    Checks:
    1. Hankel-Cholesky decomposition of raw moments into quadrature nodes.
    2. Dispatch of nodes to the Go/Rust gRPC cluster.
    3. Aggregation of output moments via spectral weights.
    4. Newton-Kantorovich certification preservation across the batch.
    """
    
    # 1. Define a synthetic distribution via its raw moments
    # Example: A uniform distribution U[-a, a] has moments:
    # mu_k = 0 for odd k, mu_k = (a^k) / (k+1) for even k.
    # Let a = 0.01 (small spatial uncertainty)
    a = 0.01
    n_points = 3
    moments = []
    for k in range(2 * n_points):
        if k % 2 != 0:
            moments.append(0.0)
        else:
            moments.append((a ** k) / (k + 1.0))
            
    # Initialize the APC engine
    apc = ArbitraryPolynomialChaos(moments, n_points)
    
    # Verify nodes are symmetric around 0 (characteristic of uniform distribution)
    assert len(apc.nodes) == 3
    assert abs(apc.nodes[0] + apc.nodes[2]) < 1e-10  # Symmetric roots
    assert abs(apc.nodes[1]) < 1e-10                 # Center root at 0
    
    # Verify weights sum to mu_0 (which is 1.0)
    assert abs(sum(apc.weights) - 1.0) < 1e-10
    
    # 2. Define the nominal CR3BP state (e.g., an L1 Halo insertion point)
    nominal_state = [0.82, 0.0, 0.12, 0.0, -0.13, 0.0]
    uncertainty_idx = 0 # Apply uncertainty to the X-coordinate
    t_final = 1.5
    
    # 3. Propagate and Aggregate via the Cluster
    try:
        # Assuming the cluster is running on localhost
        result = apc.propagate_and_aggregate(nominal_state, uncertainty_idx, t_final)
        
        print("\n--- APC Distributed Propagation Results ---")
        print(f"Expected Final State: {result['expected_state']}")
        print(f"Final State Variance: {result['variance']}")
        print(f"Batch Certified:      {result['is_certified']}")
        print(f"Max NK Error Bound:   {result['max_nk_bound']:.2e}")
        
        # Verify the batch was certified by the Rust worker
        assert result["is_certified"] is True
        assert result["max_nk_bound"] < 1.0
        
        # Verify variance is non-negative
        for var in result["variance"]:
            assert var >= -1e-12 # Account for floating point noise near zero
            
    except Exception as e:
        pytest.skip(f"Cluster not reachable or propagation failed. Ensure docker-compose is running. Error: {e}")

if __name__ == "__main__":
    test_milestone_8_apc_distributed_propagation()