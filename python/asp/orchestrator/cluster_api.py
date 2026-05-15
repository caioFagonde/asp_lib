# python/asp/orchestrator/cluster_api.py

import grpc
import numpy as np
# Note: Requires generating the pb2 files via protoc:
# python -m grpc_tools.protoc -I../go_services/proto --python_out=. --grpc_python_out=. asp_cluster.proto
try:
    # Try relative import first (works if generated correctly in package)
    from . import asp_cluster_pb2 as pb2
    from . import asp_cluster_pb2_grpc as pb2_grpc
except ImportError:
    try:
        # Fallback to absolute import
        import asp_cluster_pb2 as pb2
        import asp_cluster_pb2_grpc as pb2_grpc
    except ImportError:
        pb2 = None
        pb2_grpc = None
class ClusterDispatcher:
    """
    gRPC Client to dispatch massive trajectory batches to the Go microservices
    for Arbitrary Polynomial Chaos (APC).
    """
    def __init__(self, host: str = "localhost", port: int = 50051):
        if pb2 is None:
            raise ImportError("gRPC protobuf files not generated. Run protoc first.")
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = pb2_grpc.OrbitDispatchStub(self.channel)

    def propagate_batch(self, initial_states: list[list[float]], t_final: float, mu: float = 0.01215):
        """
        Streams a batch of initial conditions to the Go cluster.
        """
        states = [pb2.StateVector(components=st) for st in initial_states]
        
        req = pb2.PropagateBatchRequest(
            job_id="apc_batch_01",
            initial_states=states,
            t_final=t_final,
            mu=mu,
            n_cheb=32,
            tolerance=1e-11
        )
        
        print(f"Dispatching {len(initial_states)} trajectories to cluster...")
        response = self.stub.PropagateBatch(req)
        
        return [{
            "final_state": list(res.final_state),
            "jacobi_error": res.jacobi_error,
            "nk_bound": res.nk_bound,
            "is_certified": res.is_certified
        } for res in response.results]