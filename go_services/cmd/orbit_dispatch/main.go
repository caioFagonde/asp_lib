package main

import (
	"context"
	"log"
	"net"
	"sync"

	pb "asp_cluster/pb" // Assumes protoc generation to this path
	"google.golang.org/grpc"
)

type dispatchServer struct {
	pb.UnimplementedOrbitDispatchServer
}

// PropagateBatch receives a massive batch of ICs and fans them out to Rust workers
func (s *dispatchServer) PropagateBatch(ctx context.Context, req *pb.PropagateBatchRequest) (*pb.PropagateBatchResponse, error) {
	log.Printf("Received Job %s with %d trajectories", req.JobId, len(req.InitialStates))

	// In a full implementation, this channels data to a Redis queue or directly to Rust gRPC workers.
	// Here we simulate the fan-out and aggregation.
	var wg sync.WaitGroup
	results := make([]*pb.TrajectoryResult, len(req.InitialStates))

	for i, state := range req.InitialStates {
		wg.Add(1)
		go func(idx int, st *pb.StateVector) {
			defer wg.Done()
			// Mocking the Rust worker response for architectural skeleton
			results[idx] = &pb.TrajectoryResult{
				FinalState:   st.Components, // Mock
				JacobiError:  1e-13,
				NkBound:      1e-11,
				IsCertified:  true,
				SegmentsUsed: 42,
			}
		}(i, state)
	}

	wg.Wait()
	log.Printf("Job %s completed.", req.JobId)

	return &pb.PropagateBatchResponse{
		JobId:   req.JobId,
		Results: results,
	}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterOrbitDispatchServer(grpcServer, &dispatchServer{})

	log.Println("ASP Orbit Dispatch Service listening on :50051")
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}