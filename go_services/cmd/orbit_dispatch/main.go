package main

import (
	"context"
	"log"
	"net"
	"os"
	"time"

	pb "asp_cluster/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type dispatchServer struct {
	pb.UnimplementedOrbitDispatchServer
	workerClient pb.OrbitDispatchClient
}

// PropagateBatch receives a batch of ICs and forwards them to the Rust worker cluster
func (s *dispatchServer) PropagateBatch(ctx context.Context, req *pb.PropagateBatchRequest) (*pb.PropagateBatchResponse, error) {
	log.Printf("Orchestrator received Job %s. Forwarding %d trajectories to Rust worker...", req.JobId, len(req.InitialStates))

	workerCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	// Forward the request to the Rust worker node
	// In a full production system, this would chunk the array and fan out to multiple workers
	res, err := s.workerClient.PropagateBatch(workerCtx, req)
	if err != nil {
		log.Printf("Error calling Rust worker: %v", err)
		return nil, err
	}

	log.Printf("Job %s completed by Rust worker.", req.JobId)
	return res, nil
}

func main() {
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	workerURL := os.Getenv("RUST_WORKER_URL")
	if workerURL == "" {
		workerURL = "localhost:50052"
	}

	// Connect to the Rust gRPC worker
	log.Printf("Dialing Rust worker at %s...", workerURL)
	conn, err := grpc.Dial(workerURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to dial Rust worker: %v", err)
	}
	defer conn.Close()

	workerClient := pb.NewOrbitDispatchClient(conn)

	grpcServer := grpc.NewServer()
	pb.RegisterOrbitDispatchServer(grpcServer, &dispatchServer{workerClient: workerClient})

	log.Println("ASP Orbit Dispatch Service listening on :50051")
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}