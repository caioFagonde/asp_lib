// src/bin/worker.rs
//
// ASP Rust Worker Node
// ======================================================================
// Headless gRPC server executing parallelized Picard integrations.

use tonic::{transport::Server, Request, Response, Status};
use ndarray::Array1;
use rayon::prelude::*;

use asp_core::solvers::picard::{propagate_cr3bp, PicardConfig, jacobi_constant};

// Import the generated protobuf code
pub mod asp_cluster {
    tonic::include_proto!("asp.cluster");
}

use asp_cluster::orbit_dispatch_server::{OrbitDispatch, OrbitDispatchServer};
use asp_cluster::{PropagateBatchRequest, PropagateBatchResponse, TrajectoryResult};

#[derive(Debug, Default)]
pub struct AspWorker {}

#[tonic::async_trait]
impl OrbitDispatch for AspWorker {
    async fn propagate_batch(
        &self,
        request: Request<PropagateBatchRequest>,
    ) -> Result<Response<PropagateBatchResponse>, Status> {
        let req = request.into_inner();
        let mu = req.mu;
        let t_final = req.t_final;
        let config = PicardConfig {
            n_cheb: req.n_cheb as usize,
            tol: req.tolerance,
            certify: true,
            ..Default::default()
        };
        let initial_states = req.initial_states.clone();
        let job_id = req.job_id.clone();

        // Offload Rayon CPU work to a dedicated blocking thread pool
        let results = tokio::task::spawn_blocking(move || {
            let pool = rayon::ThreadPoolBuilder::new().build().unwrap();
            pool.install(|| {
                initial_states
                    .into_par_iter()
                    .map(|state_vec| {
                        let x0 = Array1::from_vec(state_vec.components.clone());
                        let cj0 = jacobi_constant(&x0, mu);
                        let (traj, cj_err) = propagate_cr3bp(&x0, t_final, mu, &config);
                        let final_state = traj.final_state().map(|a| a.to_vec()).unwrap_or_default();
                        let nk_bound_max = traj.segments.iter()
                            .map(|s| s.nk_bound.unwrap_or(0.0))
                            .fold(0.0f64, f64::max);
                        TrajectoryResult {
                            final_state,
                            jacobi_error: cj_err,
                            nk_bound: nk_bound_max,
                            is_certified: nk_bound_max > 0.0 && nk_bound_max < 1.0,
                            segments_used: traj.total_segments() as i32,
                        }
                    })
                    .collect::<Vec<_>>()
            })
        })
        .await
        .map_err(|e| Status::internal(format!("Worker panic: {e}")))?;

        Ok(Response::new(PropagateBatchResponse { job_id, results }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "0.0.0.0:50052".parse()?;
    let worker = AspWorker::default();

    println!("ASP Rust Worker listening on {}", addr);

    Server::builder()
        .add_service(OrbitDispatchServer::new(worker))
        .serve(addr)
        .await?;

    Ok(())
}