// build.rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("go_services/proto/asp_cluster.proto")?;
    Ok(())
}