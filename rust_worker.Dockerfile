FROM rust:1.78-slim AS builder

WORKDIR /usr/src/asp

RUN apt-get update && apt-get install -y --no-install-recommends \
        protobuf-compiler libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

# Layer-cache dependencies before source
COPY Cargo.toml Cargo.lock build.rs ./
RUN mkdir src && echo "fn main() {}" > src/lib.rs \
    && cargo build --locked --release --bin worker 2>/dev/null; rm -rf src

# Now copy actual source and rebuild (only changed crates recompile)
COPY src ./src
COPY go_services/proto ./go_services/proto
RUN touch src/lib.rs && cargo build --locked --release --bin worker

FROM debian:bookworm-slim AS runtime
RUN groupadd -r aspworker && useradd -r -g aspworker aspworker
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/src/asp/target/release/worker ./worker
RUN chown aspworker:aspworker ./worker

USER aspworker
EXPOSE 50052
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD ["/bin/sh", "-c", "kill -0 1"]
CMD ["./worker"]