#!/usr/bin/env bash
# install.sh - Automated dependency provisioning and build script for ASP Core
# Strictly fails on any error.
set -e
set -o pipefail

echo "============================================================"
echo " Analytic Structural Physics (ASP) - Build & Provisioning"
echo "============================================================"

# 1. Detect OS and Install System Dependencies
OS="$(uname -s)"
echo "[*] Operating System detected: $OS"

if [ "$OS" = "Linux" ]; then
    echo "[*] Installing system dependencies via apt-get..."
    sudo apt-get update -y
    sudo apt-get install -y build-essential curl protobuf-compiler libprotobuf-dev
elif [ "$OS" = "Darwin" ]; then
    echo "[*] Installing system dependencies via Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "[ERROR] Homebrew not found. Please install Homebrew first."
        exit 1
    fi
    brew install protobuf
else
    echo "[WARNING] Unsupported OS for automated system dependency installation. Ensure protoc is installed."
fi

# 2. Rust Toolchain
if ! command -v cargo &> /dev/null; then
    echo "[*] Rust not found. Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "[*] Rust toolchain detected. Updating..."
    rustup update stable
fi

# 3. Go Toolchain
if ! command -v go &> /dev/null; then
    echo "[ERROR] Go is not installed. Please install Go 1.21+ from https://go.dev/dl/"
    exit 1
else
    GO_VERSION=$(go version | awk '{print $3}')
    echo "[*] Go toolchain detected: $GO_VERSION"
fi

# 4. Python Virtual Environment
echo "[*] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "[*] Upgrading pip and installing Python dependencies..."
python -m pip install --upgrade pip
pip install -e .[test]

# 5. Generate gRPC Protobuf Stubs
echo "[*] Generating Python gRPC stubs from Protobuf definitions..."
python -m grpc_tools.protoc -Igo_services/proto \
    --python_out=python/asp/orchestrator \
    --grpc_python_out=python/asp/orchestrator \
    go_services/proto/asp_cluster.proto

echo "[*] Patching Python 3 relative imports in generated gRPC code..."
if [ "$OS" = "Darwin" ]; then
    sed -i '' 's/^import asp_cluster_pb2/from . import asp_cluster_pb2/' python/asp/orchestrator/asp_cluster_pb2_grpc.py
else
    sed -i 's/^import asp_cluster_pb2/from . import asp_cluster_pb2/' python/asp/orchestrator/asp_cluster_pb2_grpc.py
fi

# 6. Build the Rust Core via Maturin
echo "[*] Compiling the Rust core and injecting PyO3 bindings..."
maturin develop --release

echo "============================================================"
echo " SUCCESS: ASP Core Engine is compiled and ready."
echo " To activate the environment, run: source venv/bin/activate"
echo " To start the distributed cluster, run: cd go_services && docker-compose up -d"
echo "============================================================"