#!/bin/bash
# ==============================================================================
# Analytic Structural Physics (ASP) - One-Click Environment Installer
# ==============================================================================
# This script prepares the entire ASP development environment, including:
# - System dependencies (Protobuf, build tools)
# - Go language toolchain
# - Rust language toolchain
# - Python dependencies (with native Conda environment support)
# - gRPC / Protobuf stub generation
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

# Text Formatting
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

OS="$(uname -s)"
ARCH="$(uname -m)"

log_info "Starting ASP Core One-Click Installer..."
log_info "Detected Operating System: $OS ($ARCH)"

# ------------------------------------------------------------------------------
# 1. System Dependencies (Protobuf, Build Tools)
# ------------------------------------------------------------------------------
log_info "Checking and installing system dependencies..."

if [ "$OS" = "Linux" ]; then
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -y
        sudo apt-get install -y build-essential curl wget git protobuf-compiler libprotobuf-dev python3-dev
        log_success "Linux system dependencies installed."
    else
        log_warn "apt-get not found. Skipping system package installation. Please ensure protobuf-compiler is installed."
    fi
elif [ "$OS" = "Darwin" ]; then
    if command -v brew >/dev/null 2>&1; then
        brew install protobuf git curl wget
        log_success "macOS system dependencies installed."
    else
        log_error "Homebrew not found. Please install Homebrew first (https://brew.sh/) or install protobuf manually."
        exit 1
    fi
else
    log_warn "Unsupported OS for automatic system dependencies. Ensure protobuf is installed."
fi

# ------------------------------------------------------------------------------
# 2. Go Toolchain Installation
# ------------------------------------------------------------------------------
GO_VERSION="1.21.6" # Set default reliable Go version

if ! command -v go >/dev/null 2>&1; then
    log_info "Go is not installed. Installing Go ${GO_VERSION}..."
    
    if [ "$OS" = "Linux" ]; then
        if [ "$ARCH" = "x86_64" ]; then GO_ARCH="amd64"; else GO_ARCH="arm64"; fi
        TMP_DIR=$(mktemp -d)
        cd "$TMP_DIR"
        wget -q "https://go.dev/dl/go${GO_VERSION}.linux-${GO_ARCH}.tar.gz"
        sudo rm -rf /usr/local/go
        sudo tar -C /usr/local -xzf "go${GO_VERSION}.linux-${GO_ARCH}.tar.gz"
        export PATH=$PATH:/usr/local/go/bin
        echo "export PATH=\$PATH:/usr/local/go/bin" >> ~/.bashrc
        cd - > /dev/null
        log_success "Go installed successfully (added to ~/.bashrc)."
    elif [ "$OS" = "Darwin" ]; then
        brew install go
        log_success "Go installed via Homebrew."
    fi
else
    log_success "Go is already installed: $(go version)"
fi

# ------------------------------------------------------------------------------
# 3. Rust Toolchain Installation
# ------------------------------------------------------------------------------
if ! command -v cargo >/dev/null 2>&1; then
    log_info "Rust/Cargo is not installed. Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    log_success "Rust installed successfully."
else
    log_success "Rust is already installed: $(cargo --version)"
fi

# ------------------------------------------------------------------------------
# 4. Python & Conda Environment Setup
# ------------------------------------------------------------------------------
log_info "Configuring Python environment..."

# Check for active Conda environment
if [ -n "$CONDA_PREFIX" ]; then
    CONDA_ENV_NAME=$(basename "$CONDA_PREFIX")
    log_success "Active Conda environment detected: [${CONDA_ENV_NAME}] at $CONDA_PREFIX"
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
    PIP_BIN="$CONDA_PREFIX/bin/pip"
else
    log_warn "No Conda environment detected."
    log_info "Falling back to standard python3..."
    PYTHON_BIN=$(command -v python3 || command -v python)
    PIP_BIN=$(command -v pip3 || command -v pip)
fi

if [ -z "$PYTHON_BIN" ]; then
    log_error "Python not found! Please install Python or activate a Conda environment."
    exit 1
fi

log_info "Using Python: $($PYTHON_BIN --version) located at $PYTHON_BIN"

# Upgrade pip and install the ASP package with [test] extras natively
log_info "Installing Python dependencies (including grpc_tools)..."
$PYTHON_BIN -m pip install --upgrade pip
$PYTHON_BIN -m pip install grpcio grpcio-tools protobuf

# Assuming script is run from the project root where setup.py / pyproject.toml lives
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    $PYTHON_BIN -m pip install -e .[test]
    log_success "ASP Python package installed in editable mode."
else
    log_warn "No setup.py or pyproject.toml found in current directory. Skipping 'pip install -e .'"
fi

# ------------------------------------------------------------------------------
# 5. Protobuf / gRPC Stub Generation
# ------------------------------------------------------------------------------
log_info "Generating Python gRPC Stubs..."

PROTO_DIR="go_services/proto"
PYTHON_OUT_DIR="python/asp/orchestrator"

if [ -d "$PROTO_DIR" ]; then
    # Ensure output directory exists
    mkdir -p "$PYTHON_OUT_DIR"
    
    # Generate the python stubs using the Conda/System python
    $PYTHON_BIN -m grpc_tools.protoc \
        -I"$PROTO_DIR" \
        --python_out="$PYTHON_OUT_DIR" \
        --grpc_python_out="$PYTHON_OUT_DIR" \
        "$PROTO_DIR"/*.proto
        
    log_success "gRPC stubs generated successfully in $PYTHON_OUT_DIR"
else
    log_warn "Protobuf directory ($PROTO_DIR) not found. Skipping gRPC stub generation."
fi

# ------------------------------------------------------------------------------
# 6. Finalization
# ------------------------------------------------------------------------------
log_success "========================================================================"
log_success "ASP Core environment successfully configured!"
log_success "========================================================================"
log_info "If Go or Rust was freshly installed, you may need to reload your shell:"
log_info "    source ~/.bashrc   (or ~/.zshrc)"
log_info "You are now ready to run ASP!"