#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

DEVICE="${DEVICE:-metal}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"

COORDINATOR_PORT="50051"
WORKER_PORT="5001"

detect_ip() {
    if command -v ipconfig &>/dev/null; then
        ipconfig getifaddr en0 2>/dev/null && return
        ipconfig getifaddr en1 2>/dev/null && return
    fi
    if command -v hostname &>/dev/null; then
        hostname -I 2>/dev/null | awk '{print $1}' && return
    fi
    echo "127.0.0.1"
}

LOCAL_IP=$(detect_ip)

check_model() {
    if [ -z "$MODEL_PATH" ]; then
        error "MODEL_PATH not set"
        echo ""
        echo "Usage: MODEL_PATH=<path> MODEL_NAME=<name> ./scripts/wifi-cluster/cluster.sh <command>"
        echo ""
        echo "Example:"
        echo "  MODEL_PATH=models/tiny-llama MODEL_NAME=tiny-llama ./scripts/wifi-cluster/cluster.sh coordinator"
        exit 1
    fi

    if [ -z "$MODEL_NAME" ]; then
        error "MODEL_NAME not set"
        echo ""
        echo "Usage: MODEL_PATH=<path> MODEL_NAME=<name> ./scripts/wifi-cluster/cluster.sh <command>"
        exit 1
    fi

    if [ ! -d "$PROJECT_ROOT/$MODEL_PATH" ] && [ ! -d "$MODEL_PATH" ]; then
        error "Model not found: $MODEL_PATH"
        exit 1
    fi

    if [ -d "$PROJECT_ROOT/$MODEL_PATH" ]; then
        MODEL_PATH="$PROJECT_ROOT/$MODEL_PATH"
    fi
}
