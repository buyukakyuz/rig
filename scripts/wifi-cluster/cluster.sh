#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

RIG="$PROJECT_ROOT/target/release/rig"

check_built() {
    if [ ! -f "$RIG" ]; then
        error "Binary not found: $RIG"
        echo ""
        echo "Build with. Built the RIG binary first. Refer to README.md."
    fi
}

save_model() {
    echo "$MODEL_NAME" > "$SCRIPT_DIR/.model"
}

load_model() {
    if [ -f "$SCRIPT_DIR/.model" ]; then
        MODEL_NAME=$(cat "$SCRIPT_DIR/.model")
    fi
}

save_coordinator() {
    echo "$1" > "$SCRIPT_DIR/.coordinator"
}

load_coordinator() {
    if [ -f "$SCRIPT_DIR/.coordinator" ]; then
        cat "$SCRIPT_DIR/.coordinator"
    else
        echo "127.0.0.1:$COORDINATOR_PORT"
    fi
}

save_pipeline() {
    echo "$1" > "$SCRIPT_DIR/.pipeline"
}

load_pipeline() {
    if [ -f "$SCRIPT_DIR/.pipeline" ]; then
        cat "$SCRIPT_DIR/.pipeline"
    fi
}

cmd_coordinator() {
    check_built

    local SCRIPT="scripts/wifi-cluster/cluster.sh"

    echo ""
    info "Starting coordinator on port $COORDINATOR_PORT"
    info "Local IP: $LOCAL_IP"
    echo ""
    echo "Workers can connect with:"
    echo "  MODEL_PATH=<path> MODEL_NAME=<name> ./$SCRIPT worker $LOCAL_IP"
    echo ""

    save_coordinator "127.0.0.1:$COORDINATOR_PORT"

    RUST_LOG=${RIG_LOG:-info} $RIG coordinator \
        --listen-addr "0.0.0.0:$COORDINATOR_PORT"
}

cmd_worker() {
    local COORD_IP="${1:-127.0.0.1}"
    local COORD_ADDR="${COORD_IP}:$COORDINATOR_PORT"

    check_model
    save_model
    check_built

    echo ""
    info "Starting worker"
    info "Local IP: $LOCAL_IP"
    info "Coordinator: $COORD_ADDR"
    echo ""

    save_coordinator "$COORD_ADDR"

    RUST_LOG=${RIG_LOG:-info} $RIG worker \
        --coordinator "$COORD_ADDR" \
        --listen-addr "${LOCAL_IP}:$WORKER_PORT" \
        --model "${MODEL_NAME}:v1=${MODEL_PATH}" \
        --device "$DEVICE"
}

cmd_pipeline() {
    local COORD_ADDR="${1:-$(load_coordinator)}"

    load_model
    if [ -z "$MODEL_NAME" ]; then
        error "No model info. Run 'coordinator' or 'worker' first."
        exit 1
    fi

    check_built

    PIPELINE_ID=$(uuidgen | tr '[:upper:]' '[:lower:]')

    echo ""
    info "Creating 2-stage pipeline..."

    if $RIG pipeline create \
        --coordinator "$COORD_ADDR" \
        --model-name "$MODEL_NAME" \
        --pipeline "$PIPELINE_ID" \
        --stages 2; then
        save_pipeline "$PIPELINE_ID"
        save_coordinator "$COORD_ADDR"
        success "Pipeline: $PIPELINE_ID"
        echo ""
        echo "Generate: ./scripts/wifi-cluster/cluster.sh generate \"Your prompt\""
    else
        error "Failed to create pipeline"
        exit 1
    fi
}

cmd_generate() {
    local COORD_ADDR=$(load_coordinator)
    local PIPELINE_ID=$(load_pipeline)
    local EXTRA_ARGS=()
    local PROMPT=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --chat)
                EXTRA_ARGS+=("--chat")
                shift
                ;;
            --*)
                EXTRA_ARGS+=("$1")
                if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
                    EXTRA_ARGS+=("$2")
                    shift
                fi
                shift
                ;;
            *)
                PROMPT="$1"
                shift
                ;;
        esac
    done

    PROMPT="${PROMPT:-Hello, how are you?}"

    if [ -z "$PIPELINE_ID" ]; then
        error "No pipeline. Run 'pipeline' first."
        exit 1
    fi

    check_built

    echo ""
    info "Generating..."
    echo "---"

    $RIG generate \
        --coordinator "$COORD_ADDR" \
        --pipeline "$PIPELINE_ID" \
        --system-prompt "$SYSTEM_PROMPT" \
        --text "$PROMPT" \
        "${EXTRA_ARGS[@]}"

    echo ""
    echo "---"
}

cmd_status() {
    local COORD_ADDR="${1:-$(load_coordinator)}"

    check_built

    $RIG status --coordinator "$COORD_ADDR" --verbose
}

cmd_stop() {
    info "Stopping local rig processes..."
    pkill -f "target/release/rig" 2>/dev/null || true
    success "Stopped"
}

cmd_help() {
    local SCRIPT="scripts/wifi-cluster/cluster.sh"
    echo "Distributed inference over WiFi/LAN"
    echo ""
    echo "Usage: ./$SCRIPT <command>"
    echo ""
    echo "Required (for worker):"
    echo "  MODEL_PATH   Path to model directory"
    echo "  MODEL_NAME   Model identifier"
    echo ""
    echo "Optional:"
    echo "  DEVICE         metal, cuda, or cpu (default: metal)"
    echo "  SYSTEM_PROMPT  System prompt (default: \"You are a helpful assistant.\")"
    echo ""
    echo "Commands:"
    echo "  coordinator              Start the coordinator server"
    echo "  worker [coordinator-ip]  Start a worker (default: localhost)"
    echo "  pipeline [coordinator-ip] Create a 2-stage pipeline"
    echo "  generate \"prompt\"        Generate text"
    echo "  status [coordinator-ip]  Show cluster status"
    echo "  stop                     Stop local processes"
    echo ""
    echo "Quick start (2 computers):"
    echo ""
    echo "  Computer 1:"
    echo "    MODEL_PATH=models/llama MODEL_NAME=llama ./$SCRIPT coordinator"
    echo "    MODEL_PATH=models/llama MODEL_NAME=llama ./$SCRIPT worker"
    echo ""
    echo "  Computer 2:"
    echo "    MODEL_PATH=models/llama MODEL_NAME=llama ./$SCRIPT worker <computer-1-ip>"
    echo ""
    echo "  Either computer:"
    echo "    ./$SCRIPT pipeline"
    echo "    ./$SCRIPT generate \"Hello\""
}

case "${1:-}" in
    coordinator)
        cmd_coordinator
        ;;
    worker)
        cmd_worker "$2"
        ;;
    pipeline)
        cmd_pipeline "$2"
        ;;
    generate)
        shift
        cmd_generate "$@"
        ;;
    status)
        cmd_status "$2"
        ;;
    stop)
        cmd_stop
        ;;
    help|--help|-h|"")
        cmd_help
        ;;
    *)
        error "Unknown command: $1"
        echo ""
        cmd_help
        exit 1
        ;;
esac
