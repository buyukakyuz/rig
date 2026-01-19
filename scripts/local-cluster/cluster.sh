#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

RIG="$PROJECT_ROOT/target/release/rig"
COORD_ADDR="127.0.0.1:$COORDINATOR_PORT"

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

save_pipeline() {
    echo "$1" > "$SCRIPT_DIR/.pipeline"
}

load_pipeline() {
    if [ -f "$SCRIPT_DIR/.pipeline" ]; then
        cat "$SCRIPT_DIR/.pipeline"
    fi
}

cmd_start() {
    check_model
    save_model
    check_built

    info "Cleaning up any existing processes..."
    pkill -f "target/release/rig" 2>/dev/null || true
    sleep 0.5

    echo ""
    info "Starting local 2-node cluster (Ctrl+C to stop)"
    echo ""

    PIDS=()

    cleanup() {
        echo ""
        info "Shutting down..."
        for pid in "${PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        wait 2>/dev/null || true
        success "Stopped"
        exit 0
    }

    trap cleanup SIGINT SIGTERM

    check_alive() {
        local pid=$1
        local name=$2
        sleep 0.5
        if ! kill -0 "$pid" 2>/dev/null; then
            error "$name failed to start"
            cleanup
            exit 1
        fi
    }

    info "Starting coordinator on port $COORDINATOR_PORT"
    RUST_LOG=${RIG_LOG:-info} $RIG coordinator \
        --listen-addr "127.0.0.1:$COORDINATOR_PORT" &
    PID_COORD=$!
    PIDS+=($PID_COORD)
    check_alive "$PID_COORD" "Coordinator"

    sleep 0.5

    info "Starting worker 1 on port $WORKER1_PORT"
    RUST_LOG=${RIG_LOG:-info} $RIG worker \
        --coordinator "$COORD_ADDR" \
        --listen-addr "127.0.0.1:$WORKER1_PORT" \
        --model "${MODEL_NAME}:v1=${MODEL_PATH}" \
        --device "$DEVICE" &
    PID_W1=$!
    PIDS+=($PID_W1)
    check_alive "$PID_W1" "Worker 1"

    sleep 0.5

    info "Starting worker 2 on port $WORKER2_PORT"
    RUST_LOG=${RIG_LOG:-info} $RIG worker \
        --coordinator "$COORD_ADDR" \
        --listen-addr "127.0.0.1:$WORKER2_PORT" \
        --model "${MODEL_NAME}:v1=${MODEL_PATH}" \
        --device "$DEVICE" &
    PID_W2=$!
    PIDS+=($PID_W2)
    check_alive "$PID_W2" "Worker 2"

    echo ""
    success "Cluster running! Use another terminal for commands."
    echo ""
    echo "  ./scripts/local-cluster/cluster.sh pipeline"
    echo "  ./scripts/local-cluster/cluster.sh generate \"Hello\""
    echo ""

    wait
}

cmd_pipeline() {
    load_model
    if [ -z "$MODEL_NAME" ]; then
        error "No model info. Run 'start' first."
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
        success "Pipeline: $PIPELINE_ID"
        echo ""
        echo "Generate: ./scripts/local-cluster/cluster.sh generate \"Your prompt\""
    else
        error "Failed to create pipeline"
        exit 1
    fi
}

cmd_generate() {
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
    check_built
    $RIG status --coordinator "$COORD_ADDR" --verbose
}

cmd_help() {
    local SCRIPT="scripts/local-cluster/cluster.sh"
    echo "Local 2-node cluster on same machine"
    echo ""
    echo "Usage: MODEL_PATH=<path> MODEL_NAME=<name> ./$SCRIPT <command>"
    echo ""
    echo "Required:"
    echo "  MODEL_PATH   Path to model directory"
    echo "  MODEL_NAME   Model identifier"
    echo ""
    echo "Optional:"
    echo "  DEVICE         metal, cuda, or cpu (default: metal)"
    echo "  SYSTEM_PROMPT  System prompt (default: \"You are a helpful assistant.\")"
    echo ""
    echo "Commands:"
    echo "  start                       Start coordinator + 2 workers (Ctrl+C to stop)"
    echo "  pipeline                    Create a 2-stage pipeline"
    echo "  generate \"prompt\"           Generate text"
    echo "  generate --chat \"prompt\"    Generate with chat template (for instruct models)"
    echo "  status                      Show cluster status"
    echo ""
    echo "Example:"
    echo ""
    echo "  Terminal 1:"
    echo "    MODEL_PATH=models/tiny-llama MODEL_NAME=tiny-llama ./$SCRIPT start"
    echo ""
    echo "  Terminal 2:"
    echo "    ./$SCRIPT pipeline"
    echo "    ./$SCRIPT generate \"Hello!\""
    echo "    ./$SCRIPT generate --chat \"Hello!\""
}

case "${1:-}" in
    start)
        cmd_start
        ;;
    pipeline)
        cmd_pipeline
        ;;
    generate)
        shift
        cmd_generate "$@"
        ;;
    status)
        cmd_status
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
