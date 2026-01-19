# Rig

**Run 70B+ models on hardware that shouldn't be able to.**

Rig is a distributed inference framework that splits large language models across multiple machines using pipeline parallelism.

Got a MacBook, an old desktop with a GPU, and a work laptop? None of them can run Llama 70B alone, but together they can. Rig coordinates them into a single inference endpoint over your regular WiFi or LAN.

## Quick Start

**Build** (pick your platform):

Apple Silicon:
```bash
cargo build --release -p rig-cli --features metal
```

NVIDIA:
```bash
cargo build --release -p rig-cli --features cuda
```

CPU only:
```bash
cargo build --release -p rig-cli
```

**Download a model:**

```bash
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/tiny-llama
```

**Run a local 2-node cluster:**

```bash
MODEL_PATH=models/tiny-llama MODEL_NAME=tiny-llama \
    ./scripts/local-cluster/cluster.sh start
```

**Create pipeline** (in another terminal):

```bash
./scripts/local-cluster/cluster.sh pipeline
```

**Generate:**

```bash
./scripts/local-cluster/cluster.sh generate --chat "Hello, how are you?"
```

## Multi-Machine Setup

Machines must be able to reach each other (same WiFi, LAN, VPN, etc.).

**Machine 1 — Start coordinator:**

```bash
./scripts/wifi-cluster/cluster.sh coordinator
```

**Machine 1 — Start worker:**

```bash
MODEL_PATH=models/tiny-llama MODEL_NAME=tiny-llama ./scripts/wifi-cluster/cluster.sh worker
```

**Machine 2 — Start worker** (use the IP shown by coordinator):

```bash
MODEL_PATH=models/tiny-llama MODEL_NAME=tiny-llama ./scripts/wifi-cluster/cluster.sh worker <coordinator-ip>
```

**Create pipeline:**

```bash
./scripts/wifi-cluster/cluster.sh pipeline
```

**Generate:**

```bash
./scripts/wifi-cluster/cluster.sh generate --chat "Explain quantum computing"
```

## Requirements

- Rust 1.85+ (`rustup update`)
- For model downloads: [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)

## Project Status

Under active development. Tested on Apple Silicon; CUDA should work but is untested.
