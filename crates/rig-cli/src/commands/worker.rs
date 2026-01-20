use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Args;
use rig_core::{Address, ModelId};
use rig_worker::{CandleConfig, RuntimeConfig, WorkerConfig, WorkerNode};
use tokio::signal;

#[derive(Debug, Args)]
pub struct WorkerArgs {
    /// Runtime backend to use.
    #[arg(long, env = "RIG_RUNTIME", default_value = "candle")]
    pub runtime: String,

    /// Coordinator address to connect to.
    #[arg(long, env = "RIG_COORDINATOR_ADDR", default_value = "127.0.0.1:50051")]
    pub coordinator: String,

    /// Address to listen on for peer connections.
    #[arg(
        short,
        long,
        env = "RIG_WORKER_LISTEN_ADDR",
        default_value = "0.0.0.0:0"
    )]
    pub listen_addr: SocketAddr,

    /// Model path in format "name:version=path".
    #[arg(short, long = "model", value_name = "NAME:VERSION=PATH")]
    pub models: Vec<String>,

    /// Device: "cpu", "metal", "cuda", or "auto".
    #[arg(long, env = "RIG_DEVICE", default_value = "auto")]
    pub device: String,

    /// Heartbeat interval in seconds.
    #[arg(long, env = "RIG_HEARTBEAT_INTERVAL", default_value = "10")]
    pub heartbeat_interval: u64,
}

fn parse_model_spec(spec: &str) -> Result<(ModelId, PathBuf)> {
    let (id_part, path) = spec
        .split_once('=')
        .context("Model spec must be in format 'name:version=path'")?;

    let (name, version) = id_part
        .split_once(':')
        .context("Model ID must be in format 'name:version'")?;

    let model_id = ModelId::new(name, version);
    let path = PathBuf::from(path);

    if !path.exists() {
        anyhow::bail!("Model file does not exist: {}", path.display());
    }

    Ok((model_id, path))
}

pub async fn run_worker(args: WorkerArgs) -> Result<()> {
    let coordinator_addr: SocketAddr = args
        .coordinator
        .parse()
        .context("Invalid coordinator address")?;

    let mut model_paths: HashMap<ModelId, PathBuf> = HashMap::new();

    for spec in &args.models {
        let (model_id, path) = parse_model_spec(spec)?;
        model_paths.insert(model_id, path);
    }

    if args.runtime.to_lowercase() != "candle" {
        tracing::warn!(
            runtime = %args.runtime,
            "Unknown runtime '{}', using 'candle'",
            args.runtime
        );
    }

    tracing::info!(device = %args.device, "Using Candle runtime");
    let runtime_config = RuntimeConfig::Candle(CandleConfig::new().with_device(&args.device));

    let heartbeat_interval = Duration::from_secs(args.heartbeat_interval);

    let config = WorkerConfig::default()
        .with_coordinator_addr(Address::tcp(coordinator_addr))
        .with_listen_addr(args.listen_addr)
        .with_heartbeat_interval(heartbeat_interval)
        .with_model_paths(model_paths.clone())
        .with_runtime_config(runtime_config);

    tracing::info!(
        coordinator = %coordinator_addr,
        listen_addr = %args.listen_addr,
        models = ?model_paths.keys().collect::<Vec<_>>(),
        "Starting worker"
    );

    let mut node = WorkerNode::new(config);

    let (model_id, _model_path) = model_paths
        .iter()
        .next()
        .context("No model paths configured. At least one model must be available.")?;
    let model_id = model_id.clone();

    let result = tokio::select! {
        result = node.run(model_id) => result,
        _ = signal::ctrl_c() => {
            tracing::info!("Received shutdown signal");
            node.shutdown();
            Ok(())
        }
    };

    match result {
        Ok(()) => {
            tracing::info!("Worker shut down cleanly");
            Ok(())
        }
        Err(e) => {
            tracing::error!(error = %e, "Worker error");
            Err(e.into())
        }
    }
}
