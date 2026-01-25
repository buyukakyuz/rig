use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Args;
use rig_core::{Address, ModelId, RigConfig};
use rig_message_bincode::BincodeCodec;
use rig_transport_tcp::{TcpConfig, TcpTransportFactory};
use rig_worker::{WorkerConfig, WorkerNode};
use tokio::signal;

#[derive(Debug, Args)]
pub struct WorkerArgs {
    /// Runtime backend to use.
    #[arg(long, env = "RIG_RUNTIME", default_value = "candle")]
    pub runtime: String,

    /// Coordinator address to connect to.
    #[arg(long, env = "RIG_COORDINATOR_ADDR")]
    pub coordinator: Option<String>,

    /// Address to listen on for peer connections.
    #[arg(short, long, env = "RIG_WORKER_LISTEN_ADDR")]
    pub listen_addr: Option<SocketAddr>,

    /// Model path in format "name:version=path".
    #[arg(short, long = "model", value_name = "NAME:VERSION=PATH")]
    pub models: Vec<String>,

    /// Device: "cpu", "metal", "cuda", or "auto".
    #[arg(long, env = "RIG_DEVICE")]
    pub device: Option<String>,

    /// Heartbeat interval in seconds.
    #[arg(long, env = "RIG_HEARTBEAT_INTERVAL")]
    pub heartbeat_interval: Option<u64>,
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

pub async fn run_worker(args: WorkerArgs, config: &RigConfig) -> Result<()> {
    let coordinator_str = args
        .coordinator
        .as_deref()
        .unwrap_or(&config.worker.coordinator_addr);
    let coordinator_addr: SocketAddr = coordinator_str
        .parse()
        .context("Invalid coordinator address")?;

    let listen_addr = match args.listen_addr {
        Some(addr) => addr,
        None => config
            .worker
            .listen_addr
            .parse()
            .context("Invalid worker listen address in config")?,
    };

    let device = args.device.as_deref().unwrap_or(&config.runtime.device);

    let heartbeat_interval = Duration::from_secs(
        args.heartbeat_interval
            .unwrap_or(config.worker.heartbeat_interval_secs),
    );

    let mut model_paths: HashMap<ModelId, PathBuf> = HashMap::new();

    for spec in &args.models {
        let (model_id, path) = parse_model_spec(spec)?;
        model_paths.insert(model_id, path);
    }

    let worker_config = WorkerConfig::default()
        .with_coordinator_addr(Address::tcp(coordinator_addr))
        .with_listen_addr(listen_addr)
        .with_heartbeat_interval(heartbeat_interval)
        .with_model_paths(model_paths.clone());

    tracing::info!(
        coordinator = %coordinator_addr,
        listen_addr = %listen_addr,
        models = ?model_paths.keys().collect::<Vec<_>>(),
        "Starting worker"
    );

    let runtime = crate::runtime::create_runtime(device)?;

    let tcp_config = TcpConfig::default().with_read_timeout(None);
    let transport_factory = TcpTransportFactory::with_config(tcp_config);
    let codec = BincodeCodec::new();

    let mut node = WorkerNode::new(worker_config, runtime, transport_factory, codec);

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
