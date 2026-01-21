use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use rig_core::{Address, ModelId};
use rig_runtime_candle::CandleRuntime;
use rig_worker::{WorkerConfig, WorkerNode};
use tracing::{error, info};
use tracing_subscriber::fmt::format::FmtSpan;

fn parse_config() -> Result<(WorkerConfig, ModelId)> {
    let coordinator_addr =
        std::env::var("COORDINATOR_ADDR").unwrap_or_else(|_| "127.0.0.1:50051".to_string());
    let coordinator_socket: SocketAddr = coordinator_addr.parse()?;

    let listen_addr = std::env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:0".to_string());
    let listen_socket: SocketAddr = listen_addr.parse()?;

    let model_name = std::env::var("MODEL_NAME")
        .map_err(|_| anyhow::anyhow!("MODEL_NAME environment variable is required"))?;
    let model_version = std::env::var("MODEL_VERSION").unwrap_or_else(|_| "v1".to_string());
    let model_path = std::env::var("MODEL_PATH")
        .map_err(|_| anyhow::anyhow!("MODEL_PATH environment variable is required"))?;

    let model_id = ModelId::new(&model_name, &model_version);

    let config = WorkerConfig::default()
        .with_coordinator_addr(Address::tcp(coordinator_socket))
        .with_listen_addr(listen_socket)
        .with_model_path(model_id.clone(), PathBuf::from(model_path));

    Ok((config, model_id))
}

#[tokio::main]
async fn main() -> Result<()> {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_span_events(FmtSpan::CLOSE)
        .with_target(false)
        .init();

    info!("Starting rig-worker");

    let (config, model_id) = parse_config()?;

    info!(
        coordinator = %config.coordinator_addr,
        listen = %config.listen_addr,
        model = %model_id,
        "Configuration loaded"
    );

    let device = std::env::var("DEVICE").unwrap_or_else(|_| "auto".to_string());
    info!(device = %device, "Creating Candle runtime");
    let runtime = match device.as_str() {
        "cpu" => CandleRuntime::cpu()?,
        _ => CandleRuntime::new()?,
    };

    let mut node = WorkerNode::new(config, runtime);

    let node_shutdown_tx = node.shutdown_receiver();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        info!("Received Ctrl+C, initiating shutdown");
        drop(node_shutdown_tx);
    });

    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let node_shutdown = node.shutdown_receiver();
        tokio::spawn(async move {
            match signal(SignalKind::terminate()) {
                Ok(mut sigterm) => {
                    sigterm.recv().await;
                    info!("Received SIGTERM, initiating shutdown");
                }
                Err(e) => {
                    tracing::warn!("Failed to register SIGTERM handler: {e}");
                }
            }
            drop(node_shutdown);
        });
    }

    match node.run(model_id).await {
        Ok(()) => {
            info!("Worker shut down cleanly");
            Ok(())
        }
        Err(e) => {
            error!(error = %e, "Worker failed");
            Err(e.into())
        }
    }
}
