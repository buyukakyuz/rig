use std::net::SocketAddr;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Args;
use rig_coordinator::{CoordinatorConfig, CoordinatorServer, HeartbeatMonitor};
use rig_core::RigConfig;
use tokio::signal;

#[derive(Debug, Args)]
pub struct CoordinatorArgs {
    /// Address to listen on.
    #[arg(short, long, env = "RIG_COORDINATOR_LISTEN_ADDR")]
    pub listen_addr: Option<SocketAddr>,

    /// Heartbeat interval in seconds.
    #[arg(long, env = "RIG_HEARTBEAT_INTERVAL")]
    pub heartbeat_interval: Option<u64>,

    /// Heartbeat timeout in seconds.
    #[arg(long, env = "RIG_HEARTBEAT_TIMEOUT")]
    pub heartbeat_timeout: Option<u64>,

    /// Maximum number of nodes.
    #[arg(long, env = "RIG_MAX_NODES")]
    pub max_nodes: Option<usize>,
}

pub async fn run_coordinator(args: CoordinatorArgs, config: &RigConfig) -> Result<()> {
    let listen_addr = match args.listen_addr {
        Some(addr) => addr,
        None => config
            .coordinator
            .listen_addr
            .parse()
            .context("Invalid listen address in config")?,
    };
    let heartbeat_interval = Duration::from_secs(
        args.heartbeat_interval
            .unwrap_or(config.coordinator.heartbeat_interval_secs),
    );
    let heartbeat_timeout = Duration::from_secs(
        args.heartbeat_timeout
            .unwrap_or(config.coordinator.heartbeat_timeout_secs),
    );
    let max_nodes = args.max_nodes.unwrap_or(config.coordinator.max_nodes);

    let config = CoordinatorConfig::default()
        .with_listen_addr(listen_addr)
        .with_heartbeat_interval(heartbeat_interval)
        .with_heartbeat_timeout(heartbeat_timeout)
        .with_max_nodes(max_nodes);

    tracing::info!(
        listen_addr = %listen_addr,
        heartbeat_interval = ?heartbeat_interval,
        heartbeat_timeout = ?heartbeat_timeout,
        max_nodes,
        "Starting coordinator"
    );

    let mut server = CoordinatorServer::new(config.clone());

    let heartbeat_monitor = HeartbeatMonitor::new(
        server.state(),
        config.heartbeat_timeout,
        config.heartbeat_check_interval,
        server.shutdown_receiver(),
    );
    let heartbeat_handle = tokio::spawn(heartbeat_monitor.run());

    let result = tokio::select! {
        res = server.run() => res,
        _ = signal::ctrl_c() => {
            tracing::info!("Received shutdown signal");
            server.shutdown();
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(())
        }
    };

    heartbeat_handle.abort();

    match result {
        Ok(()) => {
            tracing::info!("Coordinator shut down cleanly");
            Ok(())
        }
        Err(e) => {
            tracing::error!(error = %e, "Coordinator error");
            Err(e.into())
        }
    }
}
