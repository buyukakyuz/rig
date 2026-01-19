use std::net::SocketAddr;
use std::time::Duration;

use anyhow::Result;
use clap::Args;
use rig_coordinator::{CoordinatorConfig, CoordinatorServer, HeartbeatMonitor};
use tokio::signal;

#[derive(Debug, Args)]
pub struct CoordinatorArgs {
    /// Address to listen on.
    #[arg(
        short,
        long,
        env = "RIG_COORDINATOR_LISTEN_ADDR",
        default_value = "0.0.0.0:50051"
    )]
    pub listen_addr: SocketAddr,

    /// Heartbeat interval in seconds.
    #[arg(long, env = "RIG_HEARTBEAT_INTERVAL", default_value = "10")]
    pub heartbeat_interval: u64,

    /// Heartbeat timeout in seconds.
    #[arg(long, env = "RIG_HEARTBEAT_TIMEOUT", default_value = "30")]
    pub heartbeat_timeout: u64,

    /// Maximum number of nodes.
    #[arg(long, env = "RIG_MAX_NODES", default_value = "100")]
    pub max_nodes: usize,
}

pub async fn run_coordinator(args: CoordinatorArgs) -> Result<()> {
    let listen_addr = args.listen_addr;
    let heartbeat_interval = Duration::from_secs(args.heartbeat_interval);
    let heartbeat_timeout = Duration::from_secs(args.heartbeat_timeout);
    let max_nodes = args.max_nodes;

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
