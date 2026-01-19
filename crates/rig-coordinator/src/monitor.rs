use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::broadcast;

use crate::state::CoordinatorState;

#[derive(Debug)]
pub struct HeartbeatMonitor {
    state: Arc<CoordinatorState>,
    timeout: Duration,
    check_interval: Duration,
    shutdown_rx: broadcast::Receiver<()>,
}

impl HeartbeatMonitor {
    #[must_use]
    pub const fn new(
        state: Arc<CoordinatorState>,
        timeout: Duration,
        check_interval: Duration,
        shutdown_rx: broadcast::Receiver<()>,
    ) -> Self {
        Self {
            state,
            timeout,
            check_interval,
            shutdown_rx,
        }
    }

    pub async fn run(mut self) {
        tracing::info!(
            timeout_secs = self.timeout.as_secs(),
            check_interval_secs = self.check_interval.as_secs(),
            "Heartbeat monitor started"
        );

        let mut interval = tokio::time::interval(self.check_interval);

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    self.check_heartbeats().await;
                }
                _ = self.shutdown_rx.recv() => {
                    tracing::info!("Heartbeat monitor shutting down");
                    break;
                }
            }
        }
    }

    async fn check_heartbeats(&self) {
        let now = Instant::now();
        let dead_nodes = self.state.find_dead_nodes(now, self.timeout).await;

        for node_id in dead_nodes {
            tracing::warn!(%node_id, "Node missed heartbeat, marking unhealthy");
            self.state
                .mark_unhealthy(node_id, "Heartbeat timeout")
                .await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CoordinatorConfig;
    use rig_core::{Address, NodeId, NodeInfo, NodeStatus, RuntimeCapabilities};
    use std::net::SocketAddr;

    fn test_node_info(node_id: NodeId) -> NodeInfo {
        let addr = SocketAddr::from(([127, 0, 0, 1], 5000));
        NodeInfo::new(
            node_id,
            vec![Address::tcp(addr)],
            NodeStatus::Healthy,
            RuntimeCapabilities::new("candle", 0, vec![]),
        )
    }

    #[tokio::test]
    async fn test_monitor_detects_dead_nodes() {
        let config = CoordinatorConfig::default();
        let state = Arc::new(CoordinatorState::new(&config));
        let (tx, rx) = broadcast::channel(1);

        let node_id = NodeId::new();
        state
            .register_node(test_node_info(node_id), Vec::new())
            .await
            .ok();

        let monitor = HeartbeatMonitor::new(
            Arc::clone(&state),
            Duration::from_millis(1),
            Duration::from_millis(10),
            rx,
        );

        let monitor_handle = tokio::spawn(async move {
            monitor.run().await;
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        tx.send(()).ok();
        monitor_handle.await.ok();

        let info = state.get_node_info(node_id).await;
        assert!(info.is_some());
    }

    #[tokio::test]
    async fn test_monitor_shutdown() {
        let config = CoordinatorConfig::default();
        let state = Arc::new(CoordinatorState::new(&config));
        let (tx, rx) = broadcast::channel(1);

        let monitor =
            HeartbeatMonitor::new(state, Duration::from_secs(30), Duration::from_secs(5), rx);

        let handle = tokio::spawn(async move {
            monitor.run().await;
        });

        tx.send(()).ok();

        let result = tokio::time::timeout(Duration::from_millis(100), handle).await;
        assert!(result.is_ok());
    }
}
