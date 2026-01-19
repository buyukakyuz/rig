use std::net::SocketAddr;
use std::ops::Range;
use std::sync::Arc;

use rig_core::{CoordError, NodeId, PipelineConfig, PipelineId};
use rig_transport_tcp::{TcpConfig, TcpListener};
use tokio::sync::broadcast;

use crate::config::CoordinatorConfig;
use crate::handler::ConnectionHandler;
use crate::inference::InferenceEngine;
use crate::state::CoordinatorState;

#[derive(Debug)]
pub struct CoordinatorServer {
    config: CoordinatorConfig,
    state: Arc<CoordinatorState>,
    engine: Arc<InferenceEngine>,
    shutdown_tx: broadcast::Sender<()>,
    listen_addr: Option<SocketAddr>,
}

impl CoordinatorServer {
    #[must_use]
    pub fn new(config: CoordinatorConfig) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        let state = Arc::new(CoordinatorState::new(&config));
        let engine = Arc::new(InferenceEngine::new(Arc::clone(&state)));

        Self {
            config,
            state,
            engine,
            shutdown_tx,
            listen_addr: None,
        }
    }

    #[must_use]
    pub fn state(&self) -> Arc<CoordinatorState> {
        Arc::clone(&self.state)
    }

    #[must_use]
    pub const fn config(&self) -> &CoordinatorConfig {
        &self.config
    }

    #[must_use]
    pub fn shutdown_receiver(&self) -> broadcast::Receiver<()> {
        self.shutdown_tx.subscribe()
    }

    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }

    #[must_use]
    pub const fn listen_addr(&self) -> Option<SocketAddr> {
        self.listen_addr
    }

    pub async fn run(&mut self) -> Result<(), CoordError> {
        let config = TcpConfig::default().with_read_timeout(None);
        let listener = TcpListener::bind_addr_with_config(self.config.listen_addr, config)
            .await
            .map_err(|e| {
                CoordError::CoordinatorUnreachable(format!(
                    "Failed to bind to {}: {e}",
                    self.config.listen_addr
                ))
            })?;

        let local_addr = listener.local_socket_addr().map_err(|e| {
            CoordError::CoordinatorUnreachable(format!("Failed to get local address: {e}"))
        })?;
        self.listen_addr = Some(local_addr);
        tracing::info!(addr = %local_addr, "Coordinator listening");

        let mut shutdown_rx = self.shutdown_tx.subscribe();

        loop {
            tokio::select! {
                accept_result = listener.accept_with_socket_addr() => {
                    match accept_result {
                        Ok((transport, addr)) => {
                            tracing::debug!(%addr, "Accepted connection");

                            let handler = ConnectionHandler::new(
                                Arc::clone(&self.state),
                                Arc::clone(&self.engine),
                                transport,
                                addr,
                            );

                            tokio::spawn(async move {
                                if let Err(e) = handler.run().await {
                                    tracing::debug!(%addr, error = %e, "Connection handler finished with error");
                                }
                            });
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "Failed to accept connection");
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    tracing::info!("Shutdown signal received, stopping server");
                    break;
                }
            }
        }

        Ok(())
    }

    pub async fn create_pipeline(
        &self,
        config: PipelineConfig,
        assignments: Vec<(NodeId, Range<usize>)>,
        pipeline_id: Option<PipelineId>,
    ) -> Result<PipelineId, CoordError> {
        self.state
            .create_pipeline(config, assignments, pipeline_id)
            .await
    }

    pub async fn node_count(&self) -> usize {
        self.state.node_count().await
    }

    pub async fn pipeline_count(&self) -> usize {
        self.state.pipeline_count().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_server_creation() {
        let config = CoordinatorConfig::default();
        let server = CoordinatorServer::new(config);

        assert!(server.listen_addr().is_none());
        assert_eq!(server.state().node_count().await, 0);
    }

    #[tokio::test]
    async fn test_server_shutdown() {
        let config =
            CoordinatorConfig::default().with_listen_addr(SocketAddr::from(([127, 0, 0, 1], 0)));
        let mut server = CoordinatorServer::new(config);
        let shutdown_tx = server.shutdown_tx.clone();
        let server_handle = tokio::spawn(async move { server.run().await });

        tokio::time::sleep(Duration::from_millis(50)).await;
        shutdown_tx.send(()).ok();

        let result = tokio::time::timeout(Duration::from_millis(500), server_handle).await;
        assert!(result.is_ok(), "Server should shut down within timeout");
    }
}
