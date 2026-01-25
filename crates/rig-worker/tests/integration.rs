#![allow(clippy::expect_used, clippy::panic)]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use rig_coordinator::handler::ConnectionHandler;
use rig_coordinator::{CoordinatorConfig, CoordinatorServer, CoordinatorState};
use rig_core::{Address, DType, ModelId, NodeId, PipelineConfig, TransportFactory};
use rig_message_bincode::BincodeCodec;
use rig_transport_tcp::{TcpConfig, TcpTransport, TcpTransportFactory};
use rig_worker::CoordinatorClient;

type TestCoordinatorClient = CoordinatorClient<TcpTransport, BincodeCodec>;

async fn connect_to_coordinator(addr: &Address) -> TestCoordinatorClient {
    let config = TcpConfig::default().with_read_timeout(None);
    let factory = TcpTransportFactory::with_config(config);
    let transport = factory.connect(addr).await.expect("Failed to connect");
    CoordinatorClient::new(transport, BincodeCodec::new())
}

struct TestCoordinator {
    state: Arc<CoordinatorState>,
    addr: SocketAddr,
    _handle: tokio::task::JoinHandle<()>,
}

impl TestCoordinator {
    async fn start() -> Self {
        let listen_addr = SocketAddr::from(([127, 0, 0, 1], 0));
        let config = CoordinatorConfig::new(listen_addr)
            .with_max_nodes(10)
            .with_heartbeat_timeout(Duration::from_secs(60));

        let server = CoordinatorServer::new(config);
        let state = server.state();

        let (addr_tx, addr_rx) = tokio::sync::oneshot::channel();

        let state_clone = Arc::clone(&state);

        let handle = tokio::spawn(async move {
            let listener =
                rig_transport_tcp::TcpListener::bind_addr(SocketAddr::from(([127, 0, 0, 1], 0)))
                    .await
                    .expect("Failed to bind");
            let addr = listener.local_socket_addr().expect("Failed to get addr");

            let _ = addr_tx.send(addr);

            loop {
                let Ok((transport, peer_addr)) = listener.accept_with_socket_addr().await else {
                    break;
                };

                let handler = ConnectionHandler::new(state_clone.clone(), transport, peer_addr);

                tokio::spawn(async move {
                    let _ = handler.run().await;
                });
            }
        });

        let addr = addr_rx.await.expect("Failed to get address");

        Self {
            state,
            addr,
            _handle: handle,
        }
    }

    const fn addr(&self) -> SocketAddr {
        self.addr
    }

    fn state(&self) -> Arc<CoordinatorState> {
        Arc::clone(&self.state)
    }
}

#[tokio::test]
async fn test_worker_registration() {
    let coordinator = TestCoordinator::start().await;

    let coord_addr = Address::tcp(coordinator.addr());
    let mut client = connect_to_coordinator(&coord_addr).await;

    assert!(!client.is_registered());

    let node_id = NodeId::new();
    let listen_addr = SocketAddr::from(([127, 0, 0, 1], 0));
    let info = rig_core::NodeInfo::new(
        node_id,
        vec![Address::tcp(listen_addr)],
        rig_core::NodeStatus::Healthy,
        rig_core::RuntimeCapabilities::new("candle", 0, vec![]),
    );

    let registered_id = client.register(info).await.expect("Registration failed");
    assert_eq!(registered_id, node_id);
    assert!(client.is_registered());

    tokio::time::sleep(Duration::from_millis(10)).await;

    assert_eq!(coordinator.state().node_count().await, 1);
}

#[tokio::test]
async fn test_worker_heartbeat() {
    let coordinator = TestCoordinator::start().await;

    let coord_addr = Address::tcp(coordinator.addr());
    let mut client = connect_to_coordinator(&coord_addr).await;

    let node_id = NodeId::new();
    let listen_addr = SocketAddr::from(([127, 0, 0, 1], 0));
    let info = rig_core::NodeInfo::new(
        node_id,
        vec![Address::tcp(listen_addr)],
        rig_core::NodeStatus::Healthy,
        rig_core::RuntimeCapabilities::new("candle", 0, vec![]),
    );
    client.register(info).await.expect("Registration failed");

    client
        .heartbeat(rig_core::NodeStatus::Healthy)
        .await
        .expect("Heartbeat failed");

    client
        .heartbeat(rig_core::NodeStatus::Healthy)
        .await
        .expect("Second heartbeat failed");
}

#[tokio::test]
async fn test_worker_get_assignment_none() {
    let coordinator = TestCoordinator::start().await;

    let coord_addr = Address::tcp(coordinator.addr());
    let mut client = connect_to_coordinator(&coord_addr).await;

    let node_id = NodeId::new();
    let listen_addr = SocketAddr::from(([127, 0, 0, 1], 0));
    let info = rig_core::NodeInfo::new(
        node_id,
        vec![Address::tcp(listen_addr)],
        rig_core::NodeStatus::Healthy,
        rig_core::RuntimeCapabilities::new("candle", 0, vec![]),
    );
    client.register(info).await.expect("Registration failed");

    let assignment = client
        .get_assignment()
        .await
        .expect("Get assignment failed");
    assert!(assignment.is_none());
}

#[tokio::test]
async fn test_worker_get_assignment_with_pipeline() {
    let coordinator = TestCoordinator::start().await;
    let coord_addr = Address::tcp(coordinator.addr());
    let mut client = connect_to_coordinator(&coord_addr).await;

    let node_id = NodeId::new();
    let listen_addr = SocketAddr::from(([127, 0, 0, 1], 5000));
    let info = rig_core::NodeInfo::new(
        node_id,
        vec![Address::tcp(listen_addr)],
        rig_core::NodeStatus::Healthy,
        rig_core::RuntimeCapabilities::new("candle", 0, vec![]),
    );
    client.register(info).await.expect("Registration failed");

    let model_id = ModelId::new("test-model", "v1");
    let config = PipelineConfig::new(model_id, "/models/test", 20, DType::F16);
    let _pipeline_id = coordinator
        .state()
        .create_pipeline(config, vec![(node_id, 0..20)], None)
        .await
        .expect("Failed to create pipeline");

    let assignment = client
        .get_assignment()
        .await
        .expect("Get assignment failed");
    assert!(assignment.is_some());

    let assignment = assignment.expect("Assignment should be present");
    assert_eq!(assignment.layer_range, 0..20);
    assert!(assignment.neighbors.prev.is_none());
    assert!(assignment.neighbors.next.is_none());
}

#[tokio::test]
async fn test_worker_report_ready() {
    let coordinator = TestCoordinator::start().await;

    let coord_addr = Address::tcp(coordinator.addr());
    let mut client = connect_to_coordinator(&coord_addr).await;

    let node_id = NodeId::new();
    let listen_addr = SocketAddr::from(([127, 0, 0, 1], 5001));
    let info = rig_core::NodeInfo::new(
        node_id,
        vec![Address::tcp(listen_addr)],
        rig_core::NodeStatus::Healthy,
        rig_core::RuntimeCapabilities::new("candle", 0, vec![]),
    );
    client.register(info).await.expect("Registration failed");

    let model_id = ModelId::new("test-model", "v1");
    let config = PipelineConfig::new(model_id, "/models/test", 20, DType::F16);
    let pipeline_id = coordinator
        .state()
        .create_pipeline(config, vec![(node_id, 0..20)], None)
        .await
        .expect("Failed to create pipeline");

    client
        .report_ready(pipeline_id)
        .await
        .expect("Report ready failed");
}

#[tokio::test]
async fn test_worker_deregister() {
    let coordinator = TestCoordinator::start().await;

    let coord_addr = Address::tcp(coordinator.addr());
    let mut client = connect_to_coordinator(&coord_addr).await;

    let node_id = NodeId::new();
    let listen_addr = SocketAddr::from(([127, 0, 0, 1], 0));
    let info = rig_core::NodeInfo::new(
        node_id,
        vec![Address::tcp(listen_addr)],
        rig_core::NodeStatus::Healthy,
        rig_core::RuntimeCapabilities::new("candle", 0, vec![]),
    );
    client.register(info).await.expect("Registration failed");

    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(coordinator.state().node_count().await, 1);

    client.deregister().await.expect("Deregister failed");

    tokio::time::sleep(Duration::from_millis(100)).await;

    assert_eq!(coordinator.state().node_count().await, 0);
}
