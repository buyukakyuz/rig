#![allow(clippy::expect_used, clippy::panic)]

use std::net::SocketAddr;
use std::time::Duration;

use rig_coordinator::{CoordinatorConfig, CoordinatorServer};
use rig_core::{
    Address, Codec, CoordinatorIncoming, CoordinatorMessage, CoordinatorOutgoing, FramedTransport,
    HeartbeatRequest, NodeId, NodeInfo, NodeStatus, RegisterRequest, RuntimeCapabilities,
    TransportFactory, WorkerMessage,
};
use rig_message_bincode::BincodeCodec;
use rig_transport_tcp::{TcpListener, TcpTransport, TcpTransportFactory};

fn test_node_info(node_id: NodeId, port: u16) -> NodeInfo {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    NodeInfo::new(
        node_id,
        vec![Address::tcp(addr)],
        NodeStatus::Healthy,
        RuntimeCapabilities::new("candle", 0, vec![]),
    )
}

async fn send_recv<T: FramedTransport>(
    transport: &T,
    codec: &BincodeCodec,
    msg: WorkerMessage,
) -> CoordinatorMessage {
    let incoming = CoordinatorIncoming::Worker(msg);
    let frame = codec
        .encode(&incoming)
        .unwrap_or_else(|e| panic!("encode failed: {e}"));
    transport
        .send_frame(&frame)
        .await
        .unwrap_or_else(|e| panic!("send failed: {e}"));

    let response_frame = transport
        .recv_frame()
        .await
        .unwrap_or_else(|e| panic!("recv failed: {e}"));

    let response: CoordinatorOutgoing = codec
        .decode(&response_frame)
        .unwrap_or_else(|e| panic!("decode failed: {e}"));

    match response {
        CoordinatorOutgoing::Worker(coord_msg) => coord_msg,
        CoordinatorOutgoing::Cli(cli_resp) => {
            panic!("Expected Worker response, got Cli: {cli_resp:?}")
        }
    }
}

async fn start_test_server() -> (SocketAddr, CoordinatorServer, tokio::task::JoinHandle<()>) {
    let config =
        CoordinatorConfig::default().with_listen_addr(SocketAddr::from(([127, 0, 0, 1], 0)));
    let server = CoordinatorServer::new(config);

    let (addr_tx, addr_rx) = tokio::sync::oneshot::channel();
    let state = server.state();

    let handle = tokio::spawn(async move {
        let listener = TcpListener::bind_addr(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .unwrap_or_else(|e| panic!("bind failed: {e}"));
        let addr = listener
            .local_socket_addr()
            .unwrap_or_else(|e| panic!("local_addr failed: {e}"));
        addr_tx.send(addr).ok();

        loop {
            match listener.accept_with_socket_addr().await {
                Ok((transport, remote_addr)) => {
                    let handler_state = state.clone();
                    let handler = rig_coordinator::handler::ConnectionHandler::new(
                        handler_state,
                        transport,
                        remote_addr,
                    );
                    tokio::spawn(async move {
                        if let Err(e) = handler.run().await {
                            tracing::debug!("Handler error: {e}");
                        }
                    });
                }
                Err(e) => {
                    tracing::error!("Accept error: {e}");
                    break;
                }
            }
        }
    });

    let addr = addr_rx
        .await
        .unwrap_or_else(|e| panic!("failed to get address: {e}"));

    let config =
        CoordinatorConfig::default().with_listen_addr(SocketAddr::from(([127, 0, 0, 1], 0)));
    let server = CoordinatorServer::new(config);

    (addr, server, handle)
}

async fn connect(addr: SocketAddr) -> TcpTransport {
    let factory = TcpTransportFactory::new();
    let address = Address::tcp(addr);
    factory
        .connect(&address)
        .await
        .unwrap_or_else(|e| panic!("connect failed: {e}"))
}

#[tokio::test]
async fn test_node_registration() {
    let (addr, _server, _handle) = start_test_server().await;

    let transport = connect(addr).await;
    let codec = BincodeCodec::new();

    let node_id = NodeId::new();
    let msg = WorkerMessage::Register(RegisterRequest::new(test_node_info(node_id, 5000)));
    let response = send_recv(&transport, &codec, msg).await;

    match response {
        CoordinatorMessage::RegisterResponse(r) => {
            assert!(r.accepted, "Registration should be accepted");
            assert_eq!(r.node_id, node_id);
        }
        other => panic!("Expected RegisterResponse, got {other:?}"),
    }
}

#[tokio::test]
async fn test_heartbeat() {
    let (addr, _server, _handle) = start_test_server().await;

    let transport = connect(addr).await;
    let codec = BincodeCodec::new();

    let node_id = NodeId::new();
    let msg = WorkerMessage::Register(RegisterRequest::new(test_node_info(node_id, 5001)));
    let _ = send_recv(&transport, &codec, msg).await;

    let heartbeat = WorkerMessage::Heartbeat(HeartbeatRequest::new(node_id, NodeStatus::Healthy));
    let response = send_recv(&transport, &codec, heartbeat).await;

    match response {
        CoordinatorMessage::HeartbeatAck => {}
        other => panic!("Expected HeartbeatAck, got {other:?}"),
    }
}

#[tokio::test]
async fn test_heartbeat_without_registration() {
    let (addr, _server, _handle) = start_test_server().await;

    let transport = connect(addr).await;
    let codec = BincodeCodec::new();

    let node_id = NodeId::new();
    let heartbeat = WorkerMessage::Heartbeat(HeartbeatRequest::new(node_id, NodeStatus::Healthy));
    let response = send_recv(&transport, &codec, heartbeat).await;

    match response {
        CoordinatorMessage::Error { code, .. } => {
            assert_eq!(code, rig_core::types::protocol::error_codes::NODE_NOT_FOUND);
        }
        other => panic!("Expected Error, got {other:?}"),
    }
}

#[tokio::test]
async fn test_get_assignment_no_assignment() {
    let (addr, _server, _handle) = start_test_server().await;

    let transport = connect(addr).await;
    let codec = BincodeCodec::new();

    let node_id = NodeId::new();
    let msg = WorkerMessage::Register(RegisterRequest::new(test_node_info(node_id, 5002)));
    let _ = send_recv(&transport, &codec, msg).await;

    let msg = WorkerMessage::GetAssignment;
    let response = send_recv(&transport, &codec, msg).await;

    match response {
        CoordinatorMessage::Assignment(assignment) => {
            assert!(assignment.is_none(), "Should have no assignment initially");
        }
        other => panic!("Expected Assignment, got {other:?}"),
    }
}

#[tokio::test]
async fn test_multiple_node_registration() {
    let (addr, _server, _handle) = start_test_server().await;
    let codec = BincodeCodec::new();

    let mut transports = Vec::new();
    let mut node_ids = Vec::new();

    for i in 0..3 {
        let transport = connect(addr).await;
        let node_id = NodeId::new();
        let msg = WorkerMessage::Register(RegisterRequest::new(test_node_info(node_id, 6000 + i)));
        let response = send_recv(&transport, &codec, msg).await;

        match response {
            CoordinatorMessage::RegisterResponse(r) => {
                assert!(r.accepted, "Node {i} should be accepted");
            }
            other => panic!("Node {i}: Expected RegisterResponse, got {other:?}"),
        }

        transports.push(transport);
        node_ids.push(node_id);
    }

    for (i, (transport, node_id)) in transports.iter().zip(node_ids.iter()).enumerate() {
        let heartbeat =
            WorkerMessage::Heartbeat(HeartbeatRequest::new(*node_id, NodeStatus::Healthy));
        let response = send_recv(transport, &codec, heartbeat).await;

        match response {
            CoordinatorMessage::HeartbeatAck => {}
            other => panic!("Node {i}: Expected HeartbeatAck, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn test_coordinator_config_defaults() {
    let config = CoordinatorConfig::default();
    assert_eq!(config.listen_addr.port(), 50051);
    assert_eq!(config.heartbeat_interval, Duration::from_secs(10));
    assert_eq!(config.heartbeat_timeout, Duration::from_secs(30));
    assert_eq!(config.max_nodes, 100);
}
