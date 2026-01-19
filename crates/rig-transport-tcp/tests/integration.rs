#![allow(clippy::expect_used, clippy::panic)]

use rig_core::traits::{FramedTransport, TransportFactory};
use rig_core::types::Address;
use rig_transport_tcp::{TcpConfig, TcpListener, TcpTransportFactory};
use std::time::Duration;

#[tokio::test]
async fn send_receive_roundtrip() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind listener");
    let addr = listener
        .local_socket_addr()
        .expect("failed to get local addr");

    let handle = tokio::spawn(async move {
        let (transport, _) = listener
            .accept_with_socket_addr()
            .await
            .expect("failed to accept");
        let msg = transport.recv_frame().await.expect("failed to recv");
        transport.send_frame(&msg).await.expect("failed to send");
    });

    let factory = TcpTransportFactory::default();
    let transport = factory
        .connect(&Address::Tcp(addr))
        .await
        .expect("failed to connect");

    let original = b"hello world";
    transport
        .send_frame(original)
        .await
        .expect("failed to send");

    let received = transport.recv_frame().await.expect("failed to recv");
    assert_eq!(received, original);

    handle.await.expect("server task failed");
}

#[tokio::test]
async fn large_message_transfer() {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let large_data: Vec<u8> = (0..10 * 1024 * 1024)
        .map(|i: i32| (i % 256) as u8)
        .collect();

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind listener");
    let addr = listener
        .local_socket_addr()
        .expect("failed to get local addr");

    let expected = large_data.clone();

    let handle = tokio::spawn(async move {
        let (transport, _) = listener
            .accept_with_socket_addr()
            .await
            .expect("failed to accept");
        let msg = transport.recv_frame().await.expect("failed to recv");
        transport.send_frame(&msg).await.expect("failed to send");
    });

    let factory = TcpTransportFactory::default();
    let transport = factory
        .connect(&Address::Tcp(addr))
        .await
        .expect("failed to connect");

    transport
        .send_frame(&large_data)
        .await
        .expect("failed to send large message");

    let received = transport
        .recv_frame()
        .await
        .expect("failed to recv large message");

    assert_eq!(received.len(), expected.len());
    assert_eq!(received, expected);

    handle.await.expect("server task failed");
}

#[tokio::test]
async fn multiple_messages_sequential() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind listener");
    let addr = listener
        .local_socket_addr()
        .expect("failed to get local addr");

    let handle = tokio::spawn(async move {
        let (transport, _) = listener
            .accept_with_socket_addr()
            .await
            .expect("failed to accept");

        for _ in 0..5 {
            let msg = transport.recv_frame().await.expect("failed to recv");
            transport.send_frame(&msg).await.expect("failed to send");
        }
    });

    let factory = TcpTransportFactory::default();
    let transport = factory
        .connect(&Address::Tcp(addr))
        .await
        .expect("failed to connect");

    let messages = [
        b"message one".to_vec(),
        b"message two".to_vec(),
        b"three".to_vec(),
        vec![0u8; 1000],
        vec![255u8; 5000],
    ];

    for original in &messages {
        transport
            .send_frame(original)
            .await
            .expect("failed to send");
        let received = transport.recv_frame().await.expect("failed to recv");
        assert_eq!(&received, original);
    }

    handle.await.expect("server task failed");
}

#[tokio::test]
async fn factory_connection_works() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind listener");
    let addr = listener
        .local_socket_addr()
        .expect("failed to get local addr");

    let handle = tokio::spawn(async move {
        let (transport, peer_addr) = listener
            .accept_with_socket_addr()
            .await
            .expect("failed to accept");
        assert_eq!(peer_addr.ip(), std::net::IpAddr::from([127, 0, 0, 1]));

        let msg = transport.recv_frame().await.expect("failed to recv");
        transport.send_frame(&msg).await.expect("failed to send");
    });

    let config = TcpConfig::new().with_connect_timeout(Duration::from_secs(5));
    let factory = TcpTransportFactory::with_config(config);
    let address = Address::tcp(addr);

    let transport = factory.connect(&address).await.expect("failed to connect");

    transport.send_frame(b"test").await.expect("failed to send");
    let received = transport.recv_frame().await.expect("failed to recv");
    assert_eq!(&received, b"test");

    handle.await.expect("server task failed");
}

#[tokio::test]
async fn empty_message_works() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind listener");
    let addr = listener
        .local_socket_addr()
        .expect("failed to get local addr");

    let handle = tokio::spawn(async move {
        let (transport, _) = listener
            .accept_with_socket_addr()
            .await
            .expect("failed to accept");
        let msg = transport.recv_frame().await.expect("failed to recv");
        assert!(msg.is_empty());
        transport.send_frame(&msg).await.expect("failed to send");
    });

    let factory = TcpTransportFactory::default();
    let transport = factory
        .connect(&Address::Tcp(addr))
        .await
        .expect("failed to connect");

    transport.send_frame(b"").await.expect("failed to send");
    let received = transport.recv_frame().await.expect("failed to recv");
    assert!(received.is_empty());

    handle.await.expect("server task failed");
}

#[tokio::test]
async fn bidirectional_communication() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind listener");
    let addr = listener
        .local_socket_addr()
        .expect("failed to get local addr");

    let handle = tokio::spawn(async move {
        let (transport, _) = listener
            .accept_with_socket_addr()
            .await
            .expect("failed to accept");

        transport
            .send_frame(b"server hello")
            .await
            .expect("failed to send");

        let msg = transport.recv_frame().await.expect("failed to recv");
        assert_eq!(&msg, b"client hello");

        transport
            .send_frame(b"server 2")
            .await
            .expect("failed to send");
        let msg = transport.recv_frame().await.expect("failed to recv");
        assert_eq!(&msg, b"client 2");
    });

    let factory = TcpTransportFactory::default();
    let transport = factory
        .connect(&Address::Tcp(addr))
        .await
        .expect("failed to connect");

    let msg = transport.recv_frame().await.expect("failed to recv");
    assert_eq!(&msg, b"server hello");

    transport
        .send_frame(b"client hello")
        .await
        .expect("failed to send");

    let msg = transport.recv_frame().await.expect("failed to recv");
    assert_eq!(&msg, b"server 2");
    transport
        .send_frame(b"client 2")
        .await
        .expect("failed to send");

    handle.await.expect("server task failed");
}

#[tokio::test]
async fn detects_connection_closed() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind listener");
    let addr = listener
        .local_socket_addr()
        .expect("failed to get local addr");

    let handle = tokio::spawn(async move {
        let (transport, _) = listener
            .accept_with_socket_addr()
            .await
            .expect("failed to accept");
        drop(transport);
    });

    let factory = TcpTransportFactory::default();
    let transport = factory
        .connect(&Address::Tcp(addr))
        .await
        .expect("failed to connect");

    handle.await.expect("server task failed");

    tokio::time::sleep(Duration::from_millis(50)).await;

    let result = transport.recv_frame().await;
    assert!(result.is_err());
}

#[tokio::test]
async fn concurrent_operations() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind listener");
    let addr = listener
        .local_socket_addr()
        .expect("failed to get local addr");

    let handle = tokio::spawn(async move {
        let (transport, _) = listener
            .accept_with_socket_addr()
            .await
            .expect("failed to accept");

        for _ in 0..10 {
            let msg = transport.recv_frame().await.expect("failed to recv");
            transport.send_frame(&msg).await.expect("failed to send");
        }
    });

    let factory = TcpTransportFactory::default();
    let transport = std::sync::Arc::new(
        factory
            .connect(&Address::Tcp(addr))
            .await
            .expect("failed to connect"),
    );

    for i in 0..10u32 {
        let msg = format!("message {i}");
        transport
            .send_frame(msg.as_bytes())
            .await
            .expect("failed to send");
        let received = transport.recv_frame().await.expect("failed to recv");
        assert_eq!(received, msg.as_bytes());
    }

    handle.await.expect("server task failed");
}
