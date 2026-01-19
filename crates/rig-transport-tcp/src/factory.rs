use rig_core::error::TransportError;
use rig_core::traits::{LatencyClass, TransportCharacteristics, TransportFactory};
use rig_core::types::Address;
use tokio::net::TcpStream;

use crate::config::TcpConfig;
use crate::listener::TcpListener;
use crate::transport::TcpTransport;

#[derive(Debug, Clone, Default)]
pub struct TcpTransportFactory {
    config: TcpConfig,
}

impl TcpTransportFactory {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn with_config(config: TcpConfig) -> Self {
        Self { config }
    }

    pub async fn listen_str(&self, addr: &str) -> Result<TcpListener, TransportError> {
        TcpListener::bind_with_config(addr, self.config.clone()).await
    }

    #[must_use]
    pub const fn config(&self) -> &TcpConfig {
        &self.config
    }
}

impl TransportFactory for TcpTransportFactory {
    type Transport = TcpTransport;
    type Listener = TcpListener;

    async fn connect(&self, addr: &Address) -> Result<Self::Transport, TransportError> {
        let socket_addr = match addr {
            Address::Tcp(socket_addr) => *socket_addr,
            Address::Unix(_) => {
                return Err(TransportError::ConnectionFailed {
                    addr: addr.to_string(),
                    reason: "TCP factory cannot connect to Unix socket address".to_string(),
                });
            }
        };

        let stream =
            tokio::time::timeout(self.config.connect_timeout, TcpStream::connect(socket_addr))
                .await
                .map_err(|_| TransportError::Timeout {
                    operation: "connect".to_string(),
                    duration: self.config.connect_timeout,
                })?
                .map_err(|e| TransportError::ConnectionFailed {
                    addr: addr.to_string(),
                    reason: e.to_string(),
                })?;

        if self.config.nodelay {
            stream.set_nodelay(true)?;
        }

        Ok(TcpTransport::with_config(stream, self.config.clone()))
    }

    async fn listen(&self, addr: &Address) -> Result<Self::Listener, TransportError> {
        match addr {
            Address::Tcp(socket_addr) => {
                TcpListener::bind_addr_with_config(*socket_addr, self.config.clone()).await
            }
            Address::Unix(_) => Err(TransportError::ConnectionFailed {
                addr: addr.to_string(),
                reason: "TCP factory cannot bind to Unix socket address".to_string(),
            }),
        }
    }

    fn characteristics(&self) -> TransportCharacteristics {
        TransportCharacteristics {
            latency_class: LatencyClass::Lan,
            reliable: true,
            ordered: true,
            max_message_size: Some(self.config.max_message_size),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use rig_core::traits::{FramedTransport, Listener};
    use std::net::SocketAddr;

    #[tokio::test]
    async fn connect_to_listener_using_convenience_method() {
        let factory = TcpTransportFactory::new();
        let listener = factory
            .listen_str("127.0.0.1:0")
            .await
            .expect("failed to bind");
        let addr = listener
            .local_socket_addr()
            .expect("failed to get local addr");

        let handle = tokio::spawn(async move {
            listener
                .accept_with_socket_addr()
                .await
                .expect("failed to accept")
        });

        let address = Address::tcp(addr);
        let transport = factory.connect(&address).await.expect("failed to connect");

        transport
            .send_frame(b"hello")
            .await
            .expect("failed to send");

        let (server_transport, _) = handle.await.expect("acceptor task failed");
        let received = server_transport.recv_frame().await.expect("failed to recv");
        assert_eq!(&received, b"hello");
    }

    #[tokio::test]
    async fn connect_to_listener_using_trait() {
        let factory = TcpTransportFactory::new();
        let bind_addr = Address::tcp("127.0.0.1:0".parse().expect("parse"));
        let listener = factory.listen(&bind_addr).await.expect("failed to bind");
        let local = Listener::local_addr(&listener).expect("failed to get local addr");

        let handle =
            tokio::spawn(
                async move { Listener::accept(&listener).await.expect("failed to accept") },
            );

        let transport = factory.connect(&local).await.expect("failed to connect");

        transport
            .send_frame(b"hello")
            .await
            .expect("failed to send");

        let (server_transport, _) = handle.await.expect("acceptor task failed");
        let received = server_transport.recv_frame().await.expect("failed to recv");
        assert_eq!(&received, b"hello");
    }

    #[tokio::test]
    async fn listen_rejects_unix_address() {
        let factory = TcpTransportFactory::new();
        let address = Address::unix("/tmp/test.sock");

        let result = factory.listen(&address).await;
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(TransportError::ConnectionFailed { .. })
        ));
    }

    #[tokio::test]
    async fn connect_timeout_on_unreachable() {
        let factory = TcpTransportFactory::with_config(
            TcpConfig::new().with_connect_timeout(std::time::Duration::from_millis(100)),
        );

        let addr: SocketAddr = "10.255.255.1:12345".parse().expect("failed to parse addr");
        let address = Address::tcp(addr);

        let result = factory.connect(&address).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn connect_rejects_unix_address() {
        let factory = TcpTransportFactory::new();
        let address = Address::unix("/tmp/test.sock");

        let result = factory.connect(&address).await;
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(TransportError::ConnectionFailed { .. })
        ));
    }

    #[test]
    fn characteristics_are_correct() {
        let factory = TcpTransportFactory::new();
        let chars = factory.characteristics();

        assert_eq!(chars.latency_class, LatencyClass::Lan);
        assert!(chars.reliable);
        assert!(chars.ordered);
        assert!(chars.max_message_size.is_some());
    }
}
