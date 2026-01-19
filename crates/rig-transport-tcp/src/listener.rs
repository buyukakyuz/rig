use std::net::SocketAddr;

use rig_core::error::TransportError;
use rig_core::traits::Listener;
use rig_core::types::Address;
use tokio::net::TcpListener as TokioTcpListener;

use crate::config::TcpConfig;
use crate::transport::TcpTransport;

pub struct TcpListener {
    inner: TokioTcpListener,
    config: TcpConfig,
}

impl TcpListener {
    pub async fn bind(addr: &str) -> Result<Self, TransportError> {
        Self::bind_with_config(addr, TcpConfig::default()).await
    }

    pub async fn bind_addr(addr: SocketAddr) -> Result<Self, TransportError> {
        Self::bind_addr_with_config(addr, TcpConfig::default()).await
    }

    pub async fn bind_with_config(addr: &str, config: TcpConfig) -> Result<Self, TransportError> {
        let inner = TokioTcpListener::bind(addr).await?;
        Ok(Self { inner, config })
    }

    pub async fn bind_addr_with_config(
        addr: SocketAddr,
        config: TcpConfig,
    ) -> Result<Self, TransportError> {
        let inner = TokioTcpListener::bind(addr).await?;
        Ok(Self { inner, config })
    }

    pub async fn accept_with_socket_addr(
        &self,
    ) -> Result<(TcpTransport, SocketAddr), TransportError> {
        let (stream, addr) = self.inner.accept().await?;

        if self.config.nodelay {
            stream.set_nodelay(true)?;
        }

        Ok((TcpTransport::with_config(stream, self.config.clone()), addr))
    }

    pub fn local_socket_addr(&self) -> Result<SocketAddr, TransportError> {
        self.inner.local_addr().map_err(TransportError::from)
    }

    #[must_use]
    pub const fn config(&self) -> &TcpConfig {
        &self.config
    }
}

impl std::fmt::Debug for TcpListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TcpListener")
            .field("local_addr", &self.inner.local_addr())
            .field("config", &self.config)
            .finish()
    }
}

impl Listener for TcpListener {
    type Transport = TcpTransport;

    async fn accept(&self) -> Result<(Self::Transport, Address), TransportError> {
        let (stream, addr) = self.inner.accept().await?;

        if self.config.nodelay {
            stream.set_nodelay(true)?;
        }

        Ok((
            TcpTransport::with_config(stream, self.config.clone()),
            Address::from(addr),
        ))
    }

    fn local_addr(&self) -> Result<Address, TransportError> {
        self.inner
            .local_addr()
            .map(Address::from)
            .map_err(TransportError::from)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use rig_core::traits::FramedTransport;
    use tokio::net::TcpStream;

    #[tokio::test]
    async fn bind_and_accept_with_socket_addr() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind");
        let addr = listener
            .local_socket_addr()
            .expect("failed to get local addr");

        let handle =
            tokio::spawn(async move { TcpStream::connect(addr).await.expect("failed to connect") });

        let (transport, peer_addr) = listener
            .accept_with_socket_addr()
            .await
            .expect("failed to accept");
        assert_eq!(peer_addr.ip(), std::net::IpAddr::from([127, 0, 0, 1]));

        let client_stream = handle.await.expect("client task failed");
        let client_transport = TcpTransport::new(client_stream);

        client_transport
            .send_frame(b"test")
            .await
            .expect("failed to send");

        let received = transport.recv_frame().await.expect("failed to recv");
        assert_eq!(&received, b"test");
    }

    #[tokio::test]
    async fn bind_and_accept_with_trait() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind");
        let local = Listener::local_addr(&listener).expect("failed to get local addr");
        let socket_addr = local.as_tcp().expect("expected TCP address");

        let addr_clone = *socket_addr;
        let handle = tokio::spawn(async move {
            TcpStream::connect(addr_clone)
                .await
                .expect("failed to connect")
        });

        let (transport, peer_addr) = Listener::accept(&listener).await.expect("failed to accept");
        let peer_socket = peer_addr.as_tcp().expect("expected TCP address");
        assert_eq!(peer_socket.ip(), std::net::IpAddr::from([127, 0, 0, 1]));

        let client_stream = handle.await.expect("client task failed");
        let client_transport = TcpTransport::new(client_stream);

        client_transport
            .send_frame(b"test")
            .await
            .expect("failed to send");

        let received = transport.recv_frame().await.expect("failed to recv");
        assert_eq!(&received, b"test");
    }

    #[tokio::test]
    async fn bind_with_socket_addr() {
        let addr: SocketAddr = "127.0.0.1:0".parse().expect("failed to parse addr");
        let listener = TcpListener::bind_addr(addr).await.expect("failed to bind");

        let local_addr = listener
            .local_socket_addr()
            .expect("failed to get local addr");
        assert_eq!(local_addr.ip(), std::net::IpAddr::from([127, 0, 0, 1]));
        assert_ne!(local_addr.port(), 0);

        let trait_addr = Listener::local_addr(&listener).expect("failed to get local addr");
        assert!(trait_addr.is_tcp());
        assert_eq!(trait_addr.as_tcp(), Some(&local_addr));
    }
}
