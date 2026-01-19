use std::sync::Arc;

use rig_core::error::TransportError;
use rig_core::traits::FramedTransport;
use tokio::io::{AsyncReadExt, AsyncWriteExt, ReadHalf, WriteHalf};
use tokio::net::TcpStream;
use tokio::sync::Mutex;

use crate::config::TcpConfig;

const FRAME_LENGTH_SIZE: usize = 4;

pub struct TcpTransport {
    reader: Arc<Mutex<ReadHalf<TcpStream>>>,
    writer: Arc<Mutex<WriteHalf<TcpStream>>>,
    config: TcpConfig,
}

impl TcpTransport {
    #[must_use]
    pub fn new(stream: TcpStream) -> Self {
        Self::with_config(stream, TcpConfig::default())
    }

    #[must_use]
    pub fn with_config(stream: TcpStream, config: TcpConfig) -> Self {
        let (reader, writer) = tokio::io::split(stream);
        Self {
            reader: Arc::new(Mutex::new(reader)),
            writer: Arc::new(Mutex::new(writer)),
            config,
        }
    }

    #[must_use]
    pub const fn config(&self) -> &TcpConfig {
        &self.config
    }

    async fn read_exact_bytes(
        reader: &mut ReadHalf<TcpStream>,
        n: usize,
    ) -> Result<Vec<u8>, TransportError> {
        let mut buf = vec![0u8; n];
        reader.read_exact(&mut buf).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                TransportError::Closed
            } else {
                TransportError::RecvFailed(e.to_string())
            }
        })?;
        Ok(buf)
    }
}

impl FramedTransport for TcpTransport {
    async fn send_frame(&self, frame: &[u8]) -> Result<(), TransportError> {
        let len = frame.len();

        if len > self.config.max_message_size {
            return Err(TransportError::SendFailed(format!(
                "Message size {} exceeds limit {}",
                len, self.config.max_message_size
            )));
        }

        let len_u32 = u32::try_from(len).map_err(|_| {
            TransportError::SendFailed(format!("Message size {len} exceeds u32::MAX"))
        })?;

        let mut writer = self.writer.lock().await;

        let write_future = async {
            writer.write_all(&len_u32.to_be_bytes()).await?;
            writer.write_all(frame).await?;
            writer.flush().await?;
            Ok::<(), std::io::Error>(())
        };

        if let Some(timeout) = self.config.write_timeout {
            tokio::time::timeout(timeout, write_future)
                .await
                .map_err(|_| TransportError::Timeout {
                    operation: "send_frame".to_string(),
                    duration: timeout,
                })?
                .map_err(|e| TransportError::SendFailed(e.to_string()))?;
        } else {
            write_future
                .await
                .map_err(|e| TransportError::SendFailed(e.to_string()))?;
        }

        Ok(())
    }

    async fn recv_frame(&self) -> Result<Vec<u8>, TransportError> {
        let mut reader = self.reader.lock().await;

        let read_future = async {
            let len_bytes = Self::read_exact_bytes(&mut reader, FRAME_LENGTH_SIZE).await?;

            let len_array: [u8; 4] = len_bytes.try_into().map_err(|_| {
                TransportError::RecvFailed("Failed to read length prefix".to_string())
            })?;
            let len = u32::from_be_bytes(len_array) as usize;

            if len > self.config.max_message_size {
                return Err(TransportError::RecvFailed(format!(
                    "Message size {} exceeds limit {}",
                    len, self.config.max_message_size
                )));
            }

            Self::read_exact_bytes(&mut reader, len).await
        };

        if let Some(timeout) = self.config.read_timeout {
            tokio::time::timeout(timeout, read_future)
                .await
                .map_err(|_| TransportError::Timeout {
                    operation: "recv_frame".to_string(),
                    duration: timeout,
                })?
        } else {
            read_future.await
        }
    }
}

impl std::fmt::Debug for TcpTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TcpTransport")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn send_and_receive_small_message() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind listener");
        let addr = listener.local_addr().expect("failed to get local addr");

        let handle = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("failed to accept");
            let transport = TcpTransport::new(stream);
            let frame = transport.recv_frame().await.expect("failed to recv");
            transport.send_frame(&frame).await.expect("failed to send");
        });

        let stream = TcpStream::connect(addr).await.expect("failed to connect");
        let transport = TcpTransport::new(stream);

        let original = b"hello, world!";
        transport
            .send_frame(original)
            .await
            .expect("failed to send");

        let received = transport.recv_frame().await.expect("failed to recv");
        assert_eq!(&received, original);

        handle.await.expect("server task failed");
    }

    #[tokio::test]
    async fn send_and_receive_empty_message() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind listener");
        let addr = listener.local_addr().expect("failed to get local addr");

        let handle = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("failed to accept");
            let transport = TcpTransport::new(stream);
            let frame = transport.recv_frame().await.expect("failed to recv");
            transport.send_frame(&frame).await.expect("failed to send");
        });

        let stream = TcpStream::connect(addr).await.expect("failed to connect");
        let transport = TcpTransport::new(stream);

        let original: &[u8] = b"";
        transport
            .send_frame(original)
            .await
            .expect("failed to send");

        let received = transport.recv_frame().await.expect("failed to recv");
        assert!(received.is_empty());

        handle.await.expect("server task failed");
    }

    #[tokio::test]
    async fn rejects_oversized_message() {
        let config = TcpConfig::new().with_max_message_size(100);

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind listener");
        let addr = listener.local_addr().expect("failed to get local addr");

        let _handle = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("failed to accept");
            let _transport = TcpTransport::with_config(stream, TcpConfig::default());
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        });

        let stream = TcpStream::connect(addr).await.expect("failed to connect");
        let transport = TcpTransport::with_config(stream, config);

        let oversized = vec![0u8; 200];
        let result = transport.send_frame(&oversized).await;
        assert!(result.is_err());
        assert!(matches!(result, Err(TransportError::SendFailed(_))));
    }
}
