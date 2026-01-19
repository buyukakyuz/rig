use std::marker::PhantomData;

use rig_core::error::{CodecError, TransportError};
use rig_core::traits::{Codec, FramedTransport};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ChannelError {
    #[error("Transport error: {0}")]
    Transport(#[from] TransportError),

    #[error("Codec error: {0}")]
    Codec(#[from] CodecError),
}

pub struct MessageChannel<M, T, C> {
    transport: T,
    codec: C,
    _phantom: PhantomData<M>,
}

impl<M, T, C> MessageChannel<M, T, C> {
    pub const fn new(transport: T, codec: C) -> Self {
        Self {
            transport,
            codec,
            _phantom: PhantomData,
        }
    }

    pub const fn transport(&self) -> &T {
        &self.transport
    }

    pub const fn codec(&self) -> &C {
        &self.codec
    }

    pub fn into_parts(self) -> (T, C) {
        (self.transport, self.codec)
    }
}

impl<M, T, C> MessageChannel<M, T, C>
where
    T: FramedTransport,
    C: Codec<M>,
{
    #[allow(clippy::future_not_send)]
    pub async fn send(&self, msg: &M) -> Result<(), ChannelError> {
        let bytes = self.codec.encode(msg)?;
        self.transport.send_frame(&bytes).await?;
        Ok(())
    }

    #[allow(clippy::future_not_send)]
    pub async fn recv(&self) -> Result<M, ChannelError> {
        let bytes = self.transport.recv_frame().await?;
        let msg = self.codec.decode(&bytes)?;
        Ok(msg)
    }
}

impl<M, T: std::fmt::Debug, C: std::fmt::Debug> std::fmt::Debug for MessageChannel<M, T, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MessageChannel")
            .field("transport", &self.transport)
            .field("codec", &self.codec)
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;

    struct MockTransport {
        sent: Arc<tokio::sync::Mutex<Vec<Vec<u8>>>>,
        to_recv: Arc<tokio::sync::Mutex<Vec<Vec<u8>>>>,
    }

    impl MockTransport {
        fn new() -> Self {
            Self {
                sent: Arc::new(tokio::sync::Mutex::new(Vec::new())),
                to_recv: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            }
        }

        fn with_recv_data(data: Vec<Vec<u8>>) -> Self {
            Self {
                sent: Arc::new(tokio::sync::Mutex::new(Vec::new())),
                to_recv: Arc::new(tokio::sync::Mutex::new(data)),
            }
        }

        async fn get_sent(&self) -> Vec<Vec<u8>> {
            self.sent.lock().await.clone()
        }
    }

    impl FramedTransport for MockTransport {
        async fn send_frame(&self, frame: &[u8]) -> Result<(), TransportError> {
            self.sent.lock().await.push(frame.to_vec());
            Ok(())
        }

        async fn recv_frame(&self) -> Result<Vec<u8>, TransportError> {
            self.to_recv
                .lock()
                .await
                .pop()
                .ok_or(TransportError::Closed)
        }
    }

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct TestMessage {
        id: u32,
        text: String,
    }

    #[tokio::test]
    async fn send_encodes_and_transmits() {
        let transport = MockTransport::new();
        let codec = crate::BincodeCodec::new();
        let channel: MessageChannel<TestMessage, _, _> = MessageChannel::new(transport, codec);

        let msg = TestMessage {
            id: 42,
            text: "hello".to_string(),
        };

        channel.send(&msg).await.expect("send failed");

        let sent = channel.transport().get_sent().await;
        assert_eq!(sent.len(), 1);

        let decoded: TestMessage = bincode::deserialize(&sent[0]).expect("decode failed");
        assert_eq!(decoded, msg);
    }

    #[tokio::test]
    async fn recv_receives_and_decodes() {
        let msg = TestMessage {
            id: 123,
            text: "world".to_string(),
        };
        let encoded = bincode::serialize(&msg).expect("encode failed");

        let transport = MockTransport::with_recv_data(vec![encoded]);
        let codec = crate::BincodeCodec::new();
        let channel: MessageChannel<TestMessage, _, _> = MessageChannel::new(transport, codec);

        let received = channel.recv().await.expect("recv failed");
        assert_eq!(received, msg);
    }

    #[tokio::test]
    async fn recv_returns_error_on_empty() {
        let transport = MockTransport::new();
        let codec = crate::BincodeCodec::new();
        let channel: MessageChannel<TestMessage, _, _> = MessageChannel::new(transport, codec);

        let result = channel.recv().await;
        assert!(result.is_err());
        assert!(matches!(result, Err(ChannelError::Transport(_))));
    }

    #[test]
    fn channel_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MessageChannel<TestMessage, MockTransport, crate::BincodeCodec>>();
    }
}
