use crate::error::TransportError;
use crate::types::Address;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyClass {
    SameProcess,
    SameMachine,
    Lan,
    Internet,
    HighLatency,
}

#[derive(Debug, Clone)]
pub struct TransportCharacteristics {
    pub latency_class: LatencyClass,
    pub reliable: bool,
    pub ordered: bool,
    pub max_message_size: Option<usize>,
}

impl Default for TransportCharacteristics {
    fn default() -> Self {
        Self {
            latency_class: LatencyClass::Lan,
            reliable: true,
            ordered: true,
            max_message_size: None,
        }
    }
}

pub trait ByteTransport: Send + Sync {
    fn send(
        &self,
        data: &[u8],
    ) -> impl std::future::Future<Output = Result<(), TransportError>> + Send;
    fn recv(
        &self,
        buf: &mut [u8],
    ) -> impl std::future::Future<Output = Result<usize, TransportError>> + Send;
    fn close(&self) -> Result<(), TransportError>;
}
pub trait FramedTransport: Send + Sync {
    fn send_frame(
        &self,
        frame: &[u8],
    ) -> impl std::future::Future<Output = Result<(), TransportError>> + Send;
    fn recv_frame(
        &self,
    ) -> impl std::future::Future<Output = Result<Vec<u8>, TransportError>> + Send;
}

pub trait Listener: Send + Sync {
    type Transport: FramedTransport + Send;
    fn accept(
        &self,
    ) -> impl std::future::Future<Output = Result<(Self::Transport, Address), TransportError>> + Send;
    fn local_addr(&self) -> Result<Address, TransportError>;
}
pub trait TransportFactory: Send + Sync {
    type Transport: FramedTransport + Send;
    type Listener: Listener<Transport = Self::Transport> + Send;
    fn connect(
        &self,
        addr: &Address,
    ) -> impl std::future::Future<Output = Result<Self::Transport, TransportError>> + Send;
    fn listen(
        &self,
        addr: &Address,
    ) -> impl std::future::Future<Output = Result<Self::Listener, TransportError>> + Send;
    fn characteristics(&self) -> TransportCharacteristics;
}
