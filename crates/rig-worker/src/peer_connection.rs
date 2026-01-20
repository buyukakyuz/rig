use std::net::SocketAddr;

use rig_core::{
    Activation, ActivationMetadata, Address, DType, FramedTransport, NodeId, RequestId, Shape,
    TensorData, TransportFactory,
};
use rig_transport_tcp::{TcpConfig, TcpListener, TcpTransport, TcpTransportFactory};
use tracing::{debug, instrument};

use crate::error::WorkerError;

const HEADER_SIZE: usize = 64;

pub struct PeerConnection {
    transport: TcpTransport,
    peer_node_id: NodeId,
}

impl PeerConnection {
    #[must_use]
    pub const fn new(transport: TcpTransport, peer_node_id: NodeId) -> Self {
        Self {
            transport,
            peer_node_id,
        }
    }

    #[instrument(skip_all, fields(addr = %addr))]
    pub async fn connect(addr: &Address, peer_node_id: NodeId) -> Result<Self, WorkerError> {
        let config = TcpConfig::new().with_read_timeout(None);
        let factory = TcpTransportFactory::with_config(config);
        let transport = factory.connect(addr).await?;
        debug!(%peer_node_id, "Connected to peer");
        Ok(Self::new(transport, peer_node_id))
    }

    #[must_use]
    pub const fn peer_node_id(&self) -> NodeId {
        self.peer_node_id
    }

    #[instrument(skip_all, fields(request_id = %activation.metadata.request_id))]
    pub async fn send_activation(&self, activation: &Activation) -> Result<(), WorkerError> {
        let frame = serialize_activation(activation);
        self.transport
            .send_frame(&frame)
            .await
            .map_err(WorkerError::Transport)?;
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn recv_activation(&self) -> Result<Activation, WorkerError> {
        let frame = self
            .transport
            .recv_frame()
            .await
            .map_err(WorkerError::Transport)?;
        let activation = deserialize_activation(&frame)?;
        Ok(activation)
    }
}

impl std::fmt::Debug for PeerConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerConnection")
            .field("peer_node_id", &self.peer_node_id)
            .finish_non_exhaustive()
    }
}

pub struct PeerListener {
    listener: TcpListener,
}

impl PeerListener {
    pub async fn bind(addr: SocketAddr) -> Result<Self, WorkerError> {
        let config = TcpConfig::new().with_read_timeout(None);
        let listener = TcpListener::bind_addr_with_config(addr, config)
            .await
            .map_err(WorkerError::Transport)?;
        Ok(Self { listener })
    }

    pub fn local_addr(&self) -> Result<SocketAddr, WorkerError> {
        self.listener
            .local_socket_addr()
            .map_err(WorkerError::Transport)
    }

    pub async fn accept(&self, peer_node_id: NodeId) -> Result<PeerConnection, WorkerError> {
        let (transport, addr) = self
            .listener
            .accept_with_socket_addr()
            .await
            .map_err(WorkerError::Transport)?;
        debug!(%addr, %peer_node_id, "Accepted peer connection");
        Ok(PeerConnection::new(transport, peer_node_id))
    }
}

impl std::fmt::Debug for PeerListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerListener")
            .field("local_addr", &self.listener.local_socket_addr())
            .finish_non_exhaustive()
    }
}

fn serialize_activation(activation: &Activation) -> Vec<u8> {
    let data = activation.data.as_bytes();
    let data_len = data.len();

    let mut header = [0u8; HEADER_SIZE];

    header[0..16].copy_from_slice(activation.metadata.request_id.0.as_bytes());

    header[16..20].copy_from_slice(&activation.metadata.sequence_num.to_le_bytes());

    let dims = activation.shape.dims();
    let batch_size: u32 = dims.first().copied().unwrap_or(1).try_into().unwrap_or(1);
    let seq_len: u32 = dims.get(1).copied().unwrap_or(1).try_into().unwrap_or(1);
    let hidden_dim: u32 = dims.get(2).copied().unwrap_or(1).try_into().unwrap_or(1);

    header[20..24].copy_from_slice(&batch_size.to_le_bytes());

    header[24..28].copy_from_slice(&seq_len.to_le_bytes());

    header[28..32].copy_from_slice(&hidden_dim.to_le_bytes());

    header[32] = dtype_to_u8(activation.dtype());

    header[33] = u8::from(activation.metadata.is_prefill);

    header[36..44].copy_from_slice(&(data_len as u64).to_le_bytes());

    let checksum = compute_checksum(data);
    header[44..48].copy_from_slice(&checksum.to_le_bytes());

    let pos_count: u32 = activation
        .metadata
        .positions
        .len()
        .try_into()
        .unwrap_or(u32::MAX);
    header[56..60].copy_from_slice(&pos_count.to_le_bytes());

    let positions_bytes: Vec<u8> = activation
        .metadata
        .positions
        .iter()
        .flat_map(|p| p.to_le_bytes())
        .collect();

    let mut frame = Vec::with_capacity(HEADER_SIZE + positions_bytes.len() + data_len);
    frame.extend_from_slice(&header);
    frame.extend_from_slice(&positions_bytes);
    frame.extend_from_slice(data);

    frame
}

fn deserialize_activation(frame: &[u8]) -> Result<Activation, WorkerError> {
    if frame.len() < HEADER_SIZE {
        return Err(WorkerError::serialization(format!(
            "Frame too small: {} < {}",
            frame.len(),
            HEADER_SIZE
        )));
    }

    let header = &frame[..HEADER_SIZE];

    let mut uuid_bytes = [0u8; 16];
    uuid_bytes.copy_from_slice(&header[0..16]);
    let request_id = RequestId::from_uuid(uuid::Uuid::from_bytes(uuid_bytes));

    let sequence_num = u32::from_le_bytes(header[16..20].try_into().unwrap_or([0; 4]));

    let batch_size = u32::from_le_bytes(header[20..24].try_into().unwrap_or([1; 4])) as usize;

    let seq_len = u32::from_le_bytes(header[24..28].try_into().unwrap_or([1; 4])) as usize;

    let hidden_dim = u32::from_le_bytes(header[28..32].try_into().unwrap_or([1; 4])) as usize;

    let dtype = u8_to_dtype(header[32]);

    let is_prefill = header[33] != 0;

    #[allow(clippy::cast_possible_truncation)]
    let data_len = u64::from_le_bytes(header[36..44].try_into().unwrap_or([0; 8])) as usize;

    let expected_checksum = u32::from_le_bytes(header[44..48].try_into().unwrap_or([0; 4]));

    let pos_count = u32::from_le_bytes(header[56..60].try_into().unwrap_or([0; 4])) as usize;

    let positions_size = pos_count * 4;
    let expected_size = HEADER_SIZE + positions_size + data_len;

    if frame.len() < expected_size {
        return Err(WorkerError::serialization(format!(
            "Frame too small: {} < expected {}",
            frame.len(),
            expected_size
        )));
    }

    let positions_start = HEADER_SIZE;
    let positions_end = positions_start + positions_size;
    let positions: Vec<u32> = frame[positions_start..positions_end]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap_or([0; 4])))
        .collect();

    let data = frame[positions_end..positions_end + data_len].to_vec();

    let actual_checksum = compute_checksum(&data);
    if actual_checksum != expected_checksum {
        return Err(WorkerError::serialization(format!(
            "Checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
        )));
    }

    let shape = Shape::new(vec![batch_size, seq_len, hidden_dim]);
    let metadata = ActivationMetadata::new(request_id, sequence_num, positions, is_prefill);
    let tensor_data = TensorData::cpu(data, dtype);

    Ok(Activation::new(tensor_data, shape, metadata))
}

const fn dtype_to_u8(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::I8 => 3,
        DType::I4 => 4,
    }
}

const fn u8_to_dtype(value: u8) -> DType {
    match value {
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::I8,
        4 => DType::I4,
        _ => DType::F32,
    }
}

fn compute_checksum(data: &[u8]) -> u32 {
    let mut sum: u32 = 0;
    for chunk in data.chunks(4) {
        let mut bytes = [0u8; 4];
        bytes[..chunk.len()].copy_from_slice(chunk);
        sum = sum.wrapping_add(u32::from_le_bytes(bytes));
    }
    sum
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn test_activation() -> Activation {
        let data = vec![0u8; 256];
        let shape = Shape::new(vec![1, 2, 32]);
        let metadata = ActivationMetadata::new(RequestId::new(), 42, vec![0, 1], true);
        Activation::from_bytes(data, DType::F32, shape, metadata)
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let activation = test_activation();
        let serialized = serialize_activation(&activation);
        let deserialized = deserialize_activation(&serialized).expect("deserialize failed");

        assert_eq!(
            activation.metadata.request_id.0,
            deserialized.metadata.request_id.0
        );
        assert_eq!(
            activation.metadata.sequence_num,
            deserialized.metadata.sequence_num
        );
        assert_eq!(
            activation.metadata.is_prefill,
            deserialized.metadata.is_prefill
        );
        assert_eq!(activation.dtype(), deserialized.dtype());
        assert_eq!(activation.shape.dims(), deserialized.shape.dims());
        assert_eq!(activation.as_bytes(), deserialized.as_bytes());
    }

    #[test]
    fn deserialize_fails_on_truncated_header() {
        let short_frame = vec![0u8; 32];
        let result = deserialize_activation(&short_frame);
        assert!(result.is_err());
    }

    #[test]
    fn deserialize_fails_on_checksum_mismatch() {
        let activation = test_activation();
        let mut serialized = serialize_activation(&activation);

        let data_offset = HEADER_SIZE + 8;
        if serialized.len() > data_offset {
            serialized[data_offset] ^= 0xFF;
        }

        let result = deserialize_activation(&serialized);
        assert!(result.is_err());
    }

    #[test]
    fn dtype_conversion_roundtrip() {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::I8, DType::I4] {
            let encoded = dtype_to_u8(dtype);
            let decoded = u8_to_dtype(encoded);
            assert_eq!(dtype, decoded);
        }
    }
}
