use std::path::PathBuf;
use std::time::Duration;

use thiserror::Error;

use crate::types::{NodeId, RequestId};

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("Model not found: {0}")]
    ModelNotFound(PathBuf),

    #[error("Failed to load model '{model}': {reason}")]
    LoadFailed { model: String, reason: String },

    #[error("Unsupported data type: {0}")]
    UnsupportedDtype(String),

    #[error("Out of memory: required {required} bytes, available {available} bytes")]
    OutOfMemory { required: u64, available: u64 },

    #[error("Runtime error: {0}")]
    Internal(String),
}

#[derive(Debug, Error)]
pub enum PartitionError {
    #[error("Forward pass failed: {0}")]
    ForwardFailed(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("KV cache error: {0}")]
    CacheError(#[from] CacheError),
}

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("Connection failed to '{addr}': {reason}")]
    ConnectionFailed { addr: String, reason: String },

    #[error("Send failed: {0}")]
    SendFailed(String),

    #[error("Receive failed: {0}")]
    RecvFailed(String),

    #[error("Operation '{operation}' timed out after {duration:?}")]
    Timeout {
        operation: String,
        duration: Duration,
    },

    #[error("Connection closed")]
    Closed,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("Encode failed: {0}")]
    EncodeFailed(String),

    #[error("Decode failed: {0}")]
    DecodeFailed(String),

    #[error("Message too large: {size} bytes exceeds limit of {limit} bytes")]
    MessageTooLarge { size: usize, limit: usize },
}

#[derive(Debug, Error)]
pub enum CoordinationError {
    #[error("Cluster error: {0}")]
    Cluster(#[from] ClusterError),

    #[error("Coordination error: {0}")]
    Coord(#[from] CoordError),

    #[error("Discovery error: {0}")]
    Discovery(#[from] DiscoveryError),

    #[error("Health error: {0}")]
    Health(#[from] HealthError),
}

#[derive(Debug, Error)]
pub enum ClusterError {
    #[error("Not joined to cluster")]
    NotJoined,

    #[error("Failed to join cluster: {0}")]
    JoinFailed(String),

    #[error("Failed to leave cluster: {0}")]
    LeaveFailed(String),

    #[error("Cluster state unavailable: {0}")]
    StateUnavailable(String),
}

#[derive(Debug, Error)]
pub enum CoordError {
    #[error("No assignment available")]
    NoAssignment,

    #[error("Failed to report status: {0}")]
    ReportFailed(String),

    #[error("Coordinator unreachable: {0}")]
    CoordinatorUnreachable(String),

    #[error("Node not registered")]
    NotRegistered,

    #[error("Maximum nodes ({max}) reached")]
    MaxNodesReached { max: usize },

    #[error("Node not found: {0}")]
    NodeNotFound(NodeId),

    #[error("Pipeline not found: {0}")]
    PipelineNotFound(crate::types::PipelineId),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Request not found: {0}")]
    RequestNotFound(RequestId),

    #[error("Transport error: {0}")]
    Transport(#[from] TransportError),

    #[error("Codec error: {0}")]
    Codec(#[from] CodecError),
}

#[derive(Debug, Error)]
pub enum DiscoveryError {
    #[error("Peer not found: {0}")]
    PeerNotFound(NodeId),

    #[error("Failed to resolve peer '{node_id}': {reason}")]
    ResolveFailed { node_id: NodeId, reason: String },

    #[error("No neighbors available")]
    NoNeighbors,
}

#[derive(Debug, Error)]
pub enum HealthError {
    #[error("Health reporter not started")]
    NotStarted,

    #[error("Failed to report health: {0}")]
    ReportFailed(String),
}

#[derive(Debug, Error)]
pub enum CacheError {
    #[error("Request {0} already has allocated cache slot")]
    AlreadyAllocated(RequestId),

    #[error("No cache slot for request {0}")]
    NotAllocated(RequestId),

    #[error("Out of cache memory: required {required} bytes, available {available} bytes")]
    OutOfMemory { required: u64, available: u64 },

    #[error("Maximum cache slots ({max}) reached")]
    MaxSlotsReached { max: usize },
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Config file not found: {0}")]
    NotFound(PathBuf),

    #[error("Failed to parse config: {0}")]
    ParseError(String),

    #[error("Invalid config value for '{key}': {reason}")]
    InvalidValue { key: String, reason: String },

    #[error("Missing required config: {0}")]
    MissingRequired(String),
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("Tokenization failed: {0}")]
    EncodeFailed(String),

    #[error("Detokenization failed: {0}")]
    DecodeFailed(String),

    #[error("Invalid token ID: {0}")]
    InvalidToken(u32),
}
