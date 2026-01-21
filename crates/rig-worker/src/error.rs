use std::path::PathBuf;

use rig_core::{CodecError, PartitionError, RuntimeError, TransportError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("Registration rejected: {0}")]
    RegistrationRejected(String),

    #[error("Worker not registered with coordinator")]
    NotRegistered,

    #[error("Unexpected response from coordinator: {0}")]
    UnexpectedResponse(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model file not found at path: {}", .0.display())]
    ModelPathNotFound(PathBuf),

    #[error("No assignment received")]
    NoAssignment,

    #[error("Runtime error: {0}")]
    Runtime(#[from] RuntimeError),

    #[error("Partition error: {0}")]
    Partition(#[from] PartitionError),

    #[error("Transport error: {0}")]
    Transport(#[from] TransportError),

    #[error("Codec error: {0}")]
    Codec(#[from] CodecError),

    #[error("Peer connection error: {0}")]
    PeerConnection(String),

    #[error("Activation serialization error: {0}")]
    Serialization(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Partition processing error: {0}")]
    PartitionProcessing(String),

    #[error("Coordinator error (code {code}): {message}")]
    CoordinatorError { code: u32, message: String },

    #[error("Shutdown requested")]
    Shutdown,

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

impl WorkerError {
    pub fn peer_connection(msg: impl Into<String>) -> Self {
        Self::PeerConnection(msg.into())
    }

    pub fn serialization(msg: impl Into<String>) -> Self {
        Self::Serialization(msg.into())
    }

    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    pub fn partition_processing(msg: impl Into<String>) -> Self {
        Self::PartitionProcessing(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = WorkerError::RegistrationRejected("cluster full".to_string());
        assert_eq!(err.to_string(), "Registration rejected: cluster full");

        let err = WorkerError::NotRegistered;
        assert_eq!(err.to_string(), "Worker not registered with coordinator");

        let err = WorkerError::CoordinatorError {
            code: 1001,
            message: "Maximum nodes reached".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Coordinator error (code 1001): Maximum nodes reached"
        );
    }

    #[test]
    fn helper_methods() {
        let err = WorkerError::peer_connection("connection failed");
        assert!(matches!(err, WorkerError::PeerConnection(_)));

        let err = WorkerError::serialization("invalid format");
        assert!(matches!(err, WorkerError::Serialization(_)));

        let err = WorkerError::config("missing field");
        assert!(matches!(err, WorkerError::Config(_)));

        let err = WorkerError::partition_processing("forward pass failed");
        assert!(matches!(err, WorkerError::PartitionProcessing(_)));
    }
}
