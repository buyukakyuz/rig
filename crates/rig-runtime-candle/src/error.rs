use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CandleError {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Config error: {0}")]
    Config(#[from] ConfigError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model not found: {0}")]
    ModelNotFound(PathBuf),

    #[error("No safetensor files found in: {0}")]
    NoSafetensorFiles(PathBuf),

    #[error("Weight not found: {0}")]
    WeightNotFound(String),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Invalid layer range {start}..{end} for model with {total} layers")]
    InvalidLayerRange {
        start: usize,
        end: usize,
        total: usize,
    },

    #[error("DType conversion error: {0}")]
    DTypeConversion(String),

    #[error("Failed to load tokenizer: {0}")]
    TokenizerLoad(String),

    #[error("Tokenization failed: {0}")]
    TokenizationFailed(String),

    #[error("Tokenizer config file not found: {0}")]
    TokenizerConfigNotFound(PathBuf),

    #[error("Tokenizer config missing required field: {0}")]
    TokenizerConfigMissingField(&'static str),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Unknown model architecture: {0}")]
    UnknownArchitecture(String),

    #[error("Architecture detection failed: {0}")]
    ArchitectureDetectionFailed(String),
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid value for field '{field}': {reason}")]
    InvalidValue { field: String, reason: String },
}

pub type Result<T> = std::result::Result<T, CandleError>;
pub type ConfigResult<T> = std::result::Result<T, ConfigError>;
