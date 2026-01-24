#![allow(
    clippy::missing_errors_doc,
    clippy::needless_pass_by_value,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::module_name_repetitions,
    clippy::missing_const_for_fn,
    clippy::unused_self,
    clippy::unnecessary_wraps,
    clippy::match_same_arms,
    clippy::redundant_clone,
    clippy::significant_drop_tightening
)]

pub mod cache;
pub mod config;
pub mod error;
pub mod gguf_config;
pub mod kv_cache;
pub mod layers;
pub mod memory;
pub mod model;
pub mod runtime;
pub mod tokenizer;
pub mod utils;
pub mod weights;

pub mod partition;

pub use config::{Activation, EosTokenId, TransformerConfig};
pub use error::{CandleError, ConfigError, ConfigResult, Result};
pub use gguf_config::GgufConfig;
pub use runtime::{CandleRuntime, ModelFormat, detect_model_format};
pub use tokenizer::CandleTokenizer;

pub use partition::{CandlePartition, GgufPartition, UnifiedPartition};

pub use cache::{LayerKvCache, Llama3RopeConfig, PartitionKvCache, RopeCache};
pub use kv_cache::CandleKvCache;

pub use layers::{Attention, Mlp, TransformerBlock};

pub use weights::{GgufLoader, QuantizedLinear, SafetensorLoader, Weight, WeightLoader};

pub use model::{
    ActivationFn, AttentionConfig, GgufWeightNames, LlamaArchitecture, MistralArchitecture,
    ModelArchitecture, Qwen2Architecture, RopeScaling, RopeScalingConfig, SafetensorWeightNames,
    detect_architecture_config, detect_architecture_gguf,
};

pub use candle_core::Device;
