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

pub mod attention;
pub mod block;
pub mod cache;
pub mod config;
pub mod error;
pub mod kv_cache;
pub mod memory;
pub mod mlp;
pub mod partition;
pub mod runtime;

pub use config::{Activation, EosTokenId, TransformerConfig};
pub use error::{CandleError, ConfigError, ConfigResult, Result};
pub use partition::CandlePartition;
pub use runtime::CandleRuntime;

pub use cache::{LayerKvCache, PartitionKvCache, RopeCache};
pub use kv_cache::CandleKvCache;

pub use attention::CausalSelfAttention;
pub use block::TransformerBlock;
pub use mlp::Mlp;
