use candle_core::quantized::gguf_file;
use serde::Deserialize;
use std::path::Path;

use crate::cache::Llama3RopeConfig;
use crate::error::{ConfigError, ConfigResult};

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScalingConfig {
    pub factor: Option<f32>,
    pub low_freq_factor: Option<f32>,
    pub high_freq_factor: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
    pub rope_type: Option<String>,
}

impl RopeScalingConfig {
    #[must_use]
    pub fn to_llama3_config(&self) -> Option<Llama3RopeConfig> {
        let is_llama3 = self
            .rope_type
            .as_ref()
            .is_some_and(|t| t.eq_ignore_ascii_case("llama3"));

        if is_llama3 || (self.factor.is_some() && self.original_max_position_embeddings.is_some()) {
            Some(Llama3RopeConfig::new(
                self.factor.unwrap_or(8.0),
                self.low_freq_factor.unwrap_or(1.0),
                self.high_freq_factor.unwrap_or(4.0),
                self.original_max_position_embeddings.unwrap_or(8192),
            ))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct PartialModelConfig {
    rope_scaling: Option<RopeScalingConfig>,
}

#[derive(Debug, Clone)]
pub struct GgufConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<Llama3RopeConfig>,
}

impl GgufConfig {
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn from_gguf_content(content: &gguf_file::Content) -> ConfigResult<Self> {
        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().map(ToString::to_string))
            .unwrap_or_else(|| "llama".to_string());

        let md_get = |suffix: &str| {
            let key = format!("{arch}.{suffix}");
            content
                .metadata
                .get(&key)
                .ok_or(ConfigError::MissingField(key))
        };

        let num_hidden_layers =
            md_get("block_count")?
                .to_u32()
                .map_err(|e| ConfigError::InvalidValue {
                    field: format!("{arch}.block_count"),
                    reason: e.to_string(),
                })? as usize;

        let hidden_size =
            md_get("embedding_length")?
                .to_u32()
                .map_err(|e| ConfigError::InvalidValue {
                    field: format!("{arch}.embedding_length"),
                    reason: e.to_string(),
                })? as usize;

        let intermediate_size =
            md_get("feed_forward_length")?
                .to_u32()
                .map_err(|e| ConfigError::InvalidValue {
                    field: format!("{arch}.feed_forward_length"),
                    reason: e.to_string(),
                })? as usize;

        let num_attention_heads =
            md_get("attention.head_count")?
                .to_u32()
                .map_err(|e| ConfigError::InvalidValue {
                    field: format!("{arch}.attention.head_count"),
                    reason: e.to_string(),
                })? as usize;

        let num_key_value_heads =
            md_get("attention.head_count_kv")?
                .to_u32()
                .map_err(|e| ConfigError::InvalidValue {
                    field: format!("{arch}.attention.head_count_kv"),
                    reason: e.to_string(),
                })? as usize;

        let rope_theta = content
            .metadata
            .get(&format!("{arch}.rope.freq_base"))
            .and_then(|v| v.to_f32().ok())
            .map_or(10000.0, f64::from);

        let rms_norm_eps = f64::from(
            md_get("attention.layer_norm_rms_epsilon")?
                .to_f32()
                .map_err(|e| ConfigError::InvalidValue {
                    field: format!("{arch}.attention.layer_norm_rms_epsilon"),
                    reason: e.to_string(),
                })?,
        );

        let max_position_embeddings = content
            .metadata
            .get(&format!("{arch}.context_length"))
            .and_then(|v| v.to_u32().ok())
            .map_or(4096, |v| v as usize);

        let bos_token_id = content
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.to_u32().ok());

        let eos_token_id = content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok());

        Ok(Self {
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            rope_theta,
            rms_norm_eps,
            max_position_embeddings,
            bos_token_id,
            eos_token_id,
            rope_scaling: None,
        })
    }

    pub fn from_file(path: impl AsRef<Path>) -> ConfigResult<Self> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path)?;
        let content =
            gguf_file::Content::read(&mut file).map_err(|e| ConfigError::InvalidValue {
                field: "gguf_file".to_string(),
                reason: e.to_string(),
            })?;
        Self::from_gguf_content(&content)
    }

    pub fn load_rope_scaling_from_config(
        config_path: impl AsRef<Path>,
    ) -> Option<Llama3RopeConfig> {
        let content = std::fs::read_to_string(config_path.as_ref()).ok()?;
        let config: PartialModelConfig = serde_json::from_str(&content).ok()?;
        config.rope_scaling.and_then(|rs| rs.to_llama3_config())
    }

    #[must_use]
    pub const fn with_rope_scaling(mut self, rope_scaling: Option<Llama3RopeConfig>) -> Self {
        self.rope_scaling = rope_scaling;
        self
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_load_tiny_llama_gguf_config() {
        let config = GgufConfig::from_file("../../models/tiny-llama-q4/model.gguf")
            .expect("Failed to load GGUF config");

        assert_eq!(config.num_hidden_layers, 22);
        println!("GgufConfig: {config:#?}");
    }
}
