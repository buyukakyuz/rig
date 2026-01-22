use std::path::Path;

use serde::Deserialize;

use crate::error::ConfigResult;

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    Silu,
    Gelu,
    #[serde(rename = "gelu_new")]
    GeluNew,
    Relu,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TransformerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub hidden_act: Activation,

    #[serde(default)]
    pub bos_token_id: Option<u32>,

    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

impl EosTokenId {
    #[must_use]
    pub fn to_vec(&self) -> Vec<u32> {
        match self {
            Self::Single(id) => vec![*id],
            Self::Multiple(ids) => ids.clone(),
        }
    }

    #[must_use]
    pub fn contains(&self, token_id: u32) -> bool {
        match self {
            Self::Single(id) => *id == token_id,
            Self::Multiple(ids) => ids.contains(&token_id),
        }
    }
}

fn default_rope_theta() -> f64 {
    10000.0
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_max_position_embeddings() -> usize {
    4096
}

impl TransformerConfig {
    pub fn from_file(path: impl AsRef<Path>) -> ConfigResult<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Self::from_json(&content)
    }

    pub fn from_json(json: &str) -> ConfigResult<Self> {
        let config: Self = serde_json::from_str(json)?;
        Ok(config)
    }

    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    #[must_use]
    pub fn num_queries_per_kv(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads()
    }

    #[must_use]
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads() < self.num_attention_heads
    }

    #[must_use]
    pub fn is_mqa(&self) -> bool {
        self.num_kv_heads() == 1
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerConfig {
    #[serde(default = "default_add_bos_token")]
    pub add_bos_token: bool,
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default = "default_eos_token")]
    pub eos_token: Option<String>,
}

fn default_add_bos_token() -> bool {
    true
}

fn default_eos_token() -> Option<String> {
    None
}

impl TokenizerConfig {
    pub fn from_file(path: impl AsRef<Path>) -> ConfigResult<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Self::from_json(&content)
    }

    pub fn from_json(json: &str) -> ConfigResult<Self> {
        let config: Self = serde_json::from_str(json)?;
        Ok(config)
    }

    #[must_use]
    pub fn eos_token_str(&self) -> &str {
        self.eos_token.as_deref().unwrap_or("</s>")
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_llama_config() {
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 32000,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": false,
            "hidden_act": "silu"
        }"#;

        let config = TransformerConfig::from_json(json).unwrap_or_else(|e| {
            panic!("Failed to parse config: {e}");
        });

        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads(), 32);
        assert_eq!(config.head_dim(), 128);
        assert!(!config.is_gqa());
    }

    #[test]
    fn test_parse_gqa_config() {
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "rope_theta": 500000.0,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 8192,
            "tie_word_embeddings": true
        }"#;

        let config = TransformerConfig::from_json(json).unwrap_or_else(|e| {
            panic!("Failed to parse config: {e}");
        });

        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads(), 8);
        assert_eq!(config.num_queries_per_kv(), 4);
        assert!(config.is_gqa());
        assert!(!config.is_mqa());
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn test_default_values() {
        let json = r#"{
            "hidden_size": 2048,
            "intermediate_size": 5504,
            "num_hidden_layers": 22,
            "num_attention_heads": 16,
            "vocab_size": 32000
        }"#;

        let config = TransformerConfig::from_json(json).unwrap_or_else(|e| {
            panic!("Failed to parse config: {e}");
        });

        assert_eq!(config.num_kv_heads(), 16);
        assert!((config.rope_theta - 10000.0).abs() < f64::EPSILON);
        assert!((config.rms_norm_eps - 1e-6).abs() < f64::EPSILON);
        assert_eq!(config.max_position_embeddings, 4096);
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.hidden_act, Activation::Silu);
    }

    #[test]
    fn test_eos_token_id() {
        let eos = EosTokenId::Single(2);
        assert!(eos.contains(2));
        assert!(!eos.contains(1));
        assert_eq!(eos.to_vec(), vec![2]);

        let eos = EosTokenId::Multiple(vec![128_001, 128_008, 128_009]);
        assert!(eos.contains(128_001));
        assert!(eos.contains(128_009));
        assert!(!eos.contains(2));
        assert_eq!(eos.to_vec(), vec![128_001, 128_008, 128_009]);
    }

    #[test]
    fn test_tokenizer_config_add_bos_true() {
        let json = r#"{"add_bos_token": true}"#;
        let config = TokenizerConfig::from_json(json).unwrap_or_else(|e| {
            panic!("Failed to parse tokenizer config: {e}");
        });
        assert!(config.add_bos_token);
    }

    #[test]
    fn test_tokenizer_config_add_bos_false() {
        let json = r#"{"add_bos_token": false, "add_prefix_space": false}"#;
        let config = TokenizerConfig::from_json(json).unwrap_or_else(|e| {
            panic!("Failed to parse tokenizer config: {e}");
        });
        assert!(!config.add_bos_token);
    }

    #[test]
    fn test_tokenizer_config_default() {
        let json = r"{}";
        let config = TokenizerConfig::from_json(json).unwrap_or_else(|e| {
            panic!("Failed to parse tokenizer config: {e}");
        });
        assert!(config.add_bos_token);
    }
}
