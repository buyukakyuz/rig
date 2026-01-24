mod llama;
mod mistral;
mod qwen;

pub use llama::LlamaArchitecture;
pub use mistral::MistralArchitecture;
pub use qwen::Qwen2Architecture;

use crate::error::{CandleError, Result};
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFn {
    Silu,
    SwiGLU,
    Gelu,
    GeluNew,
    Relu,
}

impl ActivationFn {
    pub fn apply(&self, tensor: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        match self {
            Self::Silu => candle_nn::ops::silu(tensor),
            Self::SwiGLU => candle_nn::ops::swiglu(tensor),
            Self::Gelu => tensor.gelu_erf(),
            Self::GeluNew => tensor.gelu(),
            Self::Relu => tensor.relu(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    None,
    Llama3 {
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_position_embeddings: usize,
    },
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionConfig {
    pub q_bias: bool,
    pub k_bias: bool,
    pub v_bias: bool,
    pub o_bias: bool,
}

impl AttentionConfig {
    pub const NO_BIAS: Self = Self {
        q_bias: false,
        k_bias: false,
        v_bias: false,
        o_bias: false,
    };

    pub const QKV_BIAS: Self = Self {
        q_bias: true,
        k_bias: true,
        v_bias: true,
        o_bias: false,
    };
}

#[derive(Debug, Clone)]
pub struct SafetensorWeightNames {
    pub embed_tokens: &'static str,
    pub layer_prefix: &'static str,
    pub q_proj: &'static str,
    pub k_proj: &'static str,
    pub v_proj: &'static str,
    pub o_proj: &'static str,
    pub input_layernorm: &'static str,
    pub post_attention_layernorm: &'static str,
    pub gate_proj: &'static str,
    pub up_proj: &'static str,
    pub down_proj: &'static str,
    pub final_norm: &'static str,
    pub lm_head: &'static str,
}

impl Default for SafetensorWeightNames {
    fn default() -> Self {
        Self {
            embed_tokens: "model.embed_tokens",
            layer_prefix: "model.layers",
            q_proj: "self_attn.q_proj",
            k_proj: "self_attn.k_proj",
            v_proj: "self_attn.v_proj",
            o_proj: "self_attn.o_proj",
            input_layernorm: "input_layernorm",
            post_attention_layernorm: "post_attention_layernorm",
            gate_proj: "mlp.gate_proj",
            up_proj: "mlp.up_proj",
            down_proj: "mlp.down_proj",
            final_norm: "model.norm",
            lm_head: "lm_head",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufWeightNames {
    pub embed_tokens: &'static str,
    pub layer_prefix: &'static str,
    pub q_proj: &'static str,
    pub k_proj: &'static str,
    pub v_proj: &'static str,
    pub o_proj: &'static str,
    pub input_layernorm: &'static str,
    pub post_attention_layernorm: &'static str,
    pub gate_proj: &'static str,
    pub up_proj: &'static str,
    pub down_proj: &'static str,
    pub final_norm: &'static str,
    pub lm_head: &'static str,
}

impl Default for GgufWeightNames {
    fn default() -> Self {
        Self {
            embed_tokens: "token_embd.weight",
            layer_prefix: "blk",
            q_proj: "attn_q.weight",
            k_proj: "attn_k.weight",
            v_proj: "attn_v.weight",
            o_proj: "attn_output.weight",
            input_layernorm: "attn_norm.weight",
            post_attention_layernorm: "ffn_norm.weight",
            gate_proj: "ffn_gate.weight",
            up_proj: "ffn_up.weight",
            down_proj: "ffn_down.weight",
            final_norm: "output_norm.weight",
            lm_head: "output.weight",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RopeScalingConfig {
    pub rope_type: Option<String>,
    pub factor: Option<f32>,
    pub low_freq_factor: Option<f32>,
    pub high_freq_factor: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
}

pub trait ModelArchitecture: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;
    fn activation(&self) -> ActivationFn;
    fn default_rms_norm_eps(&self) -> f64;
    fn default_rope_theta(&self) -> f64;
    fn attention_config(&self) -> AttentionConfig;
    fn default_tie_word_embeddings(&self) -> bool;
    fn safetensor_weight_names(&self) -> SafetensorWeightNames;
    fn gguf_weight_names(&self) -> GgufWeightNames;
    fn safetensor_layer_path(&self, layer_idx: usize, component: &str) -> String {
        let names = self.safetensor_weight_names();
        format!("{}.{layer_idx}.{component}", names.layer_prefix)
    }
    fn gguf_layer_path(&self, layer_idx: usize, component: &str) -> String {
        let names = self.gguf_weight_names();
        format!("{}.{layer_idx}.{component}", names.layer_prefix)
    }
    fn interpret_rope_scaling(&self, _config: &RopeScalingConfig) -> Option<RopeScaling> {
        None
    }
}

pub fn detect_architecture_gguf(
    content: &candle_core::quantized::gguf_file::Content,
) -> Result<Arc<dyn ModelArchitecture>> {
    let arch_name = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .ok_or_else(|| {
            CandleError::ArchitectureDetectionFailed(
                "GGUF file missing 'general.architecture' metadata".to_string(),
            )
        })?;

    get_architecture(arch_name)
}

pub fn detect_architecture_config(config_path: &Path) -> Result<Arc<dyn ModelArchitecture>> {
    let config_content = std::fs::read_to_string(config_path).map_err(|e| {
        CandleError::ArchitectureDetectionFailed(format!(
            "Failed to read config file {}: {}",
            config_path.display(),
            e
        ))
    })?;

    let config: serde_json::Value = serde_json::from_str(&config_content).map_err(|e| {
        CandleError::ArchitectureDetectionFailed(format!(
            "Failed to parse config file {}: {}",
            config_path.display(),
            e
        ))
    })?;

    let architectures = config.get("architectures").and_then(|v| v.as_array());

    if let Some(archs) = architectures {
        if let Some(first_arch) = archs.first().and_then(|v| v.as_str()) {
            return get_architecture(first_arch);
        }
    }

    if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
        return get_architecture(model_type);
    }

    Err(CandleError::ArchitectureDetectionFailed(
        "Config file missing 'architectures' or 'model_type' field".to_string(),
    ))
}

fn get_architecture(name: &str) -> Result<Arc<dyn ModelArchitecture>> {
    match name.to_lowercase().as_str() {
        "llama" | "llamaforcausallm" => Ok(Arc::new(LlamaArchitecture)),
        "qwen2" | "qwen2forcausallm" => Ok(Arc::new(Qwen2Architecture)),
        "mistral" | "mistralforcausallm" => Ok(Arc::new(MistralArchitecture)),
        _ => Err(CandleError::UnknownArchitecture(name.to_string())),
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::expect_used,
    clippy::float_cmp
)]
mod tests {
    use super::*;

    #[test]
    fn test_get_architecture_llama() {
        let arch = get_architecture("llama").expect("llama architecture should be supported");
        assert_eq!(arch.name(), "llama");
        assert_eq!(arch.activation(), ActivationFn::SwiGLU);
        assert_eq!(arch.attention_config(), AttentionConfig::NO_BIAS);
    }

    #[test]
    fn test_get_architecture_llama_causal() {
        let arch =
            get_architecture("LlamaForCausalLM").expect("LlamaForCausalLM should be supported");
        assert_eq!(arch.name(), "llama");
    }

    #[test]
    fn test_get_architecture_qwen2() {
        let arch = get_architecture("qwen2").expect("qwen2 architecture should be supported");
        assert_eq!(arch.name(), "qwen2");
        assert_eq!(arch.attention_config(), AttentionConfig::QKV_BIAS);
        assert!(arch.default_tie_word_embeddings());
    }

    #[test]
    fn test_get_architecture_qwen2_causal() {
        let arch =
            get_architecture("Qwen2ForCausalLM").expect("Qwen2ForCausalLM should be supported");
        assert_eq!(arch.name(), "qwen2");
    }

    #[test]
    fn test_get_architecture_mistral() {
        let arch = get_architecture("mistral").expect("mistral architecture should be supported");
        assert_eq!(arch.name(), "mistral");
        assert_eq!(arch.attention_config(), AttentionConfig::NO_BIAS);
    }

    #[test]
    fn test_get_architecture_mistral_causal() {
        let arch =
            get_architecture("MistralForCausalLM").expect("MistralForCausalLM should be supported");
        assert_eq!(arch.name(), "mistral");
    }

    #[test]
    fn test_get_architecture_unknown() {
        let result = get_architecture("unknown_model");
        assert!(result.is_err());
        assert!(
            matches!(result, Err(CandleError::UnknownArchitecture(ref name)) if name == "unknown_model"),
            "Expected UnknownArchitecture error with name 'unknown_model'"
        );
    }

    #[test]
    fn test_safetensor_layer_path() {
        let arch = get_architecture("llama").expect("llama should be supported");
        let path = arch.safetensor_layer_path(5, "self_attn.q_proj.weight");
        assert_eq!(path, "model.layers.5.self_attn.q_proj.weight");
    }

    #[test]
    fn test_gguf_layer_path() {
        let arch = get_architecture("llama").expect("llama should be supported");
        let path = arch.gguf_layer_path(5, "attn_q.weight");
        assert_eq!(path, "blk.5.attn_q.weight");
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_attention_config_constants() {
        assert!(!AttentionConfig::NO_BIAS.q_bias);
        assert!(!AttentionConfig::NO_BIAS.k_bias);
        assert!(!AttentionConfig::NO_BIAS.v_bias);
        assert!(!AttentionConfig::NO_BIAS.o_bias);

        assert!(AttentionConfig::QKV_BIAS.q_bias);
        assert!(AttentionConfig::QKV_BIAS.k_bias);
        assert!(AttentionConfig::QKV_BIAS.v_bias);
        assert!(!AttentionConfig::QKV_BIAS.o_bias);
    }

    #[test]
    #[ignore = "requires models/tiny-llama-q4/model.gguf"]
    fn test_detect_architecture_gguf_llama() {
        use candle_core::quantized::gguf_file;
        use std::fs::File;

        let model_path = std::path::Path::new("../../models/tiny-llama-q4/model.gguf");
        if !model_path.exists() {
            eprintln!(
                "Skipping test: model file not found at {}",
                model_path.display()
            );
            return;
        }

        let mut file = File::open(model_path).expect("Failed to open model file");
        let content = gguf_file::Content::read(&mut file).expect("Failed to read GGUF content");
        let arch = detect_architecture_gguf(&content).expect("Failed to detect architecture");
        assert_eq!(arch.name(), "llama");
    }

    #[test]
    #[ignore = "requires models/tiny-llama/config.json"]
    fn test_detect_architecture_config_llama() {
        let config_path = std::path::Path::new("../../models/tiny-llama/config.json");
        if !config_path.exists() {
            eprintln!(
                "Skipping test: config file not found at {}",
                config_path.display()
            );
            return;
        }

        let arch = detect_architecture_config(config_path).expect("Failed to detect architecture");
        assert_eq!(arch.name(), "llama");
    }

    #[test]
    #[ignore = "requires models/qwen-3b-q4/qwen2.5-3b-instruct-q4_k_m.gguf"]
    fn test_detect_architecture_gguf_qwen() {
        use candle_core::quantized::gguf_file;
        use std::fs::File;

        let model_path =
            std::path::Path::new("../../models/qwen-3b-q4/qwen2.5-3b-instruct-q4_k_m.gguf");
        if !model_path.exists() {
            eprintln!(
                "Skipping test: model file not found at {}",
                model_path.display()
            );
            return;
        }

        let mut file = File::open(model_path).expect("Failed to open model file");
        let content = gguf_file::Content::read(&mut file).expect("Failed to read GGUF content");
        let arch = detect_architecture_gguf(&content).expect("Failed to detect architecture");
        assert_eq!(arch.name(), "qwen2");
    }

    #[test]
    #[ignore = "requires models/qwen-3b-q4/config.json"]
    fn test_detect_architecture_config_qwen() {
        let config_path = std::path::Path::new("../../models/qwen-3b-q4/config.json");
        if !config_path.exists() {
            eprintln!(
                "Skipping test: config file not found at {}",
                config_path.display()
            );
            return;
        }

        let arch = detect_architecture_config(config_path).expect("Failed to detect architecture");
        assert_eq!(arch.name(), "qwen2");
    }
}
