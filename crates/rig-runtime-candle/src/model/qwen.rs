use super::{
    ActivationFn, AttentionConfig, GgufWeightNames, ModelArchitecture, RopeScaling,
    RopeScalingConfig, SafetensorWeightNames,
};

#[derive(Debug, Clone, Copy)]
pub struct Qwen2Architecture;

#[allow(clippy::unused_self)]
impl ModelArchitecture for Qwen2Architecture {
    fn name(&self) -> &'static str {
        "qwen2"
    }

    fn activation(&self) -> ActivationFn {
        ActivationFn::Silu
    }

    fn default_rms_norm_eps(&self) -> f64 {
        1e-6
    }

    fn default_rope_theta(&self) -> f64 {
        1_000_000.0
    }

    fn attention_config(&self) -> AttentionConfig {
        AttentionConfig::QKV_BIAS
    }

    fn default_tie_word_embeddings(&self) -> bool {
        true
    }

    fn safetensor_weight_names(&self) -> SafetensorWeightNames {
        SafetensorWeightNames::default()
    }

    fn gguf_weight_names(&self) -> GgufWeightNames {
        GgufWeightNames::default()
    }

    fn interpret_rope_scaling(&self, _config: &RopeScalingConfig) -> Option<RopeScaling> {
        None
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
    fn test_qwen2_architecture() {
        let arch = Qwen2Architecture;
        assert_eq!(arch.name(), "qwen2");
        assert_eq!(arch.activation(), ActivationFn::Silu);
        assert_eq!(arch.default_rms_norm_eps(), 1e-6);
        assert_eq!(arch.default_rope_theta(), 1_000_000.0);
        assert_eq!(arch.attention_config(), AttentionConfig::QKV_BIAS);
        assert!(arch.default_tie_word_embeddings());
    }

    #[test]
    fn test_qwen2_attention_config() {
        let arch = Qwen2Architecture;
        let config = arch.attention_config();

        assert!(config.q_bias);
        assert!(config.k_bias);
        assert!(config.v_bias);
        assert!(!config.o_bias);
    }

    #[test]
    fn test_qwen2_safetensor_names() {
        let arch = Qwen2Architecture;
        let names = arch.safetensor_weight_names();

        assert_eq!(names.embed_tokens, "model.embed_tokens");
        assert_eq!(names.layer_prefix, "model.layers");
        assert_eq!(names.q_proj, "self_attn.q_proj");
    }

    #[test]
    fn test_qwen2_gguf_names() {
        let arch = Qwen2Architecture;
        let names = arch.gguf_weight_names();

        assert_eq!(names.embed_tokens, "token_embd.weight");
        assert_eq!(names.layer_prefix, "blk");
        assert_eq!(names.q_proj, "attn_q.weight");
    }

    #[test]
    fn test_qwen2_no_rope_scaling() {
        let arch = Qwen2Architecture;

        let config = RopeScalingConfig {
            rope_type: Some("linear".to_string()),
            factor: Some(2.0),
            low_freq_factor: None,
            high_freq_factor: None,
            original_max_position_embeddings: None,
        };

        let scaling = arch.interpret_rope_scaling(&config);
        assert!(scaling.is_none());
    }
}
