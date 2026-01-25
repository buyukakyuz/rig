use super::{
    ActivationFn, AttentionConfig, GgufWeightNames, ModelArchitecture, RopeScaling,
    RopeScalingConfig, SafetensorWeightNames,
};

#[derive(Debug, Clone, Copy)]
pub struct MistralArchitecture;

#[allow(clippy::unused_self)]
impl ModelArchitecture for MistralArchitecture {
    fn name(&self) -> &'static str {
        "mistral"
    }

    fn activation(&self) -> ActivationFn {
        ActivationFn::Silu
    }

    fn default_rms_norm_eps(&self) -> f64 {
        1e-5
    }

    fn default_rope_theta(&self) -> f64 {
        10000.0
    }

    fn attention_config(&self) -> AttentionConfig {
        AttentionConfig::NO_BIAS
    }

    fn default_tie_word_embeddings(&self) -> bool {
        false
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
    fn test_mistral_architecture() {
        let arch = MistralArchitecture;
        assert_eq!(arch.name(), "mistral");
        assert_eq!(arch.activation(), ActivationFn::Silu);
        assert_eq!(arch.default_rms_norm_eps(), 1e-5);
        assert_eq!(arch.default_rope_theta(), 10000.0);
        assert_eq!(arch.attention_config(), AttentionConfig::NO_BIAS);
        assert!(!arch.default_tie_word_embeddings());
    }

    #[test]
    fn test_mistral_attention_config() {
        let arch = MistralArchitecture;
        let config = arch.attention_config();

        assert!(!config.q_bias);
        assert!(!config.k_bias);
        assert!(!config.v_bias);
        assert!(!config.o_bias);
    }

    #[test]
    fn test_mistral_safetensor_names() {
        let arch = MistralArchitecture;
        let names = arch.safetensor_weight_names();

        assert_eq!(names.embed_tokens, "model.embed_tokens");
        assert_eq!(names.layer_prefix, "model.layers");
        assert_eq!(names.q_proj, "self_attn.q_proj");
    }

    #[test]
    fn test_mistral_gguf_names() {
        let arch = MistralArchitecture;
        let names = arch.gguf_weight_names();

        assert_eq!(names.embed_tokens, "token_embd.weight");
        assert_eq!(names.layer_prefix, "blk");
        assert_eq!(names.q_proj, "attn_q.weight");
    }
}
