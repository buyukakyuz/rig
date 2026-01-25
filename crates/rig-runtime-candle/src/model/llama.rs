use super::{
    ActivationFn, AttentionConfig, GgufWeightNames, ModelArchitecture, RopeScaling,
    RopeScalingConfig, SafetensorWeightNames,
};

#[derive(Debug, Clone, Copy)]
pub struct LlamaArchitecture;

#[allow(clippy::unused_self)]
impl ModelArchitecture for LlamaArchitecture {
    fn name(&self) -> &'static str {
        "llama"
    }

    fn activation(&self) -> ActivationFn {
        ActivationFn::SwiGLU
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

    fn interpret_rope_scaling(&self, config: &RopeScalingConfig) -> Option<RopeScaling> {
        match config.rope_type.as_deref() {
            Some("llama3") => {
                let factor = config.factor.unwrap_or(8.0);
                let low_freq_factor = config.low_freq_factor.unwrap_or(1.0);
                let high_freq_factor = config.high_freq_factor.unwrap_or(4.0);
                let original_max_position_embeddings =
                    config.original_max_position_embeddings.unwrap_or(8192);

                Some(RopeScaling::Llama3 {
                    factor,
                    low_freq_factor,
                    high_freq_factor,
                    original_max_position_embeddings,
                })
            }
            _ => None,
        }
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
    fn test_llama_architecture() {
        let arch = LlamaArchitecture;
        assert_eq!(arch.name(), "llama");
        assert_eq!(arch.activation(), ActivationFn::SwiGLU);
        assert_eq!(arch.default_rms_norm_eps(), 1e-5);
        assert_eq!(arch.default_rope_theta(), 10000.0);
        assert_eq!(arch.attention_config(), AttentionConfig::NO_BIAS);
        assert!(!arch.default_tie_word_embeddings());
    }

    #[test]
    fn test_llama_safetensor_names() {
        let arch = LlamaArchitecture;
        let names = arch.safetensor_weight_names();
        assert_eq!(names.embed_tokens, "model.embed_tokens");
        assert_eq!(names.layer_prefix, "model.layers");
        assert_eq!(names.q_proj, "self_attn.q_proj");
        assert_eq!(names.final_norm, "model.norm");
        assert_eq!(names.lm_head, "lm_head");
    }

    #[test]
    fn test_llama_gguf_names() {
        let arch = LlamaArchitecture;
        let names = arch.gguf_weight_names();
        assert_eq!(names.embed_tokens, "token_embd.weight");
        assert_eq!(names.layer_prefix, "blk");
        assert_eq!(names.q_proj, "attn_q.weight");
        assert_eq!(names.final_norm, "output_norm.weight");
        assert_eq!(names.lm_head, "output.weight");
    }

    #[test]
    fn test_llama3_rope_scaling() {
        let arch = LlamaArchitecture;

        let config = RopeScalingConfig {
            rope_type: Some("llama3".to_string()),
            factor: Some(8.0),
            low_freq_factor: Some(1.0),
            high_freq_factor: Some(4.0),
            original_max_position_embeddings: Some(8192),
        };

        let scaling = arch
            .interpret_rope_scaling(&config)
            .expect("Llama3 scaling should be recognized");
        match scaling {
            RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            } => {
                assert_eq!(factor, 8.0);
                assert_eq!(low_freq_factor, 1.0);
                assert_eq!(high_freq_factor, 4.0);
                assert_eq!(original_max_position_embeddings, 8192);
            }
            RopeScaling::None => panic!("Expected Llama3 scaling, got None"),
        }
    }

    #[test]
    fn test_llama_no_rope_scaling() {
        let arch = LlamaArchitecture;

        let config = RopeScalingConfig {
            rope_type: None,
            factor: None,
            low_freq_factor: None,
            high_freq_factor: None,
            original_max_position_embeddings: None,
        };

        let scaling = arch.interpret_rope_scaling(&config);
        assert!(scaling.is_none());
    }
}
