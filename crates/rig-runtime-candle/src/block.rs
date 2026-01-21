use candle_core::{Module, Result, Tensor};
use candle_nn::{RmsNorm, VarBuilder, rms_norm};

use crate::attention::CausalSelfAttention;
use crate::cache::{LayerKvCache, RopeCache};
use crate::config::TransformerConfig;
use crate::mlp::Mlp;

#[derive(Debug, Clone)]
pub struct TransformerBlock {
    input_layernorm: RmsNorm,
    self_attn: CausalSelfAttention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl TransformerBlock {
    pub fn load(
        vb: VarBuilder,
        config: &TransformerConfig,
        use_attention_bias: bool,
    ) -> Result<Self> {
        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;

        let self_attn = CausalSelfAttention::load(vb.pp("self_attn"), config, use_attention_bias)?;

        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        let mlp = Mlp::load(vb.pp("mlp"), config)?;

        let span = tracing::span!(tracing::Level::TRACE, "block");

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
            span,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        rope_cache: &RopeCache,
        kv_cache: Option<&mut LayerKvCache>,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self
            .self_attn
            .forward(&x, index_pos, rope_cache, kv_cache, max_seq_len)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }
}

#[cfg(all(test, feature = "metal"))]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use crate::config::Activation;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_block_shapes() {
        let device = Device::new_metal(0).expect("Metal device required for SDPA");
        let dtype = DType::F32;

        let config = TransformerConfig {
            hidden_size: 256,
            intermediate_size: 512,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: Some(4),
            vocab_size: 1000,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 512,
            tie_word_embeddings: false,
            hidden_act: Activation::Silu,
            bos_token_id: None,
            eos_token_id: None,
        };

        let rope_cache = RopeCache::new(&config, dtype, &device)
            .unwrap_or_else(|e| panic!("Failed to create RoPE cache: {e}"));

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let block = TransformerBlock::load(vb.pp("block"), &config, false)
            .unwrap_or_else(|e| panic!("Failed to load block: {e}"));

        let input = Tensor::randn(0f32, 1f32, (2, 8, 256), &device)
            .unwrap_or_else(|e| panic!("Failed to create input: {e}"));

        let output = block
            .forward(&input, 0, &rope_cache, None, config.max_position_embeddings)
            .unwrap_or_else(|e| panic!("Forward failed: {e}"));

        assert_eq!(output.dims(), &[2, 8, 256]);
    }
}
