use candle_core::{Module, Result, Tensor};
use candle_nn::RmsNorm;

use crate::cache::{LayerKvCache, RopeCache};
use crate::layers::{Attention, Mlp};
use crate::weights::Weight;

pub struct TransformerBlock<W: Weight> {
    input_norm: RmsNorm,
    attn: Attention<W>,
    post_attention_norm: RmsNorm,
    mlp: Mlp<W>,
    span: tracing::Span,
}

impl<W: Weight> TransformerBlock<W> {
    #[must_use]
    pub fn new(
        input_norm: RmsNorm,
        attn: Attention<W>,
        post_attention_norm: RmsNorm,
        mlp: Mlp<W>,
    ) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        Self {
            input_norm,
            attn,
            post_attention_norm,
            mlp,
            span,
        }
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
        let x = self.input_norm.forward(x)?;
        let x = self
            .attn
            .forward(&x, index_pos, rope_cache, kv_cache, max_seq_len)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attention_norm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }
}
