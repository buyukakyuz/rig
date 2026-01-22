use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear, linear_no_bias};

use crate::cache::{LayerKvCache, RopeCache};
use crate::config::TransformerConfig;

#[derive(Debug, Clone)]
pub struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    span: tracing::Span,
}

impl CausalSelfAttention {
    pub fn load(
        vb: VarBuilder,
        config: &TransformerConfig,
        use_attention_bias: bool,
    ) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_attention_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads();

        let q_dim = num_attention_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let (q_proj, k_proj, v_proj) = if use_attention_bias {
            (
                linear(config.hidden_size, q_dim, vb.pp("q_proj"))?,
                linear(config.hidden_size, kv_dim, vb.pp("k_proj"))?,
                linear(config.hidden_size, kv_dim, vb.pp("v_proj"))?,
            )
        } else {
            (
                linear_no_bias(config.hidden_size, q_dim, vb.pp("q_proj"))?,
                linear_no_bias(config.hidden_size, kv_dim, vb.pp("k_proj"))?,
                linear_no_bias(config.hidden_size, kv_dim, vb.pp("v_proj"))?,
            )
        };
        let o_proj = linear_no_bias(q_dim, config.hidden_size, vb.pp("o_proj"))?;

        let span = tracing::span!(tracing::Level::TRACE, "attn");

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads,
            num_kv_heads,
            head_dim,
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
        let (batch_size, seq_len, _hidden_size) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((batch_size, seq_len, self.num_attention_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let (q, k) = self.apply_rotary_emb(&q, &k, index_pos, rope_cache)?;

        let (k, v) = match kv_cache {
            Some(cache) => {
                if !cache.is_initialized() {
                    cache.init_buffers(
                        batch_size,
                        self.num_kv_heads,
                        max_seq_len,
                        self.head_dim,
                        k.dtype(),
                        k.device(),
                    )?;
                }
                cache.update(k, v)?
            }
            None => (k, v),
        };

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let do_causal = seq_len > 1;
        let attn_output = candle_nn::ops::sdpa(&q, &k, &v, None, do_causal, scale, 1.0)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_attention_heads * self.head_dim,
        ))?;

        self.o_proj.forward(&attn_output)
    }

    fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
        rope_cache: &RopeCache,
    ) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let (cos, sin) = rope_cache.get(seq_len, index_pos)?;

        let q = candle_nn::rotary_emb::rope(q, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(k, &cos, &sin)?;

        Ok((q, k))
    }
}
