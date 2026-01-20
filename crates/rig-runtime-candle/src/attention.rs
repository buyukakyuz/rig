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
    num_queries_per_kv: usize,
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
            num_queries_per_kv: num_attention_heads / num_kv_heads,
            span,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        rope_cache: &RopeCache,
        kv_cache: Option<&mut LayerKvCache>,
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
        let v = v.transpose(1, 2)?;

        let (q, k) = self.apply_rotary_emb(&q, &k, index_pos, rope_cache)?;

        let (k, v) = match kv_cache {
            Some(cache) => cache.update(k, v)?,
            None => (k.contiguous()?, v.contiguous()?),
        };

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        let attn_weights = if seq_len > 1 {
            self.apply_causal_mask(&attn_weights)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;
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

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        if self.num_queries_per_kv == 1 {
            return Ok(x);
        }

        let (batch_size, num_kv_heads, seq_len, head_dim) = x.dims4()?;

        let x = x
            .unsqueeze(2)?
            .expand((
                batch_size,
                num_kv_heads,
                self.num_queries_per_kv,
                seq_len,
                head_dim,
            ))?
            .reshape((
                batch_size,
                num_kv_heads * self.num_queries_per_kv,
                seq_len,
                head_dim,
            ))?;

        Ok(x)
    }

    fn apply_causal_mask(&self, attn_weights: &Tensor) -> Result<Tensor> {
        let (_, _, seq_len, kv_len) = attn_weights.dims4()?;
        let device = attn_weights.device();
        let dtype = attn_weights.dtype();

        let mask = Self::create_causal_mask(seq_len, kv_len, device)?;
        let mask = mask.broadcast_as(attn_weights.shape())?;

        let on_true = Tensor::new(f32::NEG_INFINITY, device)?
            .to_dtype(dtype)?
            .broadcast_as(attn_weights.shape())?;

        mask.where_cond(&on_true, attn_weights)
    }

    fn create_causal_mask(
        seq_len: usize,
        kv_len: usize,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let offset = kv_len.saturating_sub(seq_len);

        let mask: Vec<u8> = (0..seq_len)
            .flat_map(|i| (0..kv_len).map(move |j| u8::from(j > i + offset)))
            .collect();

        Tensor::from_slice(&mask, (seq_len, kv_len), device)
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask() {
        let device = candle_core::Device::Cpu;

        let mask = CausalSelfAttention::create_causal_mask(4, 4, &device)
            .unwrap_or_else(|e| panic!("create mask failed: {e}"));

        let expected = vec![0u8, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0];
        let mask_vec: Vec<u8> = mask
            .flatten_all()
            .unwrap_or_else(|e| panic!("flatten failed: {e}"))
            .to_vec1()
            .unwrap_or_else(|e| panic!("to_vec1 failed: {e}"));
        assert_eq!(mask_vec, expected);
    }

    #[test]
    fn test_causal_mask_incremental() {
        let device = candle_core::Device::Cpu;

        let mask = CausalSelfAttention::create_causal_mask(1, 4, &device)
            .unwrap_or_else(|e| panic!("create mask failed: {e}"));

        let expected = vec![0u8, 0, 0, 0];
        let mask_vec: Vec<u8> = mask
            .flatten_all()
            .unwrap_or_else(|e| panic!("flatten failed: {e}"))
            .to_vec1()
            .unwrap_or_else(|e| panic!("to_vec1 failed: {e}"));
        assert_eq!(mask_vec, expected);
    }
}
