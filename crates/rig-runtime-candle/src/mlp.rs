use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use crate::config::{Activation, TransformerConfig};

#[derive(Debug, Clone)]
pub struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: Activation,
    span: tracing::Span,
}

impl Mlp {
    pub fn load(vb: VarBuilder, config: &TransformerConfig) -> Result<Self> {
        let gate_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;
        let down_proj = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;

        let span = tracing::span!(tracing::Level::TRACE, "mlp");

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            activation: config.hidden_act,
            span,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;

        let gate = self.apply_activation(&gate)?;

        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden)
    }

    fn apply_activation(&self, x: &Tensor) -> Result<Tensor> {
        match self.activation {
            Activation::Silu => candle_nn::ops::silu(x),
            Activation::Gelu => x.gelu_erf(),
            Activation::GeluNew => x.gelu(),
            Activation::Relu => x.relu(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_mlp_shapes() {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let config = TransformerConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: None,
            vocab_size: 1000,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 512,
            tie_word_embeddings: false,
            hidden_act: Activation::Silu,
            bos_token_id: None,
            eos_token_id: None,
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let mlp =
            Mlp::load(vb.pp("mlp"), &config).unwrap_or_else(|e| panic!("Failed to load MLP: {e}"));

        let input = Tensor::randn(0f32, 1f32, (2, 8, 64), &device)
            .unwrap_or_else(|e| panic!("Failed to create input: {e}"));

        let output = mlp
            .forward(&input)
            .unwrap_or_else(|e| panic!("Forward failed: {e}"));

        assert_eq!(output.dims(), &[2, 8, 64]);
    }
}
