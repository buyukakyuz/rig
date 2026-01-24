use candle_core::{Result, Tensor};

use crate::model::ActivationFn;
use crate::weights::Weight;

pub struct Mlp<W: Weight> {
    gate_proj: W,
    up_proj: W,
    down_proj: W,
    activation: ActivationFn,
    span: tracing::Span,
}

impl<W: Weight> Mlp<W> {
    #[must_use]
    pub fn new(gate_proj: W, up_proj: W, down_proj: W, activation: ActivationFn) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Self {
            gate_proj,
            up_proj,
            down_proj,
            activation,
            span,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;

        let gate = self.activation.apply(&gate)?;

        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden)
    }
}
