#![allow(unsafe_code)]

use candle_core::quantized::QMatMul;
use candle_core::{Module, Result, Tensor};

use super::Weight;

pub struct QuantizedLinear {
    weight: QMatMul,
    bias: Option<Tensor>,
}

impl QuantizedLinear {
    #[must_use]
    pub fn new(weight: QMatMul, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    #[must_use]
    pub fn without_bias(weight: QMatMul) -> Self {
        Self { weight, bias: None }
    }

    #[must_use]
    pub fn qmatmul(&self) -> &QMatMul {
        &self.weight
    }

    #[must_use]
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Weight for QuantizedLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = Module::forward(&self.weight, input)?;
        match &self.bias {
            Some(b) => output.broadcast_add(b),
            None => Ok(output),
        }
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

unsafe impl Send for QuantizedLinear {}
unsafe impl Sync for QuantizedLinear {}
