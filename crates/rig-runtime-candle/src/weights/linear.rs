use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module};

use super::Weight;

impl Weight for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Module::forward(self, input)
    }

    fn has_bias(&self) -> bool {
        self.bias().is_some()
    }
}
