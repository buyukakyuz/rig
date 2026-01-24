mod gguf;
mod linear;
mod quantized;
mod safetensor;

pub use gguf::GgufLoader;
pub use quantized::QuantizedLinear;
pub use safetensor::SafetensorLoader;

use candle_core::{Result, Tensor};

use crate::error::CandleError;

pub trait Weight: Send + Sync {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn has_bias(&self) -> bool;
}

pub trait WeightLoader: Send + Sync {
    type Weight: Weight;
    fn load_linear(
        &self,
        name: &str,
        with_bias: bool,
    ) -> std::result::Result<Self::Weight, CandleError>;
    fn load_tensor(&self, name: &str) -> std::result::Result<Tensor, CandleError>;
    fn contains(&self, name: &str) -> bool;
}
