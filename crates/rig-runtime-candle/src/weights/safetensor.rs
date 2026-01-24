use candle_core::Tensor;
use candle_nn::{Linear, VarBuilder};

use super::WeightLoader;
use crate::error::CandleError;

pub struct SafetensorLoader<'a> {
    vb: VarBuilder<'a>,
}

impl<'a> SafetensorLoader<'a> {
    #[must_use]
    pub fn new(vb: VarBuilder<'a>) -> Self {
        Self { vb }
    }

    #[must_use]
    pub fn pp(&self, prefix: &str) -> Self {
        Self {
            vb: self.vb.pp(prefix),
        }
    }

    #[must_use]
    pub fn var_builder(&self) -> &VarBuilder<'a> {
        &self.vb
    }
}

impl WeightLoader for SafetensorLoader<'_> {
    type Weight = Linear;

    fn load_linear(&self, name: &str, with_bias: bool) -> Result<Linear, CandleError> {
        let scoped = self.vb.pp(name);

        let weight_name = format!("{name}.weight");
        if !self.vb.contains_tensor(&weight_name) {
            return Err(CandleError::WeightNotFound(weight_name));
        }

        let weight = scoped
            .get_unchecked_dtype("weight", scoped.dtype())
            .map_err(|e| CandleError::WeightNotFound(format!("{name}.weight: {e}")))?;

        let bias = if with_bias {
            let bias_name = format!("{name}.bias");
            if !self.vb.contains_tensor(&bias_name) {
                return Err(CandleError::WeightNotFound(bias_name));
            }
            Some(
                scoped
                    .get_unchecked_dtype("bias", scoped.dtype())
                    .map_err(|e| CandleError::WeightNotFound(format!("{name}.bias: {e}")))?,
            )
        } else {
            None
        };

        Ok(Linear::new(weight, bias))
    }

    fn load_tensor(&self, name: &str) -> Result<Tensor, CandleError> {
        if !self.vb.contains_tensor(name) {
            return Err(CandleError::WeightNotFound(name.to_string()));
        }

        self.vb
            .get_unchecked_dtype(name, self.vb.dtype())
            .map_err(|e| CandleError::WeightNotFound(format!("{name}: {e}")))
    }

    fn contains(&self, name: &str) -> bool {
        self.vb.contains_tensor(name)
    }
}
