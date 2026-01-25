#![allow(unsafe_code)]

use std::cell::RefCell;
use std::fs::File;

use candle_core::quantized::{QMatMul, gguf_file};
use candle_core::{Device, Tensor};

use super::{QuantizedLinear, WeightLoader};
use crate::error::CandleError;

pub struct GgufLoader {
    content: gguf_file::Content,
    file: RefCell<File>,
    device: Device,
}

impl GgufLoader {
    #[must_use]
    pub fn new(content: gguf_file::Content, file: File, device: &Device) -> Self {
        Self {
            content,
            file: RefCell::new(file),
            device: device.clone(),
        }
    }

    #[must_use]
    pub const fn content(&self) -> &gguf_file::Content {
        &self.content
    }

    #[must_use]
    pub const fn device(&self) -> &Device {
        &self.device
    }

    pub fn load_qmatmul(&self, name: &str) -> Result<QMatMul, CandleError> {
        let mut file = self.file.borrow_mut();
        let qtensor = self
            .content
            .tensor(&mut *file, name, &self.device)
            .map_err(|_| CandleError::WeightNotFound(name.to_string()))?;
        QMatMul::from_qtensor(qtensor).map_err(Into::into)
    }

    pub fn load_dequantized(&self, name: &str) -> Result<Tensor, CandleError> {
        let mut file = self.file.borrow_mut();
        let qtensor = self
            .content
            .tensor(&mut *file, name, &self.device)
            .map_err(|_| CandleError::WeightNotFound(name.to_string()))?;
        qtensor.dequantize(&self.device).map_err(Into::into)
    }
}

impl WeightLoader for GgufLoader {
    type Weight = QuantizedLinear;

    fn load_linear(&self, name: &str, with_bias: bool) -> Result<QuantizedLinear, CandleError> {
        let weight = self.load_qmatmul(name)?;

        let bias = if with_bias {
            let bias_name = if name.ends_with(".weight") {
                name.replace(".weight", ".bias")
            } else {
                format!("{name}.bias")
            };

            if self.contains(&bias_name) {
                Some(self.load_dequantized(&bias_name)?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(QuantizedLinear::new(weight, bias))
    }

    fn load_tensor(&self, name: &str) -> Result<Tensor, CandleError> {
        self.load_dequantized(name)
    }

    fn contains(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }
}

unsafe impl Send for GgufLoader {}
unsafe impl Sync for GgufLoader {}
