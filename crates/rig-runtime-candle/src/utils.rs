#![allow(unsafe_code)]

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};

use rig_core::types::{Activation, ActivationMetadata, SamplingParams, Shape, TensorData};

use crate::config::TokenizerConfig;
use crate::error::{CandleError, Result};

#[inline]
pub fn pod_vec_to_bytes<T: bytemuck::Pod>(mut v: Vec<T>) -> Vec<u8> {
    let element_size = std::mem::size_of::<T>();
    let len = v.len() * element_size;
    let cap = v.capacity() * element_size;
    let ptr = v.as_mut_ptr().cast::<u8>();
    std::mem::forget(v);
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

pub struct CachedSampler {
    pub processor: LogitsProcessor,
    pub params: SamplingParams,
}

pub fn to_candle_sampling(params: &SamplingParams) -> Sampling {
    if params.temperature <= 0.0 {
        Sampling::ArgMax
    } else if params.top_k > 0 && params.top_p < 1.0 {
        Sampling::TopKThenTopP {
            k: params.top_k,
            p: f64::from(params.top_p),
            temperature: f64::from(params.temperature),
        }
    } else if params.top_k > 0 {
        Sampling::TopK {
            k: params.top_k,
            temperature: f64::from(params.temperature),
        }
    } else if params.top_p < 1.0 {
        Sampling::TopP {
            p: f64::from(params.top_p),
            temperature: f64::from(params.temperature),
        }
    } else {
        Sampling::All {
            temperature: f64::from(params.temperature),
        }
    }
}

pub fn activation_to_tensor(
    activation: &Activation,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let shape: Vec<usize> = activation.shape.dims().to_vec();
    let bytes = activation.as_bytes();

    match dtype {
        DType::F32 => {
            let floats: &[f32] = bytemuck::cast_slice(bytes);
            Ok(Tensor::from_slice(floats, shape.as_slice(), device)?)
        }
        DType::F16 | DType::BF16 => Ok(Tensor::from_raw_buffer(bytes, dtype, &shape, device)?),
        _ => Err(CandleError::DTypeConversion(format!(
            "Unsupported dtype: {dtype:?}"
        ))),
    }
}

#[allow(clippy::needless_pass_by_value)]
pub fn tensor_to_activation(
    tensor: Tensor,
    dtype: DType,
    metadata: ActivationMetadata,
) -> Result<Activation> {
    let shape = Shape::from_slice(tensor.dims());
    let tensor = tensor.to_device(&Device::Cpu)?;
    let flat = tensor.flatten_all()?;

    let (bytes, rig_dtype) = match dtype {
        DType::F32 => {
            let floats: Vec<f32> = flat.to_vec1()?;
            (pod_vec_to_bytes(floats), rig_core::DType::F32)
        }
        DType::F16 => {
            let halfs: Vec<half::f16> = flat.to_vec1()?;
            (pod_vec_to_bytes(halfs), rig_core::DType::F16)
        }
        DType::BF16 => {
            let bhalfs: Vec<half::bf16> = flat.to_vec1()?;
            (pod_vec_to_bytes(bhalfs), rig_core::DType::BF16)
        }
        _ => {
            return Err(CandleError::DTypeConversion(format!(
                "Unsupported dtype: {dtype:?}"
            )));
        }
    };

    let data = TensorData::cpu(bytes, rig_dtype);
    Ok(Activation::new(data, shape, metadata))
}

pub fn extract_token_ids(activation: &Activation, device: &Device) -> Result<Tensor> {
    let bytes = activation.as_bytes();
    let dims = activation.shape.dims();

    let batch_size = dims.first().copied().ok_or_else(|| {
        CandleError::Candle(candle_core::Error::Msg(
            "Activation shape missing batch dimension".to_string(),
        ))
    })?;
    let seq_len = dims.get(1).copied().ok_or_else(|| {
        CandleError::Candle(candle_core::Error::Msg(
            "Activation shape missing sequence dimension".to_string(),
        ))
    })?;

    let tokens: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Tensor::new(tokens.as_slice(), device)?
        .reshape((batch_size, seq_len))
        .map_err(Into::into)
}

pub struct TokenizerConfigResult {
    pub add_bos_token: bool,
    pub chat_template: Option<String>,
    pub eos_token_str: String,
    pub bos_token_str: Option<String>,
}

pub fn load_tokenizer_config(model_path: &Path) -> Result<TokenizerConfigResult> {
    let jinja_path = model_path.join("chat_template.jinja");
    let template_from_file = if jinja_path.exists() {
        match std::fs::read_to_string(&jinja_path) {
            Ok(content) => Some(content),
            Err(e) => {
                tracing::warn!("Failed to read chat_template.jinja: {e}");
                None
            }
        }
    } else {
        None
    };

    let tokenizer_config_path = model_path.join("tokenizer_config.json");
    if !tokenizer_config_path.exists() {
        return Err(CandleError::TokenizerConfigNotFound(tokenizer_config_path));
    }

    let tokenizer_config = TokenizerConfig::from_file(&tokenizer_config_path)?;

    let add_bos_token = tokenizer_config.add_bos_token;

    let eos_token_str = tokenizer_config
        .eos_token
        .clone()
        .ok_or(CandleError::TokenizerConfigMissingField("eos_token"))?;

    let bos_token_str = tokenizer_config.bos_token.clone();

    let chat_template = template_from_file.or(tokenizer_config.chat_template);

    Ok(TokenizerConfigResult {
        add_bos_token,
        chat_template,
        eos_token_str,
        bos_token_str,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_candle_sampling_argmax() {
        let params = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 42,
        };
        assert!(matches!(to_candle_sampling(&params), Sampling::ArgMax));
    }

    #[test]
    fn test_to_candle_sampling_top_k() {
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 50,
            top_p: 1.0,
            seed: 42,
        };
        assert!(matches!(
            to_candle_sampling(&params),
            Sampling::TopK { k: 50, .. }
        ));
    }

    #[test]
    fn test_to_candle_sampling_top_p() {
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 0,
            top_p: 0.9,
            seed: 42,
        };
        assert!(matches!(to_candle_sampling(&params), Sampling::TopP { .. }));
    }

    #[test]
    fn test_to_candle_sampling_top_k_then_top_p() {
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.9,
            seed: 42,
        };
        assert!(matches!(
            to_candle_sampling(&params),
            Sampling::TopKThenTopP { k: 50, .. }
        ));
    }

    #[test]
    fn test_to_candle_sampling_all() {
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            seed: 42,
        };
        assert!(matches!(to_candle_sampling(&params), Sampling::All { .. }));
    }
}
