#![allow(unsafe_code)]
#![allow(clippy::future_not_send)]

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm,
};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use tokenizers::Tokenizer as HfTokenizer;

use rig_core::error::PartitionError;
use rig_core::types::{
    Activation, ActivationMetadata, MemoryUsage, PartitionSpec, SampleResult, SamplingParams,
    Shape, TensorData,
};

use crate::block::TransformerBlock;
use crate::cache::RopeCache;
use crate::config::{TokenizerConfig, TransformerConfig};
use crate::error::{CandleError, Result};
use crate::kv_cache::CandleKvCache;

pub struct CandlePartition {
    spec: PartitionSpec,
    config: TransformerConfig,
    embed_tokens: Option<Embedding>,
    blocks: Vec<TransformerBlock>,
    norm: Option<RmsNorm>,
    lm_head: Option<Linear>,
    rope_cache: RopeCache,
    kv_cache: Mutex<CandleKvCache>,
    device: Device,
    dtype: DType,
    memory_usage: MemoryUsage,
    tokenizer: HfTokenizer,
    chat_template: Option<String>,
    eos_token_str: String,
    add_bos_token: bool,
}

impl CandlePartition {
    pub fn load(
        model_path: impl AsRef<Path>,
        spec: &PartitionSpec,
        _total_layers: usize,
        device: &Device,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();

        if !model_path.exists() {
            return Err(CandleError::ModelNotFound(model_path.to_path_buf()));
        }

        let config_path = model_path.join("config.json");
        let config = TransformerConfig::from_file(&config_path)?;

        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = HfTokenizer::from_file(&tokenizer_path)
            .map_err(|e| CandleError::TokenizerLoad(e.to_string()))?;

        let tokenizer_config_path = model_path.join("tokenizer_config.json");
        let add_bos_token = if tokenizer_config_path.exists() {
            TokenizerConfig::from_file(&tokenizer_config_path)
                .map(|c| c.add_bos_token)
                .unwrap_or(true)
        } else {
            true
        };
        let (chat_template, eos_token_str) = Self::load_chat_template(model_path)?;
        let dtype = Self::convert_dtype(spec.dtype)?;
        let safetensor_files = Self::find_safetensor_files(model_path)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)? };

        let use_attention_bias = vb
            .pp("model.layers.0.self_attn.q_proj")
            .contains_tensor("bias");

        tracing::info!(
            use_attention_bias = use_attention_bias,
            "Detected model architecture from weights"
        );

        let is_first = spec.layer_range.start == 0;
        let is_last = spec.layer_range.end == config.num_hidden_layers;

        let embed_tokens = if is_first {
            Some(embedding(
                config.vocab_size,
                config.hidden_size,
                vb.pp("model.embed_tokens"),
            )?)
        } else {
            None
        };

        let mut blocks = Vec::with_capacity(spec.layer_range.len());
        for layer_idx in spec.layer_range.clone() {
            let block = TransformerBlock::load(
                vb.pp(format!("model.layers.{layer_idx}")),
                &config,
                use_attention_bias,
            )?;
            blocks.push(block);
        }

        let (norm, lm_head) = if is_last {
            let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;

            let lm_head = if config.tie_word_embeddings {
                if let Some(ref embed) = embed_tokens {
                    Linear::new(embed.embeddings().clone(), None)
                } else {
                    let embed_weight = vb
                        .pp("model.embed_tokens")
                        .get((config.vocab_size, config.hidden_size), "weight")?;
                    Linear::new(embed_weight, None)
                }
            } else {
                linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
            };

            (Some(norm), Some(lm_head))
        } else {
            (None, None)
        };

        let rope_cache = RopeCache::new(&config, dtype, device)?;

        let dtype_size = match dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            _ => 2,
        };
        let memory_per_token = 2 * config.num_kv_heads() * config.head_dim() * dtype_size;
        let kv_cache = Mutex::new(CandleKvCache::new(blocks.len(), 1, memory_per_token));

        let memory_usage = Self::estimate_memory(&config, spec, is_first, is_last);

        Ok(Self {
            spec: spec.clone(),
            config,
            embed_tokens,
            blocks,
            norm,
            lm_head,
            rope_cache,
            kv_cache,
            device: device.clone(),
            dtype,
            memory_usage,
            tokenizer,
            chat_template,
            eos_token_str,
            add_bos_token,
        })
    }

    fn find_safetensor_files(model_path: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        let single_file = model_path.join("model.safetensors");
        if single_file.exists() {
            files.push(single_file);
            return Ok(files);
        }

        for entry in std::fs::read_dir(model_path)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(ext) = path.extension()
                && ext == "safetensors"
            {
                files.push(path);
            }
        }

        if files.is_empty() {
            return Err(CandleError::NoSafetensorFiles(model_path.to_path_buf()));
        }

        files.sort();
        Ok(files)
    }

    fn convert_dtype(dtype: rig_core::DType) -> Result<DType> {
        match dtype {
            rig_core::DType::F32 => Ok(DType::F32),
            rig_core::DType::F16 => Ok(DType::F16),
            rig_core::DType::BF16 => Ok(DType::BF16),
            other => Err(CandleError::DTypeConversion(format!(
                "Unsupported dtype: {other:?}"
            ))),
        }
    }

    fn load_chat_template(model_path: &Path) -> Result<(Option<String>, String)> {
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
        let (template_from_config, eos_token_str) = if tokenizer_config_path.exists() {
            match std::fs::read_to_string(&tokenizer_config_path) {
                Ok(config_str) => match serde_json::from_str::<serde_json::Value>(&config_str) {
                    Ok(config_json) => {
                        let template = config_json
                            .get("chat_template")
                            .and_then(serde_json::Value::as_str)
                            .map(String::from);
                        let eos = config_json
                            .get("eos_token")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("</s>")
                            .to_string();
                        (template, eos)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse tokenizer_config.json: {e}");
                        (None, "</s>".to_string())
                    }
                },
                Err(e) => {
                    tracing::warn!("Failed to read tokenizer_config.json: {e}");
                    (None, "</s>".to_string())
                }
            }
        } else {
            (None, "</s>".to_string())
        };

        let chat_template = template_from_file.or(template_from_config);

        Ok((chat_template, eos_token_str))
    }

    fn estimate_memory(
        config: &TransformerConfig,
        spec: &PartitionSpec,
        is_first: bool,
        is_last: bool,
    ) -> MemoryUsage {
        let dtype_size = match spec.dtype {
            rig_core::DType::F32 => 4,
            rig_core::DType::F16 | rig_core::DType::BF16 => 2,
            _ => 4,
        };

        let qkvo_size = 4 * config.hidden_size * config.hidden_size;
        let mlp_size = 3 * config.hidden_size * config.intermediate_size;
        let norm_size = 2 * config.hidden_size;
        let layer_weights = (qkvo_size + mlp_size + norm_size) * dtype_size;

        let num_layers = spec.layer_range.len();
        let mut weights_bytes = (layer_weights * num_layers) as u64;

        if is_first {
            weights_bytes += (config.vocab_size * config.hidden_size * dtype_size) as u64;
        }

        if is_last && !config.tie_word_embeddings {
            weights_bytes += (config.vocab_size * config.hidden_size * dtype_size) as u64;
        }

        MemoryUsage {
            weights_bytes,
            cache_bytes: 0,
            scratch_bytes: 0,
        }
    }

    fn activation_to_tensor(&self, activation: &Activation) -> Result<Tensor> {
        let shape: Vec<usize> = activation.shape.dims().to_vec();
        let bytes = activation.as_bytes();

        match self.dtype {
            DType::F32 => {
                let floats: &[f32] = bytemuck::cast_slice(bytes);
                Ok(Tensor::from_slice(floats, shape.as_slice(), &self.device)?)
            }
            DType::F16 | DType::BF16 => Ok(Tensor::from_raw_buffer(
                bytes,
                self.dtype,
                &shape,
                &self.device,
            )?),
            _ => Err(CandleError::DTypeConversion(format!(
                "Unsupported dtype: {:?}",
                self.dtype
            ))),
        }
    }

    fn tensor_to_activation(
        &self,
        tensor: Tensor,
        metadata: ActivationMetadata,
    ) -> Result<Activation> {
        let shape = Shape::from_slice(tensor.dims());
        let tensor = tensor.to_device(&Device::Cpu)?;
        let flat = tensor.flatten_all()?;

        let (bytes, dtype) = match self.dtype {
            DType::F32 => {
                let floats: Vec<f32> = flat.to_vec1()?;
                let bytes: Vec<u8> = bytemuck::cast_slice(&floats).to_vec();
                (bytes, rig_core::DType::F32)
            }
            DType::F16 => {
                let halfs: Vec<half::f16> = flat.to_vec1()?;
                let bytes: Vec<u8> = bytemuck::cast_slice(&halfs).to_vec();
                (bytes, rig_core::DType::F16)
            }
            DType::BF16 => {
                let bhalfs: Vec<half::bf16> = flat.to_vec1()?;
                let bytes: Vec<u8> = bytemuck::cast_slice(&bhalfs).to_vec();
                (bytes, rig_core::DType::BF16)
            }
            _ => {
                return Err(CandleError::DTypeConversion(format!(
                    "Unsupported dtype: {:?}",
                    self.dtype
                )));
            }
        };

        let data = TensorData::cpu(bytes, dtype);
        Ok(Activation::new(data, shape, metadata))
    }

    fn forward_impl(&self, input: Tensor) -> Result<Tensor> {
        let mut x = input;

        let mut kv_cache = self.kv_cache.lock().map_err(|e| {
            CandleError::Candle(candle_core::Error::Msg(format!(
                "KV cache lock failed: {e}"
            )))
        })?;

        let tensor_cache = kv_cache.tensor_cache_mut();
        let index_pos = tensor_cache.seq_len();

        tracing::debug!(
            input_shape = ?x.dims(),
            kv_cache_seq_len = index_pos,
            "forward_impl starting"
        );

        for (idx, block) in self.blocks.iter().enumerate() {
            let layer_cache = tensor_cache.layer_mut(idx);
            x = block.forward(
                &x,
                index_pos,
                &self.rope_cache,
                layer_cache,
                self.config.max_position_embeddings,
            )?;
        }

        if let Some(ref norm) = self.norm {
            x = norm.forward(&x)?;
        }

        if let Some(ref lm_head) = self.lm_head {
            let (batch_size, seq_len, _hidden_size) = x.dims3()?;
            let x_narrow = x.narrow(1, seq_len - 1, 1)?;
            let x_reshaped = x_narrow.reshape((batch_size, self.config.hidden_size))?;
            let logits = lm_head.forward(&x_reshaped)?;

            if let Ok(logits_f32) = logits.to_dtype(candle_core::DType::F32)
                && let Ok(flat) = logits_f32.flatten_all()
                && let Ok(vals) = flat.to_vec1::<f32>()
            {
                let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let min_val = vals.iter().copied().fold(f32::INFINITY, f32::min);
                let argmax = vals
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(i, _)| i);
                tracing::debug!(
                    logits_shape = ?logits.dims(),
                    min = min_val,
                    max = max_val,
                    argmax = argmax,
                    "lm_head output"
                );
            }

            return Ok(logits.reshape((batch_size, 1, self.config.vocab_size))?);
        }

        Ok(x)
    }

    #[must_use]
    pub fn has_embeddings(&self) -> bool {
        self.embed_tokens.is_some()
    }

    #[must_use]
    pub fn has_lm_head(&self) -> bool {
        self.lm_head.is_some()
    }

    #[must_use]
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn clear_kv_cache(&self) {
        if let Ok(mut cache) = self.kv_cache.lock() {
            cache.clear();
        }
    }

    pub fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        match &self.embed_tokens {
            Some(embed) => Ok(embed.forward(tokens)?),
            None => Err(CandleError::WeightNotFound(
                "embed_tokens not available (not first partition)".to_string(),
            )),
        }
    }

    fn extract_token_ids(&self, activation: &Activation) -> Result<Tensor> {
        let bytes = activation.as_bytes();
        let dims = activation.shape.dims();

        let batch_size = dims.first().copied().unwrap_or(1);
        let seq_len = dims.get(1).copied().unwrap_or(bytes.len() / 4);

        let tokens: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Tensor::new(tokens.as_slice(), &self.device)?
            .reshape((batch_size, seq_len))
            .map_err(Into::into)
    }

    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    #[must_use]
    pub fn bos_token(&self) -> u32 {
        self.config.bos_token_id.unwrap_or(1)
    }

    #[must_use]
    pub fn eos_token(&self) -> u32 {
        self.config
            .eos_token_id
            .as_ref()
            .map_or(2, |eos| eos.to_vec()[0])
    }

    pub fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| CandleError::TokenizationFailed(e.to_string()))?;

        let mut ids: Vec<u32> = encoding.get_ids().to_vec();

        if add_bos && self.add_bos_token {
            ids.insert(0, self.bos_token());
        }
        Ok(ids)
    }

    pub fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| CandleError::TokenizationFailed(e.to_string()))
    }

    pub fn tokenize_batch(&self, texts: &[&str], add_bos: bool) -> Result<Vec<Vec<u32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| CandleError::TokenizationFailed(e.to_string()))?;

        let mut results = Vec::with_capacity(encodings.len());
        for encoding in encodings {
            let mut ids: Vec<u32> = encoding.get_ids().to_vec();
            if add_bos && self.add_bos_token {
                ids.insert(0, self.bos_token());
            }
            results.push(ids);
        }
        Ok(results)
    }

    pub fn detokenize_batch(&self, token_sequences: &[&[u32]]) -> Result<Vec<String>> {
        self.tokenizer
            .decode_batch(token_sequences, true)
            .map_err(|e| CandleError::TokenizationFailed(e.to_string()))
    }

    pub fn extract_tokenizer(&self) -> crate::tokenizer::CandleTokenizer {
        crate::tokenizer::CandleTokenizer::new(
            self.tokenizer.clone(),
            self.chat_template.clone(),
            self.eos_token_str.clone(),
            self.add_bos_token,
            self.bos_token(),
            self.eos_token(),
        )
    }

    fn to_candle_sampling(params: &SamplingParams) -> Sampling {
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

    fn forward_sample_impl(
        &self,
        input: Activation,
        sampling: &SamplingParams,
    ) -> Result<Option<SampleResult>> {
        if self.lm_head.is_none() {
            return Ok(None);
        }

        let tensor = if self.has_embeddings() && input.dtype() == rig_core::DType::I8 {
            let token_ids = self.extract_token_ids(&input)?;
            self.embed(&token_ids)?
        } else {
            self.activation_to_tensor(&input)?
        };

        let mut x = tensor;

        let mut kv_cache = self.kv_cache.lock().map_err(|e| {
            CandleError::Candle(candle_core::Error::Msg(format!(
                "KV cache lock failed: {e}"
            )))
        })?;

        let tensor_cache = kv_cache.tensor_cache_mut();
        let index_pos = tensor_cache.seq_len();

        for (idx, block) in self.blocks.iter().enumerate() {
            let layer_cache = tensor_cache.layer_mut(idx);
            x = block.forward(
                &x,
                index_pos,
                &self.rope_cache,
                layer_cache,
                self.config.max_position_embeddings,
            )?;
        }

        if let Some(ref norm) = self.norm {
            x = norm.forward(&x)?;
        }

        let lm_head = self
            .lm_head
            .as_ref()
            .ok_or_else(|| CandleError::WeightNotFound("lm_head not available".to_string()))?;

        let (batch_size, seq_len, _hidden_size) = x.dims3()?;
        let x_narrow = x.narrow(1, seq_len - 1, 1)?;
        let x_reshaped = x_narrow.reshape((batch_size, self.config.hidden_size))?;
        let logits = lm_head.forward(&x_reshaped)?;

        let logits = logits.squeeze(0)?;

        let candle_sampling = Self::to_candle_sampling(sampling);
        let mut processor = LogitsProcessor::from_sampling(sampling.seed, candle_sampling);

        let token = processor.sample(&logits).map_err(CandleError::Candle)?;

        Ok(Some(SampleResult::new(token)))
    }
}

impl rig_core::Partition for CandlePartition {
    fn spec(&self) -> &PartitionSpec {
        &self.spec
    }

    fn forward(&self, input: Activation) -> std::result::Result<Activation, PartitionError> {
        let tensor = if self.has_embeddings() && input.dtype() == rig_core::DType::I8 {
            let token_ids = self
                .extract_token_ids(&input)
                .map_err(|e| PartitionError::ForwardFailed(e.to_string()))?;

            self.embed(&token_ids)
                .map_err(|e| PartitionError::ForwardFailed(e.to_string()))?
        } else {
            self.activation_to_tensor(&input)
                .map_err(|e| PartitionError::ForwardFailed(e.to_string()))?
        };

        let output = self
            .forward_impl(tensor)
            .map_err(|e| PartitionError::ForwardFailed(e.to_string()))?;

        self.tensor_to_activation(output, input.metadata.clone())
            .map_err(|e| PartitionError::ForwardFailed(e.to_string()))
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.memory_usage
    }

    fn release_request_cache(&self, _request_id: rig_core::RequestId) {
        self.clear_kv_cache();
    }

    fn forward_sample(
        &mut self,
        input: Activation,
        sampling: &SamplingParams,
    ) -> std::result::Result<Option<SampleResult>, PartitionError> {
        self.forward_sample_impl(input, sampling)
            .map_err(|e| PartitionError::ForwardFailed(e.to_string()))
    }
}

unsafe impl Send for CandlePartition {}
unsafe impl Sync for CandlePartition {}

impl rig_core::Tokenizer for CandlePartition {
    fn encode(
        &self,
        text: &str,
        add_bos: bool,
    ) -> std::result::Result<Vec<u32>, rig_core::TokenizerError> {
        self.tokenize(text, add_bos)
            .map_err(|e| rig_core::TokenizerError::EncodeFailed(e.to_string()))
    }

    fn decode(&self, tokens: &[u32]) -> std::result::Result<String, rig_core::TokenizerError> {
        self.detokenize(tokens)
            .map_err(|e| rig_core::TokenizerError::DecodeFailed(e.to_string()))
    }

    fn eos_token(&self) -> u32 {
        Self::eos_token(self)
    }

    fn bos_token(&self) -> u32 {
        Self::bos_token(self)
    }

    fn vocab_size(&self) -> usize {
        Self::vocab_size(self)
    }

    fn apply_chat_template(
        &self,
        messages: &[rig_core::ChatMessage],
        add_generation_prompt: bool,
    ) -> std::result::Result<String, rig_core::TokenizerError> {
        let template_str = self.chat_template.as_ref().ok_or_else(|| {
            rig_core::TokenizerError::EncodeFailed(
                "No chat template available for this model".into(),
            )
        })?;

        let mut env = minijinja::Environment::new();
        env.add_template("chat", template_str).map_err(|e| {
            rig_core::TokenizerError::EncodeFailed(format!("Invalid chat template: {e}"))
        })?;

        let template = env
            .get_template("chat")
            .map_err(|e| rig_core::TokenizerError::EncodeFailed(format!("Template error: {e}")))?;

        let messages_value: Vec<minijinja::Value> = messages
            .iter()
            .map(|m| {
                minijinja::context! {
                    role => m.role.as_str(),
                    content => m.content.as_str()
                }
            })
            .collect();

        let ctx = minijinja::context! {
            messages => messages_value,
            eos_token => self.eos_token_str.as_str(),
            add_generation_prompt => add_generation_prompt,
        };

        template.render(ctx).map_err(|e| {
            rig_core::TokenizerError::EncodeFailed(format!("Chat template render failed: {e}"))
        })
    }

    fn supports_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    fn encode_batch(
        &self,
        texts: &[&str],
        add_bos: bool,
    ) -> std::result::Result<Vec<Vec<u32>>, rig_core::TokenizerError> {
        self.tokenize_batch(texts, add_bos)
            .map_err(|e| rig_core::TokenizerError::EncodeFailed(e.to_string()))
    }

    fn decode_batch(
        &self,
        token_sequences: &[&[u32]],
    ) -> std::result::Result<Vec<String>, rig_core::TokenizerError> {
        self.detokenize_batch(token_sequences)
            .map_err(|e| rig_core::TokenizerError::DecodeFailed(e.to_string()))
    }

    fn create_decode_stream(
        &self,
        skip_special_tokens: bool,
    ) -> std::result::Result<Box<dyn rig_core::TokenDecodeStream>, rig_core::TokenizerError> {
        self.extract_tokenizer()
            .create_decode_stream(skip_special_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_conversion() {
        assert!(matches!(
            CandlePartition::convert_dtype(rig_core::DType::F32),
            Ok(DType::F32)
        ));
        assert!(matches!(
            CandlePartition::convert_dtype(rig_core::DType::F16),
            Ok(DType::F16)
        ));
        assert!(matches!(
            CandlePartition::convert_dtype(rig_core::DType::BF16),
            Ok(DType::BF16)
        ));
        assert!(CandlePartition::convert_dtype(rig_core::DType::I8).is_err());
    }
}
