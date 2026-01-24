#![allow(unsafe_code)]
#![allow(clippy::future_not_send)]
use std::sync::Mutex;

use candle_core::quantized::{QMatMul, gguf_file};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear, linear_no_bias, rms_norm,
};
use candle_transformers::generation::LogitsProcessor;

use rig_core::error::PartitionError;
use rig_core::types::{Activation, MemoryUsage, PartitionSpec, SampleResult, SamplingParams};

use crate::cache::RopeCache;
use crate::config::TransformerConfig;
use crate::error::{CandleError, Result};
use crate::gguf_config::GgufConfig;
use crate::kv_cache::CandleKvCache;
use crate::layers::{Attention, Mlp, TransformerBlock};
use crate::model::{
    ActivationFn, AttentionConfig, detect_architecture_config, detect_architecture_gguf,
};
use crate::utils::{
    CachedSampler, activation_to_tensor, extract_token_ids, tensor_to_activation,
    to_candle_sampling,
};
use crate::weights::{GgufLoader, QuantizedLinear, SafetensorLoader, Weight};

const DEFAULT_MAX_CONTEXT_LENGTH: usize = 4096;

pub struct UnifiedPartition<W: Weight> {
    spec: PartitionSpec,
    embed_tokens: Option<Embedding>,
    blocks: Vec<TransformerBlock<W>>,
    norm: Option<RmsNorm>,
    lm_head: Option<LmHead<W>>,
    rope_cache: RopeCache,
    kv_cache: Mutex<CandleKvCache>,
    cached_sampler: Mutex<Option<CachedSampler>>,
    hidden_size: usize,
    #[allow(dead_code)]
    vocab_size: usize,
    max_context_length: usize,
    device: Device,
    dtype: DType,
    memory_usage: MemoryUsage,
}

enum LmHead<W: Weight> {
    Weight(W),
    QMatMul(QMatMul),
}

impl<W: Weight> LmHead<W> {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Weight(w) => w.forward(input),
            Self::QMatMul(q) => q.forward(input),
        }
    }
}

impl UnifiedPartition<Linear> {
    pub fn load_safetensor(
        model_path: &std::path::Path,
        spec: &PartitionSpec,
        _total_layers: usize,
        device: &Device,
    ) -> Result<Self> {
        if !model_path.exists() {
            return Err(CandleError::ModelNotFound(model_path.to_path_buf()));
        }

        let config_path = model_path.join("config.json");
        let config = TransformerConfig::from_file(&config_path)?;

        let arch = detect_architecture_config(&config_path)?;
        let arch_ref = arch.as_ref();

        let dtype = convert_dtype(spec.dtype)?;

        let safetensor_files = find_safetensor_files(model_path)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)? };

        let attn_config = if vb
            .pp("model.layers.0.self_attn.q_proj")
            .contains_tensor("bias")
        {
            AttentionConfig::QKV_BIAS
        } else {
            arch_ref.attention_config()
        };

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

        let loader = SafetensorLoader::new(vb.clone());
        let activation = convert_activation(&config);
        let head_dim = config.head_dim();
        let num_attention_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads();

        let mut blocks = Vec::with_capacity(spec.layer_range.len());
        for layer_idx in spec.layer_range.clone() {
            let block = load_safetensor_block(
                &loader,
                layer_idx,
                &config,
                attn_config,
                activation,
                head_dim,
                num_attention_heads,
                num_kv_heads,
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

            (Some(norm), Some(LmHead::Weight(lm_head)))
        } else {
            (None, None)
        };

        let max_context_length = spec
            .max_context_length
            .unwrap_or(config.max_position_embeddings)
            .min(config.max_position_embeddings);

        let rope_cache = RopeCache::with_params(
            head_dim,
            max_context_length,
            config.rope_theta,
            dtype,
            device,
        )?;

        let dtype_size = dtype.size_in_bytes();
        let memory_per_token = 2 * num_kv_heads * head_dim * dtype_size;
        let kv_cache = Mutex::new(CandleKvCache::new(blocks.len(), 1, memory_per_token));

        let memory_usage = estimate_memory(
            config.hidden_size,
            config.intermediate_size,
            config.vocab_size,
            spec,
            is_first,
            is_last,
            config.tie_word_embeddings,
        );

        Ok(Self {
            spec: spec.clone(),
            embed_tokens,
            blocks,
            norm,
            lm_head,
            rope_cache,
            kv_cache,
            cached_sampler: Mutex::new(None),
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            max_context_length,
            device: device.clone(),
            dtype,
            memory_usage,
        })
    }
}

impl UnifiedPartition<QuantizedLinear> {
    pub fn load_gguf(
        model_path: &std::path::Path,
        spec: &PartitionSpec,
        device: &Device,
    ) -> Result<Self> {
        Self::load_gguf_with_context(model_path, spec, device, DEFAULT_MAX_CONTEXT_LENGTH)
    }
    pub fn load_gguf_with_context(
        model_path: &std::path::Path,
        spec: &PartitionSpec,
        device: &Device,
        requested_context_length: usize,
    ) -> Result<Self> {
        if !model_path.exists() {
            return Err(CandleError::ModelNotFound(model_path.to_path_buf()));
        }

        let gguf_path = find_gguf_file(model_path)?;
        let mut file = std::fs::File::open(&gguf_path)?;

        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| CandleError::Internal(format!("Failed to read GGUF content: {e}")))?;

        let mut config = GgufConfig::from_gguf_content(&content)?;

        let config_json_path = model_path.join("config.json");
        if config_json_path.exists() {
            let rope_scaling = GgufConfig::load_rope_scaling_from_config(&config_json_path);
            if let Some(ref scaling) = rope_scaling {
                tracing::debug!(
                    factor = scaling.factor,
                    low_freq_factor = scaling.low_freq_factor,
                    high_freq_factor = scaling.high_freq_factor,
                    original_max_pos = scaling.original_max_position_embeddings,
                    "Loaded Llama3 RoPE scaling from config.json"
                );
            }
            config = config.with_rope_scaling(rope_scaling);
        }

        let arch = detect_architecture_gguf(&content)?;
        let attn_config = arch.attention_config();

        let is_first = spec.layer_range.start == 0;
        let is_last = spec.layer_range.end == config.num_hidden_layers;

        let dtype = DType::F32;

        let loader = GgufLoader::new(content, file, device);
        let embed_tokens = if is_first {
            let weight = loader.load_dequantized("token_embd.weight")?;
            let weight = weight.to_dtype(dtype)?;
            Some(Embedding::new(weight, config.hidden_size))
        } else {
            None
        };

        let head_dim = config.head_dim();
        let activation = ActivationFn::Silu;

        let mut blocks = Vec::with_capacity(spec.layer_range.len());
        for layer_idx in spec.layer_range.clone() {
            let block = load_gguf_block(
                &loader,
                layer_idx,
                &config,
                attn_config,
                activation,
                head_dim,
                dtype,
            )?;
            blocks.push(block);
        }

        let (norm, lm_head) = if is_last {
            let norm_weight = loader.load_dequantized("output_norm.weight")?;
            let norm_weight = norm_weight.to_dtype(dtype)?;
            let norm = RmsNorm::new(norm_weight, config.rms_norm_eps);

            let lm_head_qmatmul = match loader.load_qmatmul("output.weight") {
                Ok(q) => q,
                Err(_) => loader.load_qmatmul("token_embd.weight")?,
            };

            (Some(norm), Some(LmHead::QMatMul(lm_head_qmatmul)))
        } else {
            (None, None)
        };

        let max_context_length = spec
            .max_context_length
            .unwrap_or(requested_context_length)
            .min(config.max_position_embeddings);

        let rope_cache = RopeCache::with_scaling(
            head_dim,
            max_context_length,
            config.rope_theta,
            config.rope_scaling.as_ref(),
            dtype,
            device,
        )?;

        let memory_per_token = 2 * config.num_key_value_heads * head_dim * dtype.size_in_bytes();
        let kv_cache = Mutex::new(CandleKvCache::new(blocks.len(), 1, memory_per_token));

        let memory_usage = estimate_gguf_memory(&config, spec, is_first, is_last);

        let vocab_size = 32000;

        Ok(Self {
            spec: spec.clone(),
            embed_tokens,
            blocks,
            norm,
            lm_head,
            rope_cache,
            kv_cache,
            cached_sampler: Mutex::new(None),
            hidden_size: config.hidden_size,
            vocab_size,
            max_context_length,
            device: device.clone(),
            dtype,
            memory_usage,
        })
    }
}

impl<W: Weight> UnifiedPartition<W> {
    fn forward_impl(&self, input: Tensor) -> Result<Tensor> {
        let mut x = input;

        let mut kv_cache = self.kv_cache.lock().map_err(|e| {
            CandleError::Candle(candle_core::Error::Msg(format!(
                "KV cache lock failed: {e}"
            )))
        })?;

        let tensor_cache = kv_cache.tensor_cache_mut();
        let index_pos = tensor_cache.seq_len();

        tracing::trace!(
            layer_range = ?self.spec.layer_range,
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
                self.max_context_length,
            )?;
        }

        if let Some(ref norm) = self.norm {
            x = norm.forward(&x)?;
        }

        if let Some(ref lm_head) = self.lm_head {
            let (batch_size, seq_len, _hidden_size) = x.dims3()?;
            let x_narrow = x.narrow(1, seq_len - 1, 1)?;
            let x_reshaped = x_narrow.reshape((batch_size, self.hidden_size))?;
            let logits = lm_head.forward(&x_reshaped)?;
            let vocab_size = logits.dim(1)?;
            return Ok(logits.reshape((batch_size, 1, vocab_size))?);
        }

        Ok(x)
    }

    fn forward_sample_impl(
        &self,
        input: Activation,
        sampling: &SamplingParams,
    ) -> Result<Option<SampleResult>> {
        if self.lm_head.is_none() {
            return Ok(None);
        }

        let tensor = if self.embed_tokens.is_some() && input.dtype() == rig_core::DType::I8 {
            let token_ids = extract_token_ids(&input, &self.device)?;
            self.embed(&token_ids)?
        } else {
            activation_to_tensor(&input, self.dtype, &self.device)?
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
                self.max_context_length,
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
        let x_reshaped = x_narrow.reshape((batch_size, self.hidden_size))?;
        let logits = lm_head.forward(&x_reshaped)?;

        let logits = logits.squeeze(0)?;

        let mut cached = self.cached_sampler.lock().map_err(|e| {
            CandleError::Candle(candle_core::Error::Msg(format!("Sampler lock failed: {e}")))
        })?;

        let needs_update = cached.as_ref().is_none_or(|c| c.params != *sampling);

        if needs_update {
            let candle_sampling = to_candle_sampling(sampling);
            let new_processor = LogitsProcessor::from_sampling(sampling.seed, candle_sampling);
            *cached = Some(CachedSampler {
                processor: new_processor,
                params: sampling.clone(),
            });
        }

        let token = cached
            .as_mut()
            .map(|c| c.processor.sample(&logits))
            .transpose()
            .map_err(CandleError::Candle)?
            .ok_or_else(|| {
                CandleError::Internal("Sampler should have been initialized".to_string())
            })?;

        Ok(Some(SampleResult::new(token)))
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
    pub fn device(&self) -> &Device {
        &self.device
    }

    #[must_use]
    pub const fn max_context_length(&self) -> usize {
        self.max_context_length
    }

    pub fn clear_kv_cache(&self) {
        if let Ok(mut cache) = self.kv_cache.lock() {
            cache.clear();
        }
        if let Ok(mut sampler) = self.cached_sampler.lock() {
            *sampler = None;
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
}

impl<W: Weight> rig_core::Partition for UnifiedPartition<W> {
    fn spec(&self) -> &PartitionSpec {
        &self.spec
    }

    fn forward(&self, input: Activation) -> std::result::Result<Activation, PartitionError> {
        let tensor = if self.has_embeddings() && input.dtype() == rig_core::DType::I8 {
            let token_ids = extract_token_ids(&input, &self.device)
                .map_err(|e| PartitionError::ForwardFailed(e.to_string()))?;

            self.embed(&token_ids)
                .map_err(|e| PartitionError::ForwardFailed(e.to_string()))?
        } else {
            activation_to_tensor(&input, self.dtype, &self.device)
                .map_err(|e| PartitionError::ForwardFailed(e.to_string()))?
        };

        let output = self
            .forward_impl(tensor)
            .map_err(|e| PartitionError::ForwardFailed(e.to_string()))?;

        tensor_to_activation(output, self.dtype, input.metadata.clone())
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

unsafe impl<W: Weight> Send for UnifiedPartition<W> {}
unsafe impl<W: Weight> Sync for UnifiedPartition<W> {}

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

fn convert_activation(config: &TransformerConfig) -> ActivationFn {
    match config.hidden_act {
        crate::config::Activation::Silu => ActivationFn::Silu,
        crate::config::Activation::Gelu => ActivationFn::Gelu,
        crate::config::Activation::GeluNew => ActivationFn::GeluNew,
        crate::config::Activation::Relu => ActivationFn::Relu,
    }
}

fn find_safetensor_files(model_path: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
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

fn find_gguf_file(dir: &std::path::Path) -> Result<std::path::PathBuf> {
    let mut gguf_files = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == "gguf" {
                gguf_files.push(path);
            }
        }
    }

    match gguf_files.len() {
        0 => Err(CandleError::Internal(format!(
            "No GGUF file found in directory: {}",
            dir.display()
        ))),
        1 => Ok(gguf_files.remove(0)),
        _ => {
            let model_gguf = gguf_files
                .iter()
                .find(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n == "model.gguf")
                })
                .cloned();

            model_gguf.map_or_else(
                || {
                    gguf_files.sort();
                    Ok(gguf_files.remove(0))
                },
                Ok,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn load_safetensor_block(
    loader: &SafetensorLoader,
    layer_idx: usize,
    config: &TransformerConfig,
    attn_config: AttentionConfig,
    activation: ActivationFn,
    head_dim: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
) -> Result<TransformerBlock<Linear>> {
    let q_dim = num_attention_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let layer_vb = loader.var_builder().pp(format!("model.layers.{layer_idx}"));

    let (q_proj, k_proj, v_proj) = if attn_config.q_bias {
        (
            linear(config.hidden_size, q_dim, layer_vb.pp("self_attn.q_proj"))?,
            linear(config.hidden_size, kv_dim, layer_vb.pp("self_attn.k_proj"))?,
            linear(config.hidden_size, kv_dim, layer_vb.pp("self_attn.v_proj"))?,
        )
    } else {
        (
            linear_no_bias(config.hidden_size, q_dim, layer_vb.pp("self_attn.q_proj"))?,
            linear_no_bias(config.hidden_size, kv_dim, layer_vb.pp("self_attn.k_proj"))?,
            linear_no_bias(config.hidden_size, kv_dim, layer_vb.pp("self_attn.v_proj"))?,
        )
    };
    let o_proj = linear_no_bias(q_dim, config.hidden_size, layer_vb.pp("self_attn.o_proj"))?;

    let attn = Attention::new(
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        num_attention_heads,
        num_kv_heads,
        head_dim,
    );

    let input_norm = rms_norm(
        config.hidden_size,
        config.rms_norm_eps,
        layer_vb.pp("input_layernorm"),
    )?;
    let post_attn_norm = rms_norm(
        config.hidden_size,
        config.rms_norm_eps,
        layer_vb.pp("post_attention_layernorm"),
    )?;

    let gate_proj = linear_no_bias(
        config.hidden_size,
        config.intermediate_size,
        layer_vb.pp("mlp.gate_proj"),
    )?;
    let up_proj = linear_no_bias(
        config.hidden_size,
        config.intermediate_size,
        layer_vb.pp("mlp.up_proj"),
    )?;
    let down_proj = linear_no_bias(
        config.intermediate_size,
        config.hidden_size,
        layer_vb.pp("mlp.down_proj"),
    )?;

    let mlp = Mlp::new(gate_proj, up_proj, down_proj, activation);

    Ok(TransformerBlock::new(input_norm, attn, post_attn_norm, mlp))
}

fn load_gguf_block(
    loader: &GgufLoader,
    layer_idx: usize,
    config: &GgufConfig,
    attn_config: AttentionConfig,
    activation: ActivationFn,
    head_dim: usize,
    dtype: DType,
) -> Result<TransformerBlock<QuantizedLinear>> {
    let prefix = format!("blk.{layer_idx}");

    let q_weight = loader.load_qmatmul(&format!("{prefix}.attn_q.weight"))?;
    let k_weight = loader.load_qmatmul(&format!("{prefix}.attn_k.weight"))?;
    let v_weight = loader.load_qmatmul(&format!("{prefix}.attn_v.weight"))?;
    let o_weight = loader.load_qmatmul(&format!("{prefix}.attn_output.weight"))?;

    let q_bias = if attn_config.q_bias {
        loader
            .load_dequantized(&format!("{prefix}.attn_q.bias"))
            .ok()
            .map(|t| t.to_dtype(dtype))
            .transpose()?
    } else {
        None
    };
    let k_bias = if attn_config.k_bias {
        loader
            .load_dequantized(&format!("{prefix}.attn_k.bias"))
            .ok()
            .map(|t| t.to_dtype(dtype))
            .transpose()?
    } else {
        None
    };
    let v_bias = if attn_config.v_bias {
        loader
            .load_dequantized(&format!("{prefix}.attn_v.bias"))
            .ok()
            .map(|t| t.to_dtype(dtype))
            .transpose()?
    } else {
        None
    };

    let q_proj = QuantizedLinear::new(q_weight, q_bias);
    let k_proj = QuantizedLinear::new(k_weight, k_bias);
    let v_proj = QuantizedLinear::new(v_weight, v_bias);
    let o_proj = QuantizedLinear::without_bias(o_weight);

    let attn = Attention::new(
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        config.num_attention_heads,
        config.num_key_value_heads,
        head_dim,
    );

    let attn_norm_weight = loader.load_dequantized(&format!("{prefix}.attn_norm.weight"))?;
    let attn_norm_weight = attn_norm_weight.to_dtype(dtype)?;
    let attn_norm = RmsNorm::new(attn_norm_weight, config.rms_norm_eps);

    let gate_weight = loader.load_qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
    let up_weight = loader.load_qmatmul(&format!("{prefix}.ffn_up.weight"))?;
    let down_weight = loader.load_qmatmul(&format!("{prefix}.ffn_down.weight"))?;

    let gate_proj = QuantizedLinear::without_bias(gate_weight);
    let up_proj = QuantizedLinear::without_bias(up_weight);
    let down_proj = QuantizedLinear::without_bias(down_weight);

    let mlp = Mlp::new(gate_proj, up_proj, down_proj, activation);

    let ffn_norm_weight = loader.load_dequantized(&format!("{prefix}.ffn_norm.weight"))?;
    let ffn_norm_weight = ffn_norm_weight.to_dtype(dtype)?;
    let ffn_norm = RmsNorm::new(ffn_norm_weight, config.rms_norm_eps);

    Ok(TransformerBlock::new(attn_norm, attn, ffn_norm, mlp))
}

fn estimate_memory(
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    spec: &PartitionSpec,
    is_first: bool,
    is_last: bool,
    tie_word_embeddings: bool,
) -> MemoryUsage {
    let dtype_size = match spec.dtype {
        rig_core::DType::F32 => 4,
        rig_core::DType::F16 | rig_core::DType::BF16 => 2,
        _ => 4,
    };

    let qkvo_size = 4 * hidden_size * hidden_size;
    let mlp_size = 3 * hidden_size * intermediate_size;
    let norm_size = 2 * hidden_size;
    let layer_weights = (qkvo_size + mlp_size + norm_size) * dtype_size;

    let num_layers = spec.layer_range.len();
    let mut weights_bytes = (layer_weights * num_layers) as u64;

    if is_first {
        weights_bytes += (vocab_size * hidden_size * dtype_size) as u64;
    }

    if is_last && !tie_word_embeddings {
        weights_bytes += (vocab_size * hidden_size * dtype_size) as u64;
    }

    MemoryUsage {
        weights_bytes,
        cache_bytes: 0,
        scratch_bytes: 0,
    }
}

fn estimate_gguf_memory(
    config: &GgufConfig,
    spec: &PartitionSpec,
    is_first: bool,
    is_last: bool,
) -> MemoryUsage {
    let bytes_per_weight = 1;

    let qkvo_size = 4 * config.hidden_size * config.hidden_size;
    let mlp_size = 3 * config.hidden_size * config.intermediate_size;
    let norm_size = 2 * config.hidden_size * 4;
    let layer_weights = qkvo_size * bytes_per_weight + mlp_size * bytes_per_weight + norm_size;

    let num_layers = spec.layer_range.len();
    let mut weights_bytes = (layer_weights * num_layers) as u64;

    if is_first {
        weights_bytes += (32000 * config.hidden_size * 4) as u64;
    }

    if is_last {
        weights_bytes += (config.hidden_size * config.hidden_size) as u64;
    }

    MemoryUsage {
        weights_bytes,
        cache_bytes: 0,
        scratch_bytes: 0,
    }
}

pub type CandlePartition = UnifiedPartition<Linear>;
pub type GgufPartition = UnifiedPartition<QuantizedLinear>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_conversion() {
        assert!(matches!(
            convert_dtype(rig_core::DType::F32),
            Ok(DType::F32)
        ));
        assert!(matches!(
            convert_dtype(rig_core::DType::F16),
            Ok(DType::F16)
        ));
        assert!(matches!(
            convert_dtype(rig_core::DType::BF16),
            Ok(DType::BF16)
        ));
        assert!(convert_dtype(rig_core::DType::I8).is_err());
    }
}
