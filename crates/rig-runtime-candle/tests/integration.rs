#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation
)]

use std::path::PathBuf;

use candle_core::{DType, Device};
#[cfg(feature = "metal")]
use rig_core::Partition;
use rig_core::Runtime;
use rig_core::types::PartitionSpec;
#[cfg(feature = "metal")]
use rig_core::types::{Activation, ActivationMetadata, RequestId, Shape, TensorData};
use rig_runtime_candle::{CandlePartition, CandleRuntime, TransformerConfig};

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("should have parent")
        .parent()
        .expect("should have grandparent")
        .join("models/tiny-llama")
}

fn model_exists() -> bool {
    let path = model_path();
    path.exists() && path.join("config.json").exists() && path.join("model.safetensors").exists()
}

#[test]
fn test_config_parsing() {
    if !model_exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path());
        return;
    }

    let config_path = model_path().join("config.json");
    let config = TransformerConfig::from_file(&config_path)
        .unwrap_or_else(|e| panic!("Failed to parse config: {e}"));

    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.intermediate_size, 5632);
    assert_eq!(config.num_hidden_layers, 22);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_kv_heads(), 4);
    assert_eq!(config.vocab_size, 32000);
    assert!((config.rope_theta - 10000.0).abs() < f64::EPSILON);
    assert!((config.rms_norm_eps - 1e-5).abs() < 1e-10);
    assert_eq!(config.max_position_embeddings, 2048);
    assert!(!config.tie_word_embeddings);

    assert_eq!(config.head_dim(), 64);
    assert_eq!(config.num_queries_per_kv(), 8);
    assert!(config.is_gqa());
    assert!(!config.is_mqa());
}

#[test]
fn test_runtime_creation() {
    let runtime = CandleRuntime::cpu().unwrap_or_else(|e| panic!("Failed to create runtime: {e}"));

    let caps = runtime.capabilities();
    assert_eq!(caps.runtime_type, "candle_cpu");
    assert!(caps.supported_dtypes.contains(&rig_core::DType::F32));
}

#[test]
fn test_load_first_partition() {
    if !model_exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path());
        return;
    }

    let device = Device::Cpu;

    let spec = PartitionSpec::new(0..4, rig_core::DType::F32);

    let partition = CandlePartition::load(model_path(), &spec, 22, &device)
        .unwrap_or_else(|e| panic!("Failed to load partition: {e}"));

    assert!(partition.has_embeddings());
    assert!(!partition.has_lm_head());
    assert_eq!(partition.config().num_hidden_layers, 22);
    assert_eq!(partition.config().hidden_size, 2048);
}

#[test]
fn test_load_middle_partition() {
    if !model_exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path());
        return;
    }

    let device = Device::Cpu;

    let spec = PartitionSpec::new(10..14, rig_core::DType::F32);
    let partition = CandlePartition::load(model_path(), &spec, 22, &device)
        .unwrap_or_else(|e| panic!("Failed to load partition: {e}"));

    assert!(!partition.has_embeddings());
    assert!(!partition.has_lm_head());
}

#[test]
fn test_load_last_partition() {
    if !model_exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path());
        return;
    }

    let device = Device::Cpu;

    let spec = PartitionSpec::new(18..22, rig_core::DType::F32);
    let partition = CandlePartition::load(model_path(), &spec, 22, &device)
        .unwrap_or_else(|e| panic!("Failed to load partition: {e}"));

    assert!(!partition.has_embeddings());
    assert!(partition.has_lm_head());
}

#[test]
fn test_load_single_partition_full_model() {
    if !model_exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path());
        return;
    }

    let device = Device::Cpu;

    let spec = PartitionSpec::new(0..22, rig_core::DType::F32);
    let partition = CandlePartition::load(model_path(), &spec, 22, &device)
        .unwrap_or_else(|e| panic!("Failed to load partition: {e}"));

    assert!(partition.has_embeddings());
    assert!(partition.has_lm_head());
}

#[test]
fn test_first_partition_embedding() {
    if !model_exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path());
        return;
    }

    let device = Device::Cpu;

    let spec = PartitionSpec::new(0..2, rig_core::DType::F32);

    let partition = CandlePartition::load(model_path(), &spec, 22, &device)
        .unwrap_or_else(|e| panic!("Failed to load partition: {e}"));

    let tokens = candle_core::Tensor::new(&[[1u32, 2, 3, 4]], &device)
        .unwrap_or_else(|e| panic!("Failed to create tokens: {e}"));

    let embedded = partition
        .embed(&tokens)
        .unwrap_or_else(|e| panic!("Failed to embed: {e}"));

    assert_eq!(embedded.dims(), &[1, 4, 2048]);
}

#[test]
fn test_rope_cache_for_model() {
    if !model_exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path());
        return;
    }

    let config_path = model_path().join("config.json");
    let config = TransformerConfig::from_file(&config_path)
        .unwrap_or_else(|e| panic!("Failed to parse config: {e}"));

    let device = Device::Cpu;

    let cache = rig_runtime_candle::RopeCache::new(&config, DType::F32, &device)
        .unwrap_or_else(|e| panic!("Failed to create cache: {e}"));

    let max_seq = cache
        .max_seq_len()
        .unwrap_or_else(|e| panic!("Failed to get max_seq_len: {e}"));
    assert_eq!(max_seq, 2048);

    assert_eq!(cache.cos().dims(), &[2048, 32]);
    assert_eq!(cache.sin().dims(), &[2048, 32]);

    let (cos, sin) = cache
        .get(16, 0)
        .unwrap_or_else(|e| panic!("Failed to get slice: {e}"));
    assert_eq!(cos.dims(), &[16, 32]);
    assert_eq!(sin.dims(), &[16, 32]);
}

#[test]
#[cfg(feature = "metal")]
fn test_forward_with_token_ids() {
    if !model_exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path());
        return;
    }

    let device = Device::new_metal(0).expect("Metal device required for SDPA");

    let spec = PartitionSpec::new(0..4, rig_core::DType::F32);
    let partition = CandlePartition::load(model_path(), &spec, 22, &device)
        .unwrap_or_else(|e| panic!("Failed to load partition: {e}"));

    let tokens: Vec<u32> = vec![1, 2, 3, 4];
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for token in &tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }

    let shape = Shape::new(vec![1, tokens.len(), 1]);
    let data = TensorData::cpu(bytes, rig_core::DType::I8);
    let metadata = ActivationMetadata::new(
        RequestId::new(),
        0,
        (0..tokens.len() as u32).collect(),
        true,
    );
    let activation = Activation::new(data, shape, metadata);

    let output = partition
        .forward(activation)
        .unwrap_or_else(|e| panic!("Failed to forward: {e}"));

    assert_eq!(output.shape.dims(), &[1, 4, 2048]);
    assert_eq!(output.dtype(), rig_core::DType::F32);
}

#[test]
#[cfg(feature = "metal")]
fn test_forward_decode_single_token() {
    if !model_exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path());
        return;
    }

    let device = Device::new_metal(0).expect("Metal device required for SDPA");

    let spec = PartitionSpec::new(0..4, rig_core::DType::F32);
    let partition = CandlePartition::load(model_path(), &spec, 22, &device)
        .unwrap_or_else(|e| panic!("Failed to load partition: {e}"));

    let tokens: Vec<u32> = vec![1, 2, 3, 4];
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for token in &tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }

    let shape = Shape::new(vec![1, tokens.len(), 1]);
    let data = TensorData::cpu(bytes, rig_core::DType::I8);
    let metadata = ActivationMetadata::new(
        RequestId::new(),
        0,
        (0..tokens.len() as u32).collect(),
        true,
    );
    let activation = Activation::new(data, shape, metadata);
    let _ = partition.forward(activation).unwrap();

    let token: u32 = 5;
    let bytes = token.to_le_bytes().to_vec();
    let shape = Shape::new(vec![1, 1, 1]);
    let data = TensorData::cpu(bytes, rig_core::DType::I8);
    let metadata = ActivationMetadata::new(RequestId::new(), 4, vec![4], false);
    let activation = Activation::new(data, shape, metadata);

    let output = partition
        .forward(activation)
        .unwrap_or_else(|e| panic!("Failed to forward decode: {e}"));

    assert_eq!(output.shape.dims(), &[1, 1, 2048]);
}
