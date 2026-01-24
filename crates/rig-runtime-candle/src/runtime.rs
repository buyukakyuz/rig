use candle_core::Device;

use std::path::{Path, PathBuf};

use rig_core::LoadedPartition;
use rig_core::error::RuntimeError;
use rig_core::types::{DType, ModelId, ModelSpec, PartitionSpec, RuntimeCapabilities, RuntimeId};

use crate::config::TransformerConfig;
use crate::error::CandleError;
use crate::gguf_config::GgufConfig;
use crate::partition::{CandlePartition, GgufPartition};
use crate::tokenizer::CandleTokenizer;
use crate::utils::load_tokenizer_config;

#[derive(Debug, Clone)]
pub enum ModelFormat {
    Gguf(PathBuf),
    Safetensor,
}

pub fn detect_model_format(path: &Path) -> ModelFormat {
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.extension().is_some_and(|ext| ext == "gguf") {
                return ModelFormat::Gguf(entry_path);
            }
        }
    }
    ModelFormat::Safetensor
}

pub struct CandleRuntime {
    id: RuntimeId,
    device: Device,
    capabilities: RuntimeCapabilities,
}

impl CandleRuntime {
    pub fn new() -> Result<Self, CandleError> {
        let device = Self::select_device()?;
        Self::with_device(device)
    }

    pub fn with_device(device: Device) -> Result<Self, CandleError> {
        let id = RuntimeId::new();

        let (_, total_memory) = crate::memory::query_device_memory(&device).unwrap_or_else(|e| {
            tracing::warn!("Failed to query device memory: {e}, using 0");
            (0, 0)
        });

        tracing::debug!(
            device = ?device,
            total_memory_bytes = total_memory,
            total_memory_gb = total_memory / (1024 * 1024 * 1024),
            "Device memory detected"
        );

        let supported_dtypes = match &device {
            Device::Cpu => vec![DType::F32, DType::F16, DType::BF16],
            Device::Cuda(_) => vec![DType::F32, DType::F16, DType::BF16],
            Device::Metal(_) => vec![DType::F32, DType::F16, DType::BF16],
        };

        let runtime_type = match &device {
            Device::Cpu => "candle_cpu",
            Device::Cuda(_) => "candle_cuda",
            Device::Metal(_) => "candle_metal",
        };

        let capabilities = RuntimeCapabilities::new(runtime_type, total_memory, supported_dtypes);

        Ok(Self {
            id,
            device,
            capabilities,
        })
    }

    pub fn cpu() -> Result<Self, CandleError> {
        Self::with_device(Device::Cpu)
    }

    fn select_device() -> Result<Device, CandleError> {
        #[cfg(feature = "metal")]
        {
            if let Ok(device) = Device::new_metal(0) {
                tracing::debug!("Using Metal device");
                return Ok(device);
            }
        }

        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                tracing::debug!("Using CUDA device 0");
                return Ok(device);
            }
        }

        tracing::debug!("Using CPU device");
        Ok(Device::Cpu)
    }

    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn load_config(
        &self,
        model_path: impl AsRef<std::path::Path>,
    ) -> Result<TransformerConfig, CandleError> {
        let config_path = model_path.as_ref().join("config.json");
        TransformerConfig::from_file(&config_path).map_err(Into::into)
    }
}

impl rig_core::Runtime for CandleRuntime {
    fn id(&self) -> RuntimeId {
        self.id
    }

    fn capabilities(&self) -> RuntimeCapabilities {
        self.capabilities.clone()
    }

    fn discover_model(&self, model_id: ModelId, path: &Path) -> Result<ModelSpec, RuntimeError> {
        match detect_model_format(path) {
            ModelFormat::Gguf(gguf_path) => {
                let config =
                    GgufConfig::from_file(&gguf_path).map_err(|e| RuntimeError::LoadFailed {
                        model: path.display().to_string(),
                        reason: e.to_string(),
                    })?;

                Ok(ModelSpec::new(
                    model_id,
                    path,
                    config.num_hidden_layers,
                    config.hidden_size,
                ))
            }
            ModelFormat::Safetensor => {
                let config = self
                    .load_config(path)
                    .map_err(|e| RuntimeError::LoadFailed {
                        model: path.display().to_string(),
                        reason: e.to_string(),
                    })?;

                Ok(ModelSpec::new(
                    model_id,
                    path,
                    config.num_hidden_layers,
                    config.hidden_size,
                ))
            }
        }
    }

    fn load_partition(
        &self,
        model: &ModelSpec,
        partition: &PartitionSpec,
    ) -> impl std::future::Future<Output = Result<LoadedPartition, RuntimeError>> + Send {
        let model_path = model.path.clone();
        let partition = partition.clone();
        let device = self.device.clone();
        let total_layers = model.num_layers;

        async move {
            if partition.layer_range.end > total_layers {
                return Err(RuntimeError::Internal(format!(
                    "Layer range {:?} exceeds model layers ({})",
                    partition.layer_range, total_layers
                )));
            }

            match detect_model_format(&model_path) {
                ModelFormat::Gguf(_) => {
                    Self::load_gguf_partition(model_path, partition, device).await
                }
                ModelFormat::Safetensor => {
                    Self::load_safetensor_partition(model_path, partition, total_layers, device)
                        .await
                }
            }
        }
    }
}

impl CandleRuntime {
    async fn load_gguf_partition(
        model_path: PathBuf,
        partition: PartitionSpec,
        device: Device,
    ) -> Result<LoadedPartition, RuntimeError> {
        let model_path_for_tokenizer = model_path.clone();
        let model_path_for_error = model_path.clone();

        if partition.dtype != DType::F32 {
            tracing::debug!(
                requested_dtype = ?partition.dtype,
                "GGUF models require F32 dtype, overriding requested dtype"
            );
        }

        let partition_result: Result<GgufPartition, CandleError> =
            tokio::task::spawn_blocking(move || {
                GgufPartition::load_gguf(&model_path, &partition, &device)
            })
            .await
            .map_err(|e| RuntimeError::Internal(format!("Task join error: {e}")))?;

        let gguf_partition = partition_result.map_err(|e| match e {
            CandleError::ModelNotFound(path) => RuntimeError::ModelNotFound(path),
            CandleError::Config(config_err) => RuntimeError::LoadFailed {
                model: model_path_for_error.display().to_string(),
                reason: config_err.to_string(),
            },
            other => RuntimeError::Internal(other.to_string()),
        })?;

        let tokenizer = Self::load_tokenizer(&model_path_for_tokenizer)?;
        let tokenizer: Box<dyn rig_core::Tokenizer> = Box::new(tokenizer);
        let partition: Box<dyn rig_core::Partition> = Box::new(gguf_partition);

        Ok(LoadedPartition::new(partition, Some(tokenizer)))
    }

    pub async fn load_gguf_partition_with_context(
        model_path: PathBuf,
        partition: PartitionSpec,
        device: Device,
        context_length: usize,
    ) -> Result<LoadedPartition, RuntimeError> {
        let model_path_for_tokenizer = model_path.clone();
        let model_path_for_error = model_path.clone();

        let partition_result: Result<GgufPartition, CandleError> =
            tokio::task::spawn_blocking(move || {
                GgufPartition::load_gguf_with_context(
                    &model_path,
                    &partition,
                    &device,
                    context_length,
                )
            })
            .await
            .map_err(|e| RuntimeError::Internal(format!("Task join error: {e}")))?;

        let gguf_partition = partition_result.map_err(|e| match e {
            CandleError::ModelNotFound(path) => RuntimeError::ModelNotFound(path),
            CandleError::Config(config_err) => RuntimeError::LoadFailed {
                model: model_path_for_error.display().to_string(),
                reason: config_err.to_string(),
            },
            other => RuntimeError::Internal(other.to_string()),
        })?;

        let tokenizer = Self::load_tokenizer(&model_path_for_tokenizer)?;
        let tokenizer: Box<dyn rig_core::Tokenizer> = Box::new(tokenizer);
        let partition: Box<dyn rig_core::Partition> = Box::new(gguf_partition);

        Ok(LoadedPartition::new(partition, Some(tokenizer)))
    }

    async fn load_safetensor_partition(
        model_path: PathBuf,
        partition: PartitionSpec,
        total_layers: usize,
        device: Device,
    ) -> Result<LoadedPartition, RuntimeError> {
        let model_path_for_tokenizer = model_path.clone();
        let model_path_for_error = model_path.clone();

        let partition_result: Result<CandlePartition, CandleError> =
            tokio::task::spawn_blocking(move || {
                CandlePartition::load_safetensor(&model_path, &partition, total_layers, &device)
            })
            .await
            .map_err(|e| RuntimeError::Internal(format!("Task join error: {e}")))?;

        let candle_partition = partition_result.map_err(|e| match e {
            CandleError::ModelNotFound(path) => RuntimeError::ModelNotFound(path),
            CandleError::Config(config_err) => RuntimeError::LoadFailed {
                model: model_path_for_error.display().to_string(),
                reason: config_err.to_string(),
            },
            CandleError::NoSafetensorFiles(path) => RuntimeError::LoadFailed {
                model: path.display().to_string(),
                reason: "No safetensor files found".to_string(),
            },
            other => RuntimeError::Internal(other.to_string()),
        })?;

        let tokenizer = Self::load_tokenizer(&model_path_for_tokenizer)?;
        let tokenizer: Box<dyn rig_core::Tokenizer> = Box::new(tokenizer);
        let partition: Box<dyn rig_core::Partition> = Box::new(candle_partition);

        Ok(LoadedPartition::new(partition, Some(tokenizer)))
    }

    fn load_tokenizer(model_path: &Path) -> Result<CandleTokenizer, RuntimeError> {
        use tokenizers::Tokenizer as HfTokenizer;

        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer =
            HfTokenizer::from_file(&tokenizer_path).map_err(|e| RuntimeError::LoadFailed {
                model: model_path.display().to_string(),
                reason: format!("Failed to load tokenizer: {e}"),
            })?;

        let tokenizer_config =
            load_tokenizer_config(model_path).map_err(|e| RuntimeError::LoadFailed {
                model: model_path.display().to_string(),
                reason: format!("Failed to load tokenizer config: {e}"),
            })?;

        let (bos_token_id, eos_token_id) = Self::get_token_ids(model_path)?;

        Ok(CandleTokenizer::new(
            tokenizer,
            tokenizer_config.chat_template,
            tokenizer_config.eos_token_str,
            tokenizer_config.bos_token_str,
            tokenizer_config.add_bos_token,
            bos_token_id,
            eos_token_id,
        ))
    }

    fn get_token_ids(model_path: &Path) -> Result<(u32, u32), RuntimeError> {
        let config_path = model_path.join("config.json");
        if config_path.exists() {
            if let Ok(config) = TransformerConfig::from_file(&config_path) {
                if let (Some(bos), Some(eos_ids)) =
                    (config.bos_token_id, config.eos_token_id.as_ref())
                {
                    if let Some(eos) = eos_ids.to_vec().first().copied() {
                        return Ok((bos, eos));
                    }
                }
            }
        }

        if let ModelFormat::Gguf(gguf_path) = detect_model_format(model_path) {
            if let Ok(config) = GgufConfig::from_file(&gguf_path) {
                if let (Some(bos), Some(eos)) = (config.bos_token_id, config.eos_token_id) {
                    return Ok((bos, eos));
                }
            }
        }

        Err(RuntimeError::LoadFailed {
            model: model_path.display().to_string(),
            reason: "Could not find BOS/EOS token IDs in model config".to_string(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use rig_core::Runtime;

    #[test]
    fn test_runtime_creation() {
        let runtime =
            CandleRuntime::cpu().unwrap_or_else(|e| panic!("Failed to create runtime: {e}"));

        assert!(!runtime.id().0.is_nil());
        assert!(runtime.capabilities().runtime_type.starts_with("candle"));
        assert!(
            runtime
                .capabilities()
                .supported_dtypes
                .contains(&DType::F32)
        );
    }

    #[test]
    fn test_runtime_capabilities() {
        let runtime =
            CandleRuntime::cpu().unwrap_or_else(|e| panic!("Failed to create runtime: {e}"));
        let caps = runtime.capabilities();

        assert_eq!(caps.runtime_type, "candle_cpu");
        assert!(caps.supported_dtypes.contains(&DType::F32));
        assert!(caps.supported_dtypes.contains(&DType::F16));
        assert!(caps.supported_dtypes.contains(&DType::BF16));
        assert_eq!(caps.vram_bytes, 0);
    }

    #[test]
    fn test_runtime_auto_detect() {
        let runtime =
            CandleRuntime::new().unwrap_or_else(|e| panic!("Failed to create runtime: {e}"));

        assert!(!runtime.id().0.is_nil());
        assert!(runtime.capabilities().runtime_type.starts_with("candle"));
    }
}
