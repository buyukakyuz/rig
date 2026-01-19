use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use rig_core::{Address, ModelId};

#[derive(Debug, Clone)]
pub enum RuntimeConfig {
    Candle(CandleConfig),
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::Candle(CandleConfig::default())
    }
}

#[derive(Debug, Clone, Default)]
pub struct CandleConfig {
    pub device: String,
}

impl CandleConfig {
    #[must_use]
    pub fn new() -> Self {
        Self {
            device: "auto".to_string(),
        }
    }

    #[must_use]
    pub fn cpu() -> Self {
        Self {
            device: "cpu".to_string(),
        }
    }

    #[must_use]
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = device.into();
        self
    }
}

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub coordinator_addr: Address,
    pub listen_addr: SocketAddr,
    pub heartbeat_interval: Duration,
    pub model_paths: HashMap<ModelId, PathBuf>,
    pub runtime_config: RuntimeConfig,
}

impl WorkerConfig {
    #[must_use]
    pub fn new(coordinator_addr: Address) -> Self {
        Self {
            coordinator_addr,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn with_coordinator_addr(mut self, addr: Address) -> Self {
        self.coordinator_addr = addr;
        self
    }

    #[must_use]
    pub const fn with_listen_addr(mut self, addr: SocketAddr) -> Self {
        self.listen_addr = addr;
        self
    }

    #[must_use]
    pub const fn with_heartbeat_interval(mut self, interval: Duration) -> Self {
        self.heartbeat_interval = interval;
        self
    }

    #[must_use]
    pub fn with_model_path(mut self, model_id: ModelId, path: impl Into<PathBuf>) -> Self {
        self.model_paths.insert(model_id, path.into());
        self
    }

    #[must_use]
    pub fn with_model_paths(mut self, paths: HashMap<ModelId, PathBuf>) -> Self {
        self.model_paths = paths;
        self
    }

    #[must_use]
    pub fn with_runtime_config(mut self, config: RuntimeConfig) -> Self {
        self.runtime_config = config;
        self
    }

    #[must_use]
    pub fn get_model_path(&self, model_id: &ModelId) -> Option<&PathBuf> {
        self.model_paths.get(model_id)
    }
}

impl Default for WorkerConfig {
    fn default() -> Self {
        let coordinator_addr = Address::tcp(SocketAddr::from(([127, 0, 0, 1], 50051)));

        let listen_addr = SocketAddr::from(([0, 0, 0, 0], 0));

        Self {
            coordinator_addr,
            listen_addr,
            heartbeat_interval: Duration::from_secs(10),
            model_paths: HashMap::new(),
            runtime_config: RuntimeConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = WorkerConfig::default();
        assert_eq!(config.heartbeat_interval, Duration::from_secs(10));
        assert!(config.model_paths.is_empty());
    }

    #[test]
    fn builder_pattern() {
        let coord_addr = Address::tcp(SocketAddr::from(([192, 168, 1, 100], 50051)));
        let listen_addr = SocketAddr::from(([0, 0, 0, 0], 5000));

        let config = WorkerConfig::new(coord_addr.clone())
            .with_listen_addr(listen_addr)
            .with_heartbeat_interval(Duration::from_secs(5))
            .with_model_path(ModelId::new("llama-7b", "q4"), "/models/llama-7b-q4.gguf");

        assert_eq!(config.coordinator_addr, coord_addr);
        assert_eq!(config.listen_addr, listen_addr);
        assert_eq!(config.heartbeat_interval, Duration::from_secs(5));
        assert!(
            config
                .get_model_path(&ModelId::new("llama-7b", "q4"))
                .is_some()
        );
    }

    #[test]
    fn model_path_lookup() {
        let model_id = ModelId::new("test-model", "v1");
        let path = PathBuf::from("/models/test.gguf");

        let config = WorkerConfig::default().with_model_path(model_id.clone(), path.clone());

        assert_eq!(config.get_model_path(&model_id), Some(&path));
        assert_eq!(config.get_model_path(&ModelId::new("other", "v1")), None);
    }
}
