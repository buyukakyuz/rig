use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkerConfigFile {
    pub coordinator_addr: String,
    pub listen_addr: String,
    pub heartbeat_interval_secs: u64,
    pub enable_warmup: bool,
}

impl Default for WorkerConfigFile {
    fn default() -> Self {
        Self {
            coordinator_addr: "127.0.0.1:50051".to_string(),
            listen_addr: "0.0.0.0:0".to_string(),
            heartbeat_interval_secs: 10,
            enable_warmup: true,
        }
    }
}

impl WorkerConfigFile {
    pub fn apply_env_overrides(&mut self) {
        if let Ok(val) = std::env::var("RIG_WORKER_COORDINATOR_ADDR") {
            self.coordinator_addr = val;
        }
        if let Ok(val) = std::env::var("RIG_COORDINATOR_ADDR") {
            self.coordinator_addr = val;
        }
        if let Ok(val) = std::env::var("RIG_WORKER_LISTEN_ADDR") {
            self.listen_addr = val;
        }
        if let Ok(val) = std::env::var("RIG_WORKER_HEARTBEAT_INTERVAL") {
            if let Ok(v) = val.parse() {
                self.heartbeat_interval_secs = v;
            }
        }
        if let Ok(val) = std::env::var("RIG_HEARTBEAT_INTERVAL") {
            if let Ok(v) = val.parse() {
                self.heartbeat_interval_secs = v;
            }
        }
        if let Ok(val) = std::env::var("RIG_WORKER_ENABLE_WARMUP") {
            if let Ok(v) = val.parse() {
                self.enable_warmup = v;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let config = WorkerConfigFile::default();
        assert_eq!(config.coordinator_addr, "127.0.0.1:50051");
        assert_eq!(config.listen_addr, "0.0.0.0:0");
        assert_eq!(config.heartbeat_interval_secs, 10);
        assert!(config.enable_warmup);
    }

    #[test]
    fn serde_roundtrip() {
        let config = WorkerConfigFile::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: WorkerConfigFile = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.coordinator_addr, config.coordinator_addr);
        assert_eq!(parsed.listen_addr, config.listen_addr);
    }
}
