use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CoordinatorConfigFile {
    pub listen_addr: String,
    pub heartbeat_interval_secs: u64,
    pub heartbeat_timeout_secs: u64,
    pub heartbeat_check_interval_secs: u64,
    pub max_nodes: usize,
    pub idle_timeout_ms: u64,
}

impl Default for CoordinatorConfigFile {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:50051".to_string(),
            heartbeat_interval_secs: 10,
            heartbeat_timeout_secs: 30,
            heartbeat_check_interval_secs: 5,
            max_nodes: 100,
            idle_timeout_ms: 60_000,
        }
    }
}

impl CoordinatorConfigFile {
    pub fn apply_env_overrides(&mut self) {
        if let Ok(val) = std::env::var("RIG_COORDINATOR_LISTEN_ADDR") {
            self.listen_addr = val;
        }
        if let Ok(val) = std::env::var("RIG_COORDINATOR_HEARTBEAT_INTERVAL") {
            if let Ok(v) = val.parse() {
                self.heartbeat_interval_secs = v;
            }
        }
        if let Ok(val) = std::env::var("RIG_COORDINATOR_HEARTBEAT_TIMEOUT") {
            if let Ok(v) = val.parse() {
                self.heartbeat_timeout_secs = v;
            }
        }
        if let Ok(val) = std::env::var("RIG_COORDINATOR_HEARTBEAT_CHECK_INTERVAL") {
            if let Ok(v) = val.parse() {
                self.heartbeat_check_interval_secs = v;
            }
        }
        if let Ok(val) = std::env::var("RIG_COORDINATOR_MAX_NODES") {
            if let Ok(v) = val.parse() {
                self.max_nodes = v;
            }
        }
        if let Ok(val) = std::env::var("RIG_COORDINATOR_IDLE_TIMEOUT") {
            if let Ok(v) = val.parse() {
                self.idle_timeout_ms = v;
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
        let config = CoordinatorConfigFile::default();
        assert_eq!(config.listen_addr, "0.0.0.0:50051");
        assert_eq!(config.heartbeat_interval_secs, 10);
        assert_eq!(config.heartbeat_timeout_secs, 30);
        assert_eq!(config.heartbeat_check_interval_secs, 5);
        assert_eq!(config.max_nodes, 100);
        assert_eq!(config.idle_timeout_ms, 60_000);
    }

    #[test]
    fn serde_roundtrip() {
        let config = CoordinatorConfigFile::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: CoordinatorConfigFile = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.listen_addr, config.listen_addr);
        assert_eq!(
            parsed.heartbeat_interval_secs,
            config.heartbeat_interval_secs
        );
    }
}
