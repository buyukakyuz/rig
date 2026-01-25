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
