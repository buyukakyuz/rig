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
