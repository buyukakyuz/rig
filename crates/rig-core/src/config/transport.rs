use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TransportConfigFile {
    pub max_message_size: usize,
    pub connect_timeout_ms: u64,
    pub read_timeout_ms: u64,
    pub write_timeout_ms: u64,
    pub nodelay: bool,
}

impl Default for TransportConfigFile {
    fn default() -> Self {
        Self {
            max_message_size: 64 * 1024 * 1024,
            connect_timeout_ms: 10_000,
            read_timeout_ms: 30_000,
            write_timeout_ms: 30_000,
            nodelay: true,
        }
    }
}
