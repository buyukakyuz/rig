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

impl TransportConfigFile {
    pub fn apply_env_overrides(&mut self) {
        if let Ok(val) = std::env::var("RIG_TRANSPORT_MAX_MESSAGE_SIZE") {
            if let Ok(v) = val.parse() {
                self.max_message_size = v;
            }
        }
        if let Ok(val) = std::env::var("RIG_TRANSPORT_CONNECT_TIMEOUT") {
            if let Ok(v) = val.parse() {
                self.connect_timeout_ms = v;
            }
        }
        if let Ok(val) = std::env::var("RIG_TRANSPORT_READ_TIMEOUT") {
            if let Ok(v) = val.parse() {
                self.read_timeout_ms = v;
            }
        }
        if let Ok(val) = std::env::var("RIG_TRANSPORT_WRITE_TIMEOUT") {
            if let Ok(v) = val.parse() {
                self.write_timeout_ms = v;
            }
        }
        if let Ok(val) = std::env::var("RIG_TRANSPORT_NODELAY") {
            if let Ok(v) = val.parse() {
                self.nodelay = v;
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
        let config = TransportConfigFile::default();
        assert_eq!(config.max_message_size, 64 * 1024 * 1024);
        assert_eq!(config.connect_timeout_ms, 10_000);
        assert_eq!(config.read_timeout_ms, 30_000);
        assert_eq!(config.write_timeout_ms, 30_000);
        assert!(config.nodelay);
    }

    #[test]
    fn serde_roundtrip() {
        let config = TransportConfigFile::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: TransportConfigFile = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.max_message_size, config.max_message_size);
        assert_eq!(parsed.connect_timeout_ms, config.connect_timeout_ms);
    }
}
