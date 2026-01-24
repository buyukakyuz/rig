use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LoggingConfigFile {
    pub level: String,
    pub format: String,
}

impl Default for LoggingConfigFile {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "pretty".to_string(),
        }
    }
}

impl LoggingConfigFile {
    pub fn apply_env_overrides(&mut self) {
        if let Ok(val) = std::env::var("RIG_LOG") {
            self.level = val;
        }
        if let Ok(val) = std::env::var("RIG_LOG_FORMAT") {
            self.format = val;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let config = LoggingConfigFile::default();
        assert_eq!(config.level, "info");
        assert_eq!(config.format, "pretty");
    }

    #[test]
    fn serde_roundtrip() {
        let config = LoggingConfigFile::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: LoggingConfigFile = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.level, config.level);
        assert_eq!(parsed.format, config.format);
    }
}
