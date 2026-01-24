use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeConfigFile {
    pub device: String,
    pub default_dtype: String,
}

impl Default for RuntimeConfigFile {
    fn default() -> Self {
        Self {
            device: "auto".to_string(),
            default_dtype: "f16".to_string(),
        }
    }
}

impl RuntimeConfigFile {
    pub fn apply_env_overrides(&mut self) {
        if let Ok(val) = std::env::var("RIG_DEVICE") {
            self.device = val;
        }
        if let Ok(val) = std::env::var("RIG_DTYPE") {
            self.default_dtype = val;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let config = RuntimeConfigFile::default();
        assert_eq!(config.device, "auto");
        assert_eq!(config.default_dtype, "f16");
    }

    #[test]
    fn serde_roundtrip() {
        let config = RuntimeConfigFile::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: RuntimeConfigFile = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.device, config.device);
        assert_eq!(parsed.default_dtype, config.default_dtype);
    }
}
