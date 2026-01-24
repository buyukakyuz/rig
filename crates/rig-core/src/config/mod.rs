mod coordinator;
mod generation;
mod logging;
mod runtime;
mod transport;
mod worker;

pub use coordinator::CoordinatorConfigFile;
pub use generation::GenerationConfigFile;
pub use logging::LoggingConfigFile;
pub use runtime::RuntimeConfigFile;
pub use transport::TransportConfigFile;
pub use worker::WorkerConfigFile;

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::ConfigError;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RigConfig {
    pub coordinator: CoordinatorConfigFile,
    pub worker: WorkerConfigFile,
    pub transport: TransportConfigFile,
    pub runtime: RuntimeConfigFile,
    pub logging: LoggingConfigFile,
    pub generation: GenerationConfigFile,
}

impl RigConfig {
    pub fn load(path: Option<&Path>) -> Result<Self, ConfigError> {
        let mut config = if let Some(p) = path {
            Self::from_file(p)?
        } else {
            Self::load_from_default_locations()?
        };

        config.apply_env_overrides();
        config.validate()?;

        Ok(config)
    }

    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ConfigError::NotFound(path.to_path_buf())
            } else {
                ConfigError::ParseError(format!("Failed to read config file: {e}"))
            }
        })?;

        toml::from_str(&content).map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    fn load_from_default_locations() -> Result<Self, ConfigError> {
        let local_config = PathBuf::from("rig.toml");
        if local_config.exists() {
            return Self::from_file(&local_config);
        }

        if let Some(config_dir) = dirs::config_dir() {
            let user_config = config_dir.join("rig").join("config.toml");
            if user_config.exists() {
                return Self::from_file(&user_config);
            }
        }

        Ok(Self::default())
    }

    pub fn apply_env_overrides(&mut self) {
        self.coordinator.apply_env_overrides();
        self.worker.apply_env_overrides();
        self.transport.apply_env_overrides();
        self.runtime.apply_env_overrides();
        self.logging.apply_env_overrides();
        self.generation.apply_env_overrides();
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.coordinator.heartbeat_timeout_secs <= self.coordinator.heartbeat_interval_secs {
            return Err(ConfigError::InvalidValue {
                key: "coordinator.heartbeat_timeout_secs".to_string(),
                reason: format!(
                    "heartbeat_timeout_secs ({}) must be greater than heartbeat_interval_secs ({})",
                    self.coordinator.heartbeat_timeout_secs,
                    self.coordinator.heartbeat_interval_secs
                ),
            });
        }

        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.logging.level.to_lowercase().as_str()) {
            return Err(ConfigError::InvalidValue {
                key: "logging.level".to_string(),
                reason: format!("must be one of: {}", valid_levels.join(", ")),
            });
        }

        let valid_formats = ["pretty", "json", "compact"];
        if !valid_formats.contains(&self.logging.format.to_lowercase().as_str()) {
            return Err(ConfigError::InvalidValue {
                key: "logging.format".to_string(),
                reason: format!("must be one of: {}", valid_formats.join(", ")),
            });
        }

        let valid_devices = ["auto", "cpu", "metal", "cuda"];
        if !valid_devices.contains(&self.runtime.device.to_lowercase().as_str()) {
            return Err(ConfigError::InvalidValue {
                key: "runtime.device".to_string(),
                reason: format!("must be one of: {}", valid_devices.join(", ")),
            });
        }

        let valid_dtypes = ["f16", "bf16", "f32"];
        if !valid_dtypes.contains(&self.runtime.default_dtype.to_lowercase().as_str()) {
            return Err(ConfigError::InvalidValue {
                key: "runtime.default_dtype".to_string(),
                reason: format!("must be one of: {}", valid_dtypes.join(", ")),
            });
        }

        if self.generation.temperature < 0.0 || self.generation.temperature > 2.0 {
            return Err(ConfigError::InvalidValue {
                key: "generation.temperature".to_string(),
                reason: "must be between 0.0 and 2.0".to_string(),
            });
        }

        if self.generation.top_p < 0.0 || self.generation.top_p > 1.0 {
            return Err(ConfigError::InvalidValue {
                key: "generation.top_p".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }

        if self
            .coordinator
            .listen_addr
            .parse::<std::net::SocketAddr>()
            .is_err()
        {
            return Err(ConfigError::InvalidValue {
                key: "coordinator.listen_addr".to_string(),
                reason: format!(
                    "'{}' is not a valid socket address",
                    self.coordinator.listen_addr
                ),
            });
        }

        if self
            .worker
            .coordinator_addr
            .parse::<std::net::SocketAddr>()
            .is_err()
        {
            return Err(ConfigError::InvalidValue {
                key: "worker.coordinator_addr".to_string(),
                reason: format!(
                    "'{}' is not a valid socket address",
                    self.worker.coordinator_addr
                ),
            });
        }

        Ok(())
    }

    pub fn to_toml(&self) -> Result<String, ConfigError> {
        toml::to_string_pretty(self).map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    pub fn generate_default_config() -> String {
        let config = Self::default();
        let toml_str = toml::to_string_pretty(&config).unwrap_or_default();

        format!(
            "# Rig Configuration File\n\
             \n\
             {toml_str}"
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let config = RigConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn serde_roundtrip() {
        let config = RigConfig::default();
        let toml_str = config.to_toml().unwrap();
        let parsed: RigConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(
            parsed.coordinator.listen_addr,
            config.coordinator.listen_addr
        );
        assert_eq!(
            parsed.worker.coordinator_addr,
            config.worker.coordinator_addr
        );
        assert_eq!(parsed.logging.level, config.logging.level);
    }

    #[test]
    fn validation_catches_invalid_heartbeat() {
        let mut config = RigConfig::default();
        config.coordinator.heartbeat_timeout_secs = 5;
        config.coordinator.heartbeat_interval_secs = 10;
        assert!(config.validate().is_err());
    }

    #[test]
    fn validation_catches_invalid_log_level() {
        let mut config = RigConfig::default();
        config.logging.level = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn validation_catches_invalid_device() {
        let mut config = RigConfig::default();
        config.runtime.device = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn validation_catches_invalid_address() {
        let mut config = RigConfig::default();
        config.coordinator.listen_addr = "not-an-address".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn validation_catches_invalid_temperature() {
        let mut config = RigConfig::default();
        config.generation.temperature = 3.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn generate_default_config_produces_valid_toml() {
        let config_str = RigConfig::generate_default_config();
        assert!(config_str.contains("[coordinator]"));
        assert!(config_str.contains("[worker]"));
        assert!(config_str.contains("[logging]"));
        let lines: Vec<&str> = config_str.lines().filter(|l| !l.starts_with('#')).collect();
        let toml_only = lines.join("\n");
        let parsed: Result<RigConfig, _> = toml::from_str(&toml_only);
        assert!(parsed.is_ok());
    }
}
