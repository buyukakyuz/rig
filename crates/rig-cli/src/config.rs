use std::path::{Path, PathBuf};

use rig_core::{
    ConfigError, CoordinatorConfigFile, GenerationConfigFile, LoggingConfigFile, RigConfig,
    RuntimeConfigFile, TransportConfigFile, WorkerConfigFile,
};

pub trait RigConfigLoader: Sized {
    /// Load configuration from the given path, or from default locations if `None`.
    fn load(path: Option<&Path>) -> Result<Self, ConfigError>;

    /// Load configuration from a specific file path.
    fn from_file(path: &Path) -> Result<Self, ConfigError>;

    /// Validate the configuration values.
    fn validate(&self) -> Result<(), ConfigError>;

    /// Serialize the configuration to a TOML string.
    fn to_toml(&self) -> Result<String, ConfigError>;

    /// Generate a default configuration file with comments.
    fn generate_default_config() -> String;
}

pub trait EnvOverrides {
    fn apply_env_overrides(&mut self);
}

impl RigConfigLoader for RigConfig {
    fn load(path: Option<&Path>) -> Result<Self, ConfigError> {
        let mut config = if let Some(p) = path {
            Self::from_file(p)?
        } else {
            load_from_default_locations()?
        };

        config.apply_env_overrides();
        config.validate()?;

        Ok(config)
    }

    fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ConfigError::NotFound(path.to_path_buf())
            } else {
                ConfigError::ParseError(format!("Failed to read config file: {e}"))
            }
        })?;

        toml::from_str(&content).map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    fn validate(&self) -> Result<(), ConfigError> {
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

    fn to_toml(&self) -> Result<String, ConfigError> {
        toml::to_string_pretty(self).map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    fn generate_default_config() -> String {
        let config = Self::default();
        let toml_str = toml::to_string_pretty(&config).unwrap_or_default();

        format!(
            "# Rig Configuration File\n\
             \n\
             {toml_str}"
        )
    }
}

fn load_from_default_locations() -> Result<RigConfig, ConfigError> {
    let local_config = PathBuf::from("rig.toml");
    if local_config.exists() {
        return RigConfig::from_file(&local_config);
    }

    if let Some(config_dir) = dirs::config_dir() {
        let user_config = config_dir.join("rig").join("config.toml");
        if user_config.exists() {
            return RigConfig::from_file(&user_config);
        }
    }

    Ok(RigConfig::default())
}

impl EnvOverrides for RigConfig {
    fn apply_env_overrides(&mut self) {
        self.coordinator.apply_env_overrides();
        self.worker.apply_env_overrides();
        self.transport.apply_env_overrides();
        self.runtime.apply_env_overrides();
        self.logging.apply_env_overrides();
        self.generation.apply_env_overrides();
    }
}

impl EnvOverrides for CoordinatorConfigFile {
    fn apply_env_overrides(&mut self) {
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

impl EnvOverrides for WorkerConfigFile {
    fn apply_env_overrides(&mut self) {
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

impl EnvOverrides for TransportConfigFile {
    fn apply_env_overrides(&mut self) {
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

impl EnvOverrides for RuntimeConfigFile {
    fn apply_env_overrides(&mut self) {
        if let Ok(val) = std::env::var("RIG_DEVICE") {
            self.device = val;
        }
        if let Ok(val) = std::env::var("RIG_DTYPE") {
            self.default_dtype = val;
        }
    }
}

impl EnvOverrides for LoggingConfigFile {
    fn apply_env_overrides(&mut self) {
        if let Ok(val) = std::env::var("RIG_LOG") {
            self.level = val;
        }
        if let Ok(val) = std::env::var("RIG_LOG_FORMAT") {
            self.format = val;
        }
    }
}

impl EnvOverrides for GenerationConfigFile {
    fn apply_env_overrides(&mut self) {
        if let Ok(val) = std::env::var("RIG_GENERATION_MAX_TOKENS") {
            if let Ok(v) = val.parse() {
                self.max_tokens = v;
            }
        }
        if let Ok(val) = std::env::var("RIG_GENERATION_TEMPERATURE") {
            if let Ok(v) = val.parse() {
                self.temperature = v;
            }
        }
        if let Ok(val) = std::env::var("RIG_GENERATION_TOP_P") {
            if let Ok(v) = val.parse() {
                self.top_p = v;
            }
        }
        if let Ok(val) = std::env::var("RIG_GENERATION_TOP_K") {
            if let Ok(v) = val.parse() {
                self.top_k = v;
            }
        }
        if let Ok(val) = std::env::var("RIG_GENERATION_TIMEOUT") {
            if let Ok(v) = val.parse() {
                self.timeout_secs = v;
            }
        }
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

    #[test]
    fn coordinator_default_values() {
        let config = CoordinatorConfigFile::default();
        assert_eq!(config.listen_addr, "0.0.0.0:50051");
        assert_eq!(config.heartbeat_interval_secs, 10);
        assert_eq!(config.heartbeat_timeout_secs, 30);
        assert_eq!(config.heartbeat_check_interval_secs, 5);
        assert_eq!(config.max_nodes, 100);
        assert_eq!(config.idle_timeout_ms, 60_000);
    }

    #[test]
    fn coordinator_serde_roundtrip() {
        let config = CoordinatorConfigFile::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: CoordinatorConfigFile = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.listen_addr, config.listen_addr);
        assert_eq!(
            parsed.heartbeat_interval_secs,
            config.heartbeat_interval_secs
        );
    }

    #[test]
    fn worker_default_values() {
        let config = WorkerConfigFile::default();
        assert_eq!(config.coordinator_addr, "127.0.0.1:50051");
        assert_eq!(config.listen_addr, "0.0.0.0:0");
        assert_eq!(config.heartbeat_interval_secs, 10);
        assert!(config.enable_warmup);
    }

    #[test]
    fn worker_serde_roundtrip() {
        let config = WorkerConfigFile::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: WorkerConfigFile = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.coordinator_addr, config.coordinator_addr);
        assert_eq!(parsed.listen_addr, config.listen_addr);
    }

    #[test]
    fn logging_default_values() {
        let config = LoggingConfigFile::default();
        assert_eq!(config.level, "info");
        assert_eq!(config.format, "pretty");
    }

    #[test]
    fn logging_serde_roundtrip() {
        let config = LoggingConfigFile::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: LoggingConfigFile = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.level, config.level);
        assert_eq!(parsed.format, config.format);
    }

    #[test]
    fn runtime_default_values() {
        let config = RuntimeConfigFile::default();
        assert_eq!(config.device, "auto");
        assert_eq!(config.default_dtype, "f16");
    }

    #[test]
    fn runtime_serde_roundtrip() {
        let config = RuntimeConfigFile::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: RuntimeConfigFile = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.device, config.device);
        assert_eq!(parsed.default_dtype, config.default_dtype);
    }

    #[test]
    fn generation_default_values() {
        let config = GenerationConfigFile::default();
        assert_eq!(config.max_tokens, 256);
        assert!((config.temperature - 0.7).abs() < f32::EPSILON);
        assert!((config.top_p - 0.9).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 40);
        assert_eq!(config.timeout_secs, 60);
    }

    #[test]
    fn generation_serde_roundtrip() {
        let config = GenerationConfigFile::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: GenerationConfigFile = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.max_tokens, config.max_tokens);
        assert!((parsed.temperature - config.temperature).abs() < f32::EPSILON);
    }

    #[test]
    fn transport_default_values() {
        let config = TransportConfigFile::default();
        assert_eq!(config.max_message_size, 64 * 1024 * 1024);
        assert_eq!(config.connect_timeout_ms, 10_000);
        assert_eq!(config.read_timeout_ms, 30_000);
        assert_eq!(config.write_timeout_ms, 30_000);
        assert!(config.nodelay);
    }

    #[test]
    fn transport_serde_roundtrip() {
        let config = TransportConfigFile::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: TransportConfigFile = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.max_message_size, config.max_message_size);
        assert_eq!(parsed.connect_timeout_ms, config.connect_timeout_ms);
    }
}
