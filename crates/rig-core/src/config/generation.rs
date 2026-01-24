use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GenerationConfigFile {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub timeout_secs: u64,
}

impl Default for GenerationConfigFile {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            timeout_secs: 60,
        }
    }
}

impl GenerationConfigFile {
    pub fn apply_env_overrides(&mut self) {
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
    fn default_values() {
        let config = GenerationConfigFile::default();
        assert_eq!(config.max_tokens, 256);
        assert!((config.temperature - 0.7).abs() < f32::EPSILON);
        assert!((config.top_p - 0.9).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 40);
        assert_eq!(config.timeout_secs, 60);
    }

    #[test]
    fn serde_roundtrip() {
        let config = GenerationConfigFile::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: GenerationConfigFile = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.max_tokens, config.max_tokens);
        assert!((parsed.temperature - config.temperature).abs() < f32::EPSILON);
    }
}
