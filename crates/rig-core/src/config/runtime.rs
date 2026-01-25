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
