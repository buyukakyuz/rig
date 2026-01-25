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

use serde::{Deserialize, Serialize};

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
