use anyhow::Result;
use rig_core::Runtime;
use rig_runtime_candle::CandleRuntime;

pub fn create_runtime(device: &str) -> Result<impl Runtime + 'static> {
    tracing::info!(device = %device, "Creating Candle runtime");
    match device {
        "cpu" => CandleRuntime::cpu(),
        _ => CandleRuntime::new(),
    }
    .map_err(Into::into)
}
