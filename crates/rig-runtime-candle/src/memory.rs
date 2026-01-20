use candle_core::Device;

use crate::error::CandleError;

pub fn query_device_memory(device: &Device) -> Result<(u64, u64), CandleError> {
    match device {
        Device::Cpu => Ok(query_cpu_memory()),
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => query_cuda_memory(),
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => Err(CandleError::Internal(
            "CUDA support not compiled in".to_string(),
        )),
        #[cfg(feature = "metal")]
        Device::Metal(_) => Ok(query_metal_memory()),
        #[cfg(not(feature = "metal"))]
        Device::Metal(_) => Err(CandleError::Internal(
            "Metal support not compiled in".to_string(),
        )),
    }
}

fn query_cpu_memory() -> (u64, u64) {
    (0, 0)
}

#[cfg(feature = "cuda")]
fn query_cuda_memory() -> Result<(u64, u64), CandleError> {
    let (free, total) = cudarc::driver::result::mem_get_info()
        .map_err(|e| CandleError::Internal(format!("CUDA memory query failed: {e}")))?;
    Ok((free as u64, total as u64))
}

#[cfg(feature = "metal")]
fn query_metal_memory() -> (u64, u64) {
    use sysinfo::System;

    let sys = System::new_all();
    let total_ram = sys.total_memory();

    let available = (total_ram as f64 * 0.75) as u64;
    (available, available)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_memory_query() {
        let (free, total) = query_cpu_memory();
        assert_eq!(free, 0);
        assert_eq!(total, 0);
    }

    #[test]
    fn test_device_memory_cpu() {
        let device = Device::Cpu;
        let result = query_device_memory(&device);
        assert!(result.is_ok());
        let (free, total) = result.unwrap();
        assert_eq!(free, 0);
        assert_eq!(total, 0);
    }
}
