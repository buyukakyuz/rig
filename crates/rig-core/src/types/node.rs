use std::fmt;
use std::net::SocketAddr;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::types::id::{ModelId, NodeId};
use crate::types::tensor::DType;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuntimeType {
    LlamaCpp,
    Candle,
    Onnx,
    TensorRt,
    Custom(String),
}

impl fmt::Display for RuntimeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LlamaCpp => write!(f, "llama_cpp"),
            Self::Candle => write!(f, "candle"),
            Self::Onnx => write!(f, "onnx"),
            Self::TensorRt => write!(f, "tensorrt"),
            Self::Custom(name) => write!(f, "custom:{name}"),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    Cpu,
    Gpu,
    Npu,
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Gpu => write!(f, "gpu"),
            Self::Npu => write!(f, "npu"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeCapability {
    pub tflops_estimate: f32,
    pub device_type: DeviceType,
}

impl ComputeCapability {
    #[must_use]
    pub const fn new(tflops_estimate: f32, device_type: DeviceType) -> Self {
        Self {
            tflops_estimate,
            device_type,
        }
    }

    #[must_use]
    pub const fn cpu(tflops_estimate: f32) -> Self {
        Self::new(tflops_estimate, DeviceType::Cpu)
    }

    #[must_use]
    pub const fn gpu(tflops_estimate: f32) -> Self {
        Self::new(tflops_estimate, DeviceType::Gpu)
    }
}

impl Default for ComputeCapability {
    fn default() -> Self {
        Self::cpu(0.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub runtime_type: RuntimeType,
    pub vram_bytes: u64,
    pub compute_capability: ComputeCapability,
    pub supported_dtypes: Vec<DType>,
}

impl NodeCapabilities {
    #[must_use]
    pub const fn new(
        runtime_type: RuntimeType,
        vram_bytes: u64,
        compute_capability: ComputeCapability,
        supported_dtypes: Vec<DType>,
    ) -> Self {
        Self {
            runtime_type,
            vram_bytes,
            compute_capability,
            supported_dtypes,
        }
    }

    #[must_use]
    pub fn supports_dtype(&self, dtype: DType) -> bool {
        self.supported_dtypes.contains(&dtype)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeCapabilities {
    pub runtime_type: String,
    pub vram_bytes: u64,
    pub supported_dtypes: Vec<DType>,
}

impl RuntimeCapabilities {
    #[must_use]
    pub fn new(
        runtime_type: impl Into<String>,
        vram_bytes: u64,
        supported_dtypes: Vec<DType>,
    ) -> Self {
        Self {
            runtime_type: runtime_type.into(),
            vram_bytes,
            supported_dtypes,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_id: ModelId,
    pub model_path: PathBuf,
    pub num_layers: usize,
    pub hidden_dim: usize,
}

impl ModelInfo {
    #[must_use]
    pub fn new(
        model_id: ModelId,
        model_path: impl Into<PathBuf>,
        num_layers: usize,
        hidden_dim: usize,
    ) -> Self {
        Self {
            model_id,
            model_path: model_path.into(),
            num_layers,
            hidden_dim,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    #[default]
    Healthy,
    Unhealthy {
        reason: String,
    },
    Draining,
    Offline,
}

impl NodeStatus {
    #[must_use]
    pub const fn can_accept_work(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    #[must_use]
    pub const fn is_online(&self) -> bool {
        matches!(self, Self::Healthy | Self::Draining)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Address {
    Tcp(SocketAddr),
    Unix(PathBuf),
}

impl Address {
    #[must_use]
    pub const fn tcp(addr: SocketAddr) -> Self {
        Self::Tcp(addr)
    }

    #[must_use]
    pub fn unix(path: impl Into<PathBuf>) -> Self {
        Self::Unix(path.into())
    }

    #[must_use]
    pub fn tcp_from_host_port(host: &str, port: u16) -> Option<Self> {
        use std::net::IpAddr;
        host.parse::<IpAddr>()
            .ok()
            .map(|ip| Self::Tcp(SocketAddr::new(ip, port)))
    }

    #[must_use]
    pub const fn is_tcp(&self) -> bool {
        matches!(self, Self::Tcp(_))
    }

    #[must_use]
    pub const fn is_unix(&self) -> bool {
        matches!(self, Self::Unix(_))
    }

    #[must_use]
    pub const fn as_tcp(&self) -> Option<&SocketAddr> {
        match self {
            Self::Tcp(addr) => Some(addr),
            Self::Unix(_) => None,
        }
    }

    #[must_use]
    pub const fn as_unix(&self) -> Option<&PathBuf> {
        match self {
            Self::Tcp(_) => None,
            Self::Unix(path) => Some(path),
        }
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tcp(addr) => write!(f, "{addr}"),
            Self::Unix(path) => write!(f, "unix:{}", path.display()),
        }
    }
}

impl From<SocketAddr> for Address {
    fn from(addr: SocketAddr) -> Self {
        Self::Tcp(addr)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: NodeId,
    pub addresses: Vec<Address>,
    pub status: NodeStatus,
    pub capabilities: RuntimeCapabilities,
}

impl NodeInfo {
    #[must_use]
    pub const fn new(
        node_id: NodeId,
        addresses: Vec<Address>,
        status: NodeStatus,
        capabilities: RuntimeCapabilities,
    ) -> Self {
        Self {
            node_id,
            addresses,
            status,
            capabilities,
        }
    }

    #[must_use]
    pub fn first_tcp_address(&self) -> Option<&SocketAddr> {
        self.addresses.iter().find_map(Address::as_tcp)
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn runtime_type_display() {
        assert_eq!(RuntimeType::LlamaCpp.to_string(), "llama_cpp");
        assert_eq!(RuntimeType::Candle.to_string(), "candle");
        assert_eq!(
            RuntimeType::Custom("my-runtime".to_string()).to_string(),
            "custom:my-runtime"
        );
    }

    #[test]
    fn device_type_display() {
        assert_eq!(DeviceType::Cpu.to_string(), "cpu");
        assert_eq!(DeviceType::Gpu.to_string(), "gpu");
        assert_eq!(DeviceType::Npu.to_string(), "npu");
    }

    #[test]
    fn address_tcp() {
        let addr: SocketAddr = "127.0.0.1:8080"
            .parse()
            .unwrap_or_else(|e| panic!("parse failed: {e}"));
        let address = Address::tcp(addr);
        assert!(address.is_tcp());
        assert!(!address.is_unix());
        assert_eq!(address.as_tcp(), Some(&addr));
        assert_eq!(address.to_string(), "127.0.0.1:8080");
    }

    #[test]
    fn address_unix() {
        let address = Address::unix("/var/run/rig.sock");
        assert!(address.is_unix());
        assert!(!address.is_tcp());
        assert_eq!(address.to_string(), "unix:/var/run/rig.sock");
    }

    #[test]
    fn address_tcp_from_host_port() {
        let address = Address::tcp_from_host_port("192.168.1.1", 5000);
        assert!(address.is_some());
        assert!(address.as_ref().is_some_and(Address::is_tcp));

        let invalid = Address::tcp_from_host_port("not-an-ip", 5000);
        assert!(invalid.is_none());
    }

    #[test]
    fn address_serialization_roundtrip() {
        let addr: SocketAddr = "10.0.0.1:9000"
            .parse()
            .unwrap_or_else(|e| panic!("parse failed: {e}"));
        let address = Address::tcp(addr);
        let json =
            serde_json::to_string(&address).unwrap_or_else(|e| panic!("serialize failed: {e}"));
        let recovered: Address =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("deserialize failed: {e}"));
        assert_eq!(address, recovered);
    }

    #[test]
    fn node_status_can_accept_work() {
        assert!(NodeStatus::Healthy.can_accept_work());
        assert!(!NodeStatus::Draining.can_accept_work());
        assert!(!NodeStatus::Offline.can_accept_work());
        assert!(
            !NodeStatus::Unhealthy {
                reason: "test".to_string()
            }
            .can_accept_work()
        );
    }

    #[test]
    fn node_status_is_online() {
        assert!(NodeStatus::Healthy.is_online());
        assert!(NodeStatus::Draining.is_online());
        assert!(!NodeStatus::Offline.is_online());
    }

    #[test]
    fn node_capabilities_supports_dtype() {
        let caps = NodeCapabilities::new(
            RuntimeType::LlamaCpp,
            0,
            ComputeCapability::cpu(10.0),
            vec![DType::F32, DType::F16],
        );
        assert!(caps.supports_dtype(DType::F32));
        assert!(caps.supports_dtype(DType::F16));
        assert!(!caps.supports_dtype(DType::I8));
    }
}
