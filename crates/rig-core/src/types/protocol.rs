use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::types::id::{ModelId, NodeId, PipelineId, RequestId};
use crate::types::node::{ModelInfo, NodeInfo, NodeStatus};
use crate::types::pipeline::{Assignment, PipelineConfig};
use crate::types::request::{GenerationParams, InferenceInput, InferenceRequest, UsageStats};
use crate::types::tensor::DType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerMessage {
    Register(RegisterRequest),
    Heartbeat(HeartbeatRequest),
    GetAssignment,
    ReportReady {
        pipeline_id: PipelineId,
    },
    Deregister,
    GetPendingRequest {
        pipeline_id: PipelineId,
    },
    ReturnLogits {
        request_id: RequestId,
        logits: Vec<f32>,
        eos_token: u32,
    },
    TokenGenerated {
        request_id: RequestId,
        token_text: String,
    },
    StreamingComplete {
        request_id: RequestId,
        usage: UsageStats,
    },
    GetGenerationControl {
        request_id: RequestId,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinatorMessage {
    RegisterResponse(RegisterResponse),
    HeartbeatAck,
    Assignment(Option<Assignment>),
    ReadyAck,
    ResultAck,
    PendingRequest(Option<InferenceRequest>),
    Error {
        code: u32,
        message: String,
    },
    ContinueGeneration {
        request_id: RequestId,
        token: u32,
        position: u32,
    },
    FinishGeneration {
        request_id: RequestId,
        generated_tokens: Vec<u32>,
        time_to_first_token_ms: u64,
    },
    GenerationPending {
        request_id: RequestId,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterRequest {
    pub node_info: NodeInfo,
    #[serde(default)]
    pub available_models: Vec<ModelInfo>,
}

impl RegisterRequest {
    #[must_use]
    pub const fn new(node_info: NodeInfo) -> Self {
        Self {
            node_info,
            available_models: Vec::new(),
        }
    }

    #[must_use]
    pub const fn with_models(node_info: NodeInfo, available_models: Vec<ModelInfo>) -> Self {
        Self {
            node_info,
            available_models,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterResponse {
    pub node_id: NodeId,
    pub accepted: bool,
    pub reason: Option<String>,
}

impl RegisterResponse {
    #[must_use]
    pub const fn accepted(node_id: NodeId) -> Self {
        Self {
            node_id,
            accepted: true,
            reason: None,
        }
    }

    #[must_use]
    pub fn rejected(reason: impl Into<String>) -> Self {
        Self {
            node_id: NodeId::nil(),
            accepted: false,
            reason: Some(reason.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    pub node_id: NodeId,
    pub status: NodeStatus,
    pub metrics: Option<NodeMetrics>,
}

impl HeartbeatRequest {
    #[must_use]
    pub const fn new(node_id: NodeId, status: NodeStatus) -> Self {
        Self {
            node_id,
            status,
            metrics: None,
        }
    }

    #[must_use]
    pub const fn with_metrics(node_id: NodeId, status: NodeStatus, metrics: NodeMetrics) -> Self {
        Self {
            node_id,
            status,
            metrics: Some(metrics),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeMetrics {
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub active_requests: usize,
}

impl NodeMetrics {
    #[must_use]
    pub const fn new(
        memory_used_bytes: u64,
        memory_total_bytes: u64,
        active_requests: usize,
    ) -> Self {
        Self {
            memory_used_bytes,
            memory_total_bytes,
            active_requests,
        }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn memory_usage_percent(&self) -> f64 {
        if self.memory_total_bytes == 0 {
            0.0
        } else {
            (self.memory_used_bytes as f64) / (self.memory_total_bytes as f64)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinatorIncoming {
    Worker(WorkerMessage),
    Cli(CliMessage),
}

impl From<WorkerMessage> for CoordinatorIncoming {
    fn from(msg: WorkerMessage) -> Self {
        Self::Worker(msg)
    }
}

impl From<CliMessage> for CoordinatorIncoming {
    fn from(msg: CliMessage) -> Self {
        Self::Cli(msg)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinatorOutgoing {
    Worker(CoordinatorMessage),
    Cli(CliResponse),
}

impl From<CoordinatorMessage> for CoordinatorOutgoing {
    fn from(msg: CoordinatorMessage) -> Self {
        Self::Worker(msg)
    }
}

impl From<CliResponse> for CoordinatorOutgoing {
    fn from(msg: CliResponse) -> Self {
        Self::Cli(msg)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CliMessage {
    GetStatus,
    SubmitRequest(CliSubmitRequest),
    CreatePipeline(CliCreatePipelineRequest),
    CreatePipelineAuto(CliCreatePipelineAutoRequest),
    GetPipeline { pipeline_id: PipelineId },
    ListPipelines,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliSubmitRequest {
    pub pipeline_id: PipelineId,
    pub input: InferenceInput,
    pub params: GenerationParams,
    pub timeout_ms: Option<u64>,
}

impl CliSubmitRequest {
    #[must_use]
    pub fn new(pipeline_id: PipelineId, input: InferenceInput) -> Self {
        Self {
            pipeline_id,
            input,
            params: GenerationParams::default(),
            timeout_ms: None,
        }
    }

    #[must_use]
    pub fn with_params(mut self, params: GenerationParams) -> Self {
        self.params = params;
        self
    }

    #[must_use]
    pub const fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliCreatePipelineRequest {
    pub config: PipelineConfig,
    pub assignments: Vec<(NodeId, usize, usize)>,
    pub pipeline_id: Option<PipelineId>,
}

impl CliCreatePipelineRequest {
    #[must_use]
    pub const fn new(config: PipelineConfig, assignments: Vec<(NodeId, usize, usize)>) -> Self {
        Self {
            config,
            assignments,
            pipeline_id: None,
        }
    }

    #[must_use]
    pub const fn with_pipeline_id(mut self, pipeline_id: PipelineId) -> Self {
        self.pipeline_id = Some(pipeline_id);
        self
    }

    #[must_use]
    pub fn assignments_as_ranges(&self) -> Vec<(NodeId, Range<usize>)> {
        self.assignments
            .iter()
            .map(|(node_id, start, end)| (*node_id, *start..(*end + 1)))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliCreatePipelineAutoRequest {
    pub model_name: String,
    #[serde(default = "default_model_version")]
    pub model_version: String,
    pub dtype: DType,
    pub num_stages: Option<usize>,
    pub pipeline_id: Option<PipelineId>,
}

fn default_model_version() -> String {
    "v1".to_string()
}

impl CliCreatePipelineAutoRequest {
    #[must_use]
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            model_version: default_model_version(),
            dtype: DType::F16,
            num_stages: None,
            pipeline_id: None,
        }
    }

    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.model_version = version.into();
        self
    }

    #[must_use]
    pub const fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    #[must_use]
    pub const fn with_stages(mut self, num_stages: usize) -> Self {
        self.num_stages = Some(num_stages);
        self
    }

    #[must_use]
    pub const fn with_pipeline_id(mut self, pipeline_id: PipelineId) -> Self {
        self.pipeline_id = Some(pipeline_id);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CliResponse {
    Status(ClusterStatusResponse),
    PipelineCreated {
        pipeline_id: PipelineId,
    },
    PipelineInfo(PipelineInfoResponse),
    PipelineList(Vec<PipelineInfoResponse>),
    Error {
        code: u32,
        message: String,
    },
    StreamStart {
        request_id: RequestId,
    },
    Token {
        request_id: RequestId,
        token_text: String,
    },
    StreamComplete {
        request_id: RequestId,
        usage: UsageStats,
    },
}

impl CliResponse {
    #[must_use]
    pub fn error(code: u32, message: impl Into<String>) -> Self {
        Self::Error {
            code,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStatusResponse {
    pub total_nodes: usize,
    pub healthy_nodes: usize,
    pub total_pipelines: usize,
    pub ready_pipelines: usize,
    pub nodes: Vec<NodeStatusInfo>,
    pub pipelines: Vec<PipelineInfoResponse>,
}

impl ClusterStatusResponse {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            total_nodes: 0,
            healthy_nodes: 0,
            total_pipelines: 0,
            ready_pipelines: 0,
            nodes: Vec::new(),
            pipelines: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatusInfo {
    pub node_id: NodeId,
    pub addresses: Vec<String>,
    pub status: NodeStatus,
    pub runtime_type: Option<String>,
    pub vram_bytes: Option<u64>,
}

impl NodeStatusInfo {
    #[must_use]
    pub fn from_node_info(info: &NodeInfo, status: NodeStatus) -> Self {
        Self {
            node_id: info.node_id,
            addresses: info.addresses.iter().map(ToString::to_string).collect(),
            status,
            runtime_type: Some(info.capabilities.runtime_type.clone()),
            vram_bytes: Some(info.capabilities.vram_bytes),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineInfoResponse {
    pub pipeline_id: PipelineId,
    pub model_id: ModelId,
    pub status: String,
    pub stages: Vec<StageInfoResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageInfoResponse {
    pub stage_id: u32,
    pub node_id: NodeId,
    pub layer_start: usize,
    pub layer_end: usize,
    pub ready: bool,
}

pub mod error_codes {
    pub const NOT_REGISTERED: u32 = 1000;
    pub const MAX_NODES_REACHED: u32 = 1001;
    pub const NODE_NOT_FOUND: u32 = 1002;
    pub const PIPELINE_NOT_FOUND: u32 = 1003;
    pub const INVALID_REQUEST: u32 = 1004;
    pub const INTERNAL_ERROR: u32 = 5000;
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use crate::types::node::{Address, RuntimeCapabilities};

    fn test_node_info() -> NodeInfo {
        let addr: std::net::SocketAddr = "127.0.0.1:5000"
            .parse()
            .unwrap_or_else(|e| panic!("parse failed: {e}"));
        NodeInfo::new(
            NodeId::new(),
            vec![Address::tcp(addr)],
            NodeStatus::Healthy,
            RuntimeCapabilities::new("llama_cpp", 0, vec![]),
        )
    }

    #[test]
    fn register_request_creation() {
        let info = test_node_info();
        let req = RegisterRequest::new(info.clone());
        assert_eq!(req.node_info.node_id, info.node_id);
    }

    #[test]
    fn register_response_accepted() {
        let node_id = NodeId::new();
        let resp = RegisterResponse::accepted(node_id);
        assert!(resp.accepted);
        assert_eq!(resp.node_id, node_id);
        assert!(resp.reason.is_none());
    }

    #[test]
    fn register_response_rejected() {
        let resp = RegisterResponse::rejected("cluster full");
        assert!(!resp.accepted);
        assert_eq!(resp.reason.as_deref(), Some("cluster full"));
    }

    #[test]
    fn heartbeat_request_creation() {
        let node_id = NodeId::new();
        let req = HeartbeatRequest::new(node_id, NodeStatus::Healthy);
        assert_eq!(req.node_id, node_id);
        assert_eq!(req.status, NodeStatus::Healthy);
        assert!(req.metrics.is_none());
    }

    #[test]
    fn heartbeat_request_with_metrics() {
        let node_id = NodeId::new();
        let metrics = NodeMetrics::new(1024, 2048, 5);
        let req = HeartbeatRequest::with_metrics(node_id, NodeStatus::Healthy, metrics);
        assert!(req.metrics.is_some());
        let m = req
            .metrics
            .as_ref()
            .unwrap_or_else(|| panic!("expected metrics"));
        assert_eq!(m.memory_used_bytes, 1024);
        assert_eq!(m.memory_total_bytes, 2048);
        assert_eq!(m.active_requests, 5);
    }

    #[test]
    fn node_metrics_usage_percent() {
        let metrics = NodeMetrics::new(512, 1024, 0);
        assert!((metrics.memory_usage_percent() - 0.5).abs() < f64::EPSILON);

        let empty = NodeMetrics::new(0, 0, 0);
        assert!((empty.memory_usage_percent() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn worker_message_serialization_roundtrip() {
        let info = test_node_info();
        let msg = WorkerMessage::Register(RegisterRequest::new(info));
        let json = serde_json::to_string(&msg).unwrap_or_else(|e| panic!("serialize failed: {e}"));
        let recovered: WorkerMessage =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("deserialize failed: {e}"));
        match recovered {
            WorkerMessage::Register(_) => {}
            _ => panic!("expected Register variant"),
        }
    }

    #[test]
    fn coordinator_message_serialization_roundtrip() {
        let node_id = NodeId::new();
        let msg = CoordinatorMessage::RegisterResponse(RegisterResponse::accepted(node_id));
        let json = serde_json::to_string(&msg).unwrap_or_else(|e| panic!("serialize failed: {e}"));
        let recovered: CoordinatorMessage =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("deserialize failed: {e}"));
        match recovered {
            CoordinatorMessage::RegisterResponse(r) => {
                assert!(r.accepted);
                assert_eq!(r.node_id, node_id);
            }
            _ => panic!("expected RegisterResponse variant"),
        }
    }
}
