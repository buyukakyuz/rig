use serde::{Deserialize, Serialize};

use crate::types::id::{ModelId, NodeId, PipelineId, RequestId, TenantId};
use crate::types::node::{NodeInfo, NodeStatus};

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum Priority {
    Low = 0,
    #[default]
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Priority {
    #[must_use]
    pub const fn all() -> [Self; 4] {
        [Self::Low, Self::Normal, Self::High, Self::Critical]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceInput {
    Tokens(Vec<u32>),
    Text(String),
}

impl InferenceInput {
    #[must_use]
    pub const fn tokens(tokens: Vec<u32>) -> Self {
        Self::Tokens(tokens)
    }

    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    #[must_use]
    pub const fn is_tokens(&self) -> bool {
        matches!(self, Self::Tokens(_))
    }

    #[must_use]
    pub const fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }

    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub stop_sequences: Vec<String>,
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub use_chat_template: bool,
}

impl GenerationParams {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    #[must_use]
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    #[must_use]
    pub const fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    #[must_use]
    pub fn with_stop_sequence(mut self, stop: impl Into<String>) -> Self {
        self.stop_sequences.push(stop.into());
        self
    }

    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    #[must_use]
    pub const fn with_chat_template(mut self, use_chat_template: bool) -> Self {
        self.use_chat_template = use_chat_template;
        self
    }
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            stop_sequences: Vec::new(),
            system_prompt: None,
            use_chat_template: false,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestContext {
    pub tenant_id: Option<TenantId>,
    pub priority: Priority,
    pub deadline_unix_ms: Option<u64>,
}

impl RequestContext {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_tenant(mut self, tenant_id: TenantId) -> Self {
        self.tenant_id = Some(tenant_id);
        self
    }

    #[must_use]
    pub const fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    #[must_use]
    pub const fn with_deadline_unix_ms(mut self, deadline: u64) -> Self {
        self.deadline_unix_ms = Some(deadline);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub request_id: RequestId,
    pub model_id: ModelId,
    pub input: InferenceInput,
    pub params: GenerationParams,
    pub context: RequestContext,
}

impl InferenceRequest {
    #[must_use]
    pub fn new(model_id: ModelId, input: InferenceInput) -> Self {
        Self {
            request_id: RequestId::new(),
            model_id,
            input,
            params: GenerationParams::default(),
            context: RequestContext::default(),
        }
    }

    #[must_use]
    pub fn text(model_id: ModelId, text: impl Into<String>) -> Self {
        Self::new(model_id, InferenceInput::text(text))
    }

    #[must_use]
    pub fn tokens(model_id: ModelId, tokens: Vec<u32>) -> Self {
        Self::new(model_id, InferenceInput::tokens(tokens))
    }

    #[must_use]
    pub fn with_params(mut self, params: GenerationParams) -> Self {
        self.params = params;
        self
    }

    #[must_use]
    pub fn with_context(mut self, context: RequestContext) -> Self {
        self.context = context;
        self
    }

    #[must_use]
    pub const fn with_priority(mut self, priority: Priority) -> Self {
        self.context.priority = priority;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceOutput {
    Tokens(Vec<u32>),
    Text(String),
    Error(String),
}

impl InferenceOutput {
    #[must_use]
    pub const fn tokens(tokens: Vec<u32>) -> Self {
        Self::Tokens(tokens)
    }

    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error(message.into())
    }

    #[must_use]
    pub const fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    #[must_use]
    pub const fn is_success(&self) -> bool {
        !self.is_error()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_time_ms: u64,
    pub time_to_first_token_ms: u64,
}

impl UsageStats {
    #[must_use]
    pub const fn new(
        prompt_tokens: usize,
        completion_tokens: usize,
        total_time_ms: u64,
        time_to_first_token_ms: u64,
    ) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_time_ms,
            time_to_first_token_ms,
        }
    }

    #[must_use]
    pub const fn total_tokens(&self) -> usize {
        self.prompt_tokens + self.completion_tokens
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn tokens_per_second(&self) -> f64 {
        if self.total_time_ms == 0 {
            0.0
        } else {
            (self.completion_tokens as f64) / (self.total_time_ms as f64 / 1000.0)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub request_id: RequestId,
    pub output: InferenceOutput,
    pub usage: UsageStats,
}

impl InferenceResult {
    #[must_use]
    pub const fn new(request_id: RequestId, output: InferenceOutput, usage: UsageStats) -> Self {
        Self {
            request_id,
            output,
            usage,
        }
    }

    #[must_use]
    pub fn success_text(request_id: RequestId, text: impl Into<String>, usage: UsageStats) -> Self {
        Self::new(request_id, InferenceOutput::text(text), usage)
    }

    #[must_use]
    pub fn error(request_id: RequestId, message: impl Into<String>) -> Self {
        Self::new(
            request_id,
            InferenceOutput::error(message),
            UsageStats::default(),
        )
    }

    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.output.is_success()
    }
}

#[derive(Debug, Clone)]
pub struct Credentials {
    pub token: String,
}

impl Credentials {
    #[must_use]
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    pub id: String,
    pub name: String,
}

impl Identity {
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub action_type: String,
    pub resource: String,
}

impl Action {
    #[must_use]
    pub fn new(action_type: impl Into<String>, resource: impl Into<String>) -> Self {
        Self {
            action_type: action_type.into(),
            resource: resource.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterState {
    pub nodes: Vec<NodeInfo>,
    pub pipelines: Vec<PipelineId>,
}

impl ClusterState {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            pipelines: Vec::new(),
        }
    }

    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[must_use]
    pub fn find_node(&self, node_id: &NodeId) -> Option<&NodeInfo> {
        self.nodes.iter().find(|n| &n.node_id == node_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusterEvent {
    NodeJoined(NodeInfo),
    NodeLeft(NodeId),
    NodeStatusChanged { node_id: NodeId, status: NodeStatus },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JoinResult {
    Success { node_id: NodeId },
    Rejected { reason: String },
}

impl JoinResult {
    #[must_use]
    pub const fn success(node_id: NodeId) -> Self {
        Self::Success { node_id }
    }

    #[must_use]
    pub fn rejected(reason: impl Into<String>) -> Self {
        Self::Rejected {
            reason: reason.into(),
        }
    }

    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    #[must_use]
    pub const fn node_id(&self) -> Option<NodeId> {
        match self {
            Self::Success { node_id } => Some(*node_id),
            Self::Rejected { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkResult {
    pub request_id: RequestId,
    pub success: bool,
    pub error: Option<String>,
}

impl WorkResult {
    #[must_use]
    pub const fn success(request_id: RequestId) -> Self {
        Self {
            request_id,
            success: true,
            error: None,
        }
    }

    #[must_use]
    pub fn failure(request_id: RequestId, error: impl Into<String>) -> Self {
        Self {
            request_id,
            success: false,
            error: Some(error.into()),
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn priority_ordering() {
        assert!(Priority::Low < Priority::Normal);
        assert!(Priority::Normal < Priority::High);
        assert!(Priority::High < Priority::Critical);
    }

    #[test]
    fn generation_params_builder() {
        let params = GenerationParams::new()
            .with_max_tokens(100)
            .with_temperature(0.5)
            .with_top_p(0.95)
            .with_stop_sequence("\n");

        assert_eq!(params.max_tokens, 100);
        assert!((params.temperature - 0.5).abs() < f32::EPSILON);
        assert!((params.top_p - 0.95).abs() < f32::EPSILON);
        assert_eq!(params.stop_sequences, vec!["\n"]);
    }

    #[test]
    fn generation_params_defaults() {
        let params = GenerationParams::default();
        assert_eq!(params.max_tokens, 256);
        assert!((params.temperature - 0.7).abs() < f32::EPSILON);
        assert!(params.stop_sequences.is_empty());
    }

    #[test]
    fn inference_request_builder() {
        let request = InferenceRequest::text(ModelId::new("test", "v1"), "Hello")
            .with_priority(Priority::High);

        assert!(request.input.is_text());
        assert_eq!(request.context.priority, Priority::High);
    }

    #[test]
    fn inference_output_variants() {
        let text = InferenceOutput::text("hello");
        assert!(text.is_success());
        assert!(!text.is_error());

        let error = InferenceOutput::error("oops");
        assert!(error.is_error());
        assert!(!error.is_success());
    }

    #[test]
    fn usage_stats_calculations() {
        let stats = UsageStats::new(100, 50, 1000, 100);
        assert_eq!(stats.total_tokens(), 150);
        assert!((stats.tokens_per_second() - 50.0).abs() < 0.01);
    }

    #[test]
    fn usage_stats_zero_time() {
        let stats = UsageStats::new(0, 0, 0, 0);
        assert!((stats.tokens_per_second() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn inference_result_variants() {
        let success =
            InferenceResult::success_text(RequestId::new(), "output", UsageStats::default());
        assert!(success.is_success());

        let error = InferenceResult::error(RequestId::new(), "failed");
        assert!(!error.is_success());
    }

    #[test]
    fn join_result_accessors() {
        let node_id = NodeId::new();
        let success = JoinResult::success(node_id);
        assert!(success.is_success());
        assert_eq!(success.node_id(), Some(node_id));

        let rejected = JoinResult::rejected("cluster full");
        assert!(!rejected.is_success());
        assert_eq!(rejected.node_id(), None);
    }

    #[test]
    fn work_result_variants() {
        let success = WorkResult::success(RequestId::new());
        assert!(success.success);
        assert!(success.error.is_none());

        let failure = WorkResult::failure(RequestId::new(), "timeout");
        assert!(!failure.success);
        assert_eq!(failure.error.as_deref(), Some("timeout"));
    }

    #[test]
    fn inference_request_serialization_roundtrip() {
        let request = InferenceRequest::text(ModelId::new("model", "v1"), "test input");
        let json =
            serde_json::to_string(&request).unwrap_or_else(|e| panic!("serialize failed: {e}"));
        let recovered: InferenceRequest =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("deserialize failed: {e}"));
        assert_eq!(request.request_id, recovered.request_id);
    }
}
