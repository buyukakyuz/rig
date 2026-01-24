use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::Range;
use std::time::Instant;

use rig_core::{
    Address, Assignment, CoordError, GenerationDecision, GenerationParams, InferenceRequest,
    ModelId, ModelInfo, Neighbors, NodeId, NodeInfo, NodeStatus, PeerAddress, PipelineConfig,
    PipelineId, RequestId, StageId, UsageStats,
};
use tokio::sync::{RwLock, mpsc, oneshot};
use tracing::warn;

use crate::config::CoordinatorConfig;
use crate::generation::{GenerationSession, GenerationStatus};

#[derive(Debug)]
pub struct NodeRecord {
    pub info: NodeInfo,
    pub last_heartbeat: Instant,
    pub status: NodeStatus,
    pub available_models: Vec<ModelInfo>,
}

impl NodeRecord {
    fn new(info: NodeInfo, available_models: Vec<ModelInfo>) -> Self {
        Self {
            status: info.status.clone(),
            info,
            last_heartbeat: Instant::now(),
            available_models,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PipelineStatus {
    #[default]
    Creating,
    Ready,
    Error,
}

#[derive(Debug)]
pub struct StageRecord {
    pub stage_id: StageId,
    pub node_id: NodeId,
    pub layer_range: Range<usize>,
    pub ready: bool,
}

#[derive(Debug)]
pub struct PipelineRecord {
    pub config: PipelineConfig,
    pub stages: Vec<StageRecord>,
    pub status: PipelineStatus,
}

pub struct StreamingSession {
    pub token_tx: mpsc::UnboundedSender<String>,
    pub complete_tx: oneshot::Sender<UsageStats>,
}

impl std::fmt::Debug for StreamingSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingSession")
            .field("token_tx", &"...")
            .field("complete_tx", &"...")
            .finish()
    }
}

pub struct CoordinatorState {
    max_nodes: usize,
    nodes: RwLock<HashMap<NodeId, NodeRecord>>,
    pipelines: RwLock<HashMap<PipelineId, PipelineRecord>>,
    assignments: RwLock<HashMap<NodeId, Assignment>>,
    pending_requests: RwLock<HashMap<PipelineId, VecDeque<InferenceRequest>>>,
    streaming_sessions: RwLock<HashMap<RequestId, StreamingSession>>,
    model_registry: RwLock<HashMap<ModelId, (usize, usize)>>,
    generation_decisions: RwLock<HashMap<RequestId, GenerationDecision>>,
    active_multi_stage_requests: RwLock<HashSet<RequestId>>,
    generation_sessions: RwLock<HashMap<RequestId, GenerationSession>>,
}

impl std::fmt::Debug for CoordinatorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoordinatorState")
            .field("max_nodes", &self.max_nodes)
            .field("nodes", &"...")
            .field("pipelines", &"...")
            .field("assignments", &"...")
            .field("pending_requests", &"...")
            .field("streaming_sessions", &"...")
            .field("model_registry", &"...")
            .field("generation_decisions", &"...")
            .field("active_multi_stage_requests", &"...")
            .field("generation_sessions", &"...")
            .finish()
    }
}

impl CoordinatorState {
    #[must_use]
    pub fn new(config: &CoordinatorConfig) -> Self {
        Self {
            max_nodes: config.max_nodes,
            nodes: RwLock::new(HashMap::new()),
            pipelines: RwLock::new(HashMap::new()),
            assignments: RwLock::new(HashMap::new()),
            pending_requests: RwLock::new(HashMap::new()),
            streaming_sessions: RwLock::new(HashMap::new()),
            model_registry: RwLock::new(HashMap::new()),
            generation_decisions: RwLock::new(HashMap::new()),
            active_multi_stage_requests: RwLock::new(HashSet::new()),
            generation_sessions: RwLock::new(HashMap::new()),
        }
    }

    pub async fn register_node(
        &self,
        info: NodeInfo,
        available_models: Vec<ModelInfo>,
    ) -> Result<NodeId, CoordError> {
        let node_id = info.node_id;

        {
            let mut registry = self.model_registry.write().await;

            for model in &available_models {
                if let Some(&(existing_layers, existing_hidden)) = registry.get(&model.model_id) {
                    if model.num_layers != existing_layers {
                        return Err(CoordError::InvalidRequest(format!(
                            "Model {} layer count mismatch: node reports {} layers, \
                             but {} layers already registered",
                            model.model_id, model.num_layers, existing_layers
                        )));
                    }
                    if model.hidden_dim != existing_hidden {
                        return Err(CoordError::InvalidRequest(format!(
                            "Model {} hidden dim mismatch: node reports {}, \
                             but {} already registered",
                            model.model_id, model.hidden_dim, existing_hidden
                        )));
                    }
                } else {
                    registry.insert(model.model_id.clone(), (model.num_layers, model.hidden_dim));
                    tracing::debug!(
                        model_id = %model.model_id,
                        num_layers = model.num_layers,
                        hidden_dim = model.hidden_dim,
                        "Model registered in global registry"
                    );
                }
            }
            drop(registry);
        }

        {
            let mut nodes = self.nodes.write().await;

            if nodes.len() >= self.max_nodes {
                return Err(CoordError::MaxNodesReached {
                    max: self.max_nodes,
                });
            }

            let record = NodeRecord::new(info, available_models.clone());
            nodes.insert(node_id, record);
        }

        tracing::debug!(
            %node_id,
            num_models = available_models.len(),
            "Node registered"
        );
        Ok(node_id)
    }

    pub async fn get_nodes_with_model(&self, model_id: &ModelId) -> Vec<(NodeId, ModelInfo)> {
        let nodes = self.nodes.read().await;

        nodes
            .iter()
            .filter_map(|(node_id, record)| {
                if !record.status.can_accept_work() {
                    return None;
                }
                record
                    .available_models
                    .iter()
                    .find(|m: &&ModelInfo| &m.model_id == model_id)
                    .map(|m| (*node_id, m.clone()))
            })
            .collect()
    }

    pub async fn get_model_info(&self, model_id: &ModelId) -> Option<(usize, usize)> {
        let registry = self.model_registry.read().await;
        registry.get(model_id).copied()
    }

    #[allow(clippy::significant_drop_tightening)]
    pub async fn heartbeat(&self, node_id: NodeId, status: NodeStatus) -> Result<(), CoordError> {
        {
            let mut nodes = self.nodes.write().await;

            let record = nodes
                .get_mut(&node_id)
                .ok_or(CoordError::NodeNotFound(node_id))?;

            record.last_heartbeat = Instant::now();
            record.status = status;
        }

        tracing::trace!(%node_id, "Heartbeat received");
        Ok(())
    }

    pub async fn get_assignment(&self, node_id: NodeId) -> Result<Option<Assignment>, CoordError> {
        {
            let nodes = self.nodes.read().await;
            if !nodes.contains_key(&node_id) {
                return Err(CoordError::NodeNotFound(node_id));
            }
        }

        let assignments = self.assignments.read().await;
        Ok(assignments.get(&node_id).cloned())
    }

    #[allow(clippy::significant_drop_tightening)]
    pub async fn create_pipeline(
        &self,
        config: PipelineConfig,
        stage_assignments: Vec<(NodeId, Range<usize>)>,
        pipeline_id: Option<PipelineId>,
    ) -> Result<PipelineId, CoordError> {
        let node_addresses: HashMap<NodeId, Vec<Address>> = {
            let nodes = self.nodes.read().await;
            let mut addresses = HashMap::new();

            for (node_id, _) in &stage_assignments {
                let record = nodes
                    .get(node_id)
                    .ok_or(CoordError::NodeNotFound(*node_id))?;
                addresses.insert(*node_id, record.info.addresses.clone());
            }

            addresses
        };

        let pipeline_id = pipeline_id.unwrap_or_default();
        let num_stages = stage_assignments.len();

        if num_stages > u32::MAX as usize {
            return Err(CoordError::InvalidRequest(format!(
                "Too many stages: {} exceeds maximum of {}",
                num_stages,
                u32::MAX
            )));
        }

        #[allow(clippy::cast_possible_truncation)]
        let stages: Vec<StageRecord> = stage_assignments
            .iter()
            .enumerate()
            .map(|(idx, (node_id, layer_range))| {
                let stage_idx = idx as u32;
                StageRecord {
                    stage_id: StageId::new(stage_idx),
                    node_id: *node_id,
                    layer_range: layer_range.clone(),
                    ready: false,
                }
            })
            .collect();

        {
            let mut assignments = self.assignments.write().await;

            #[allow(clippy::cast_possible_truncation)]
            for (idx, (node_id, layer_range)) in stage_assignments.iter().enumerate() {
                let prev = if idx > 0 {
                    let prev_node_id = stage_assignments[idx - 1].0;
                    let addresses = node_addresses
                        .get(&prev_node_id)
                        .ok_or(CoordError::NodeNotFound(prev_node_id))?
                        .clone();
                    Some(PeerAddress::new(prev_node_id, addresses))
                } else {
                    None
                };

                let next = if idx < num_stages - 1 {
                    let next_node_id = stage_assignments[idx + 1].0;
                    let addresses = node_addresses
                        .get(&next_node_id)
                        .ok_or(CoordError::NodeNotFound(next_node_id))?
                        .clone();
                    Some(PeerAddress::new(next_node_id, addresses))
                } else {
                    None
                };

                let neighbors = Neighbors { prev, next };

                let stage_idx = idx as u32;
                let assignment = Assignment::new(
                    pipeline_id,
                    StageId::new(stage_idx),
                    layer_range.clone(),
                    neighbors,
                );

                assignments.insert(*node_id, assignment);
            }
        }

        {
            let mut pipelines = self.pipelines.write().await;
            pipelines.insert(
                pipeline_id,
                PipelineRecord {
                    config,
                    stages,
                    status: PipelineStatus::Creating,
                },
            );
        }

        tracing::debug!(%pipeline_id, num_stages, "Pipeline created");
        Ok(pipeline_id)
    }

    #[allow(clippy::significant_drop_tightening)]
    pub async fn mark_ready(
        &self,
        node_id: NodeId,
        pipeline_id: PipelineId,
    ) -> Result<(), CoordError> {
        {
            let nodes = self.nodes.read().await;
            if !nodes.contains_key(&node_id) {
                return Err(CoordError::NodeNotFound(node_id));
            }
        }

        let mut pipelines = self.pipelines.write().await;
        let pipeline = pipelines
            .get_mut(&pipeline_id)
            .ok_or(CoordError::PipelineNotFound(pipeline_id))?;

        for stage in &mut pipeline.stages {
            if stage.node_id == node_id {
                stage.ready = true;
                tracing::debug!(%node_id, %pipeline_id, "Node marked ready");
                break;
            }
        }

        if pipeline.stages.iter().all(|s| s.ready) {
            pipeline.status = PipelineStatus::Ready;
            tracing::debug!(%pipeline_id, "Pipeline ready");
        }

        Ok(())
    }

    pub async fn find_dead_nodes(&self, now: Instant, timeout: std::time::Duration) -> Vec<NodeId> {
        let nodes = self.nodes.read().await;

        nodes
            .iter()
            .filter(|(_, record)| {
                record.status.is_online() && now.duration_since(record.last_heartbeat) > timeout
            })
            .map(|(id, _)| *id)
            .collect()
    }

    pub async fn mark_unhealthy(&self, node_id: NodeId, reason: &str) {
        let mut nodes = self.nodes.write().await;

        if let Some(record) = nodes.get_mut(&node_id) {
            record.status = NodeStatus::Unhealthy {
                reason: reason.to_string(),
            };
            tracing::warn!(%node_id, reason, "Node marked unhealthy");
        }
    }

    pub async fn deregister_node(&self, node_id: NodeId) {
        {
            let mut nodes = self.nodes.write().await;
            nodes.remove(&node_id);
        }

        {
            let mut assignments = self.assignments.write().await;
            assignments.remove(&node_id);
        }

        tracing::debug!(%node_id, "Node deregistered");
    }

    pub async fn is_registered(&self, node_id: NodeId) -> bool {
        let nodes = self.nodes.read().await;
        nodes.contains_key(&node_id)
    }

    #[allow(clippy::significant_drop_tightening)]
    pub async fn is_pipeline_ready(&self, pipeline_id: PipelineId) -> Result<bool, CoordError> {
        let pipelines = self.pipelines.read().await;
        let pipeline = pipelines
            .get(&pipeline_id)
            .ok_or(CoordError::PipelineNotFound(pipeline_id))?;
        Ok(pipeline.status == PipelineStatus::Ready)
    }

    #[allow(clippy::significant_drop_tightening)]
    pub async fn get_pipeline_first_stage(
        &self,
        pipeline_id: PipelineId,
    ) -> Result<NodeId, CoordError> {
        let pipelines = self.pipelines.read().await;
        let pipeline = pipelines
            .get(&pipeline_id)
            .ok_or(CoordError::PipelineNotFound(pipeline_id))?;
        pipeline
            .stages
            .first()
            .map(|stage| stage.node_id)
            .ok_or(CoordError::NoAssignment)
    }

    #[allow(clippy::significant_drop_tightening)]
    pub async fn get_pipeline_last_stage(
        &self,
        pipeline_id: PipelineId,
    ) -> Result<NodeId, CoordError> {
        let pipelines = self.pipelines.read().await;
        let pipeline = pipelines
            .get(&pipeline_id)
            .ok_or(CoordError::PipelineNotFound(pipeline_id))?;
        pipeline
            .stages
            .last()
            .map(|stage| stage.node_id)
            .ok_or(CoordError::NoAssignment)
    }

    #[allow(clippy::significant_drop_tightening)]
    pub async fn get_pipeline_status(
        &self,
        pipeline_id: PipelineId,
    ) -> Result<PipelineStatus, CoordError> {
        let pipelines = self.pipelines.read().await;
        let pipeline = pipelines
            .get(&pipeline_id)
            .ok_or(CoordError::PipelineNotFound(pipeline_id))?;
        Ok(pipeline.status)
    }

    pub async fn submit_request(
        &self,
        pipeline_id: PipelineId,
        request: InferenceRequest,
    ) -> Result<(), CoordError> {
        if !self.is_pipeline_ready(pipeline_id).await? {
            return Err(CoordError::InvalidRequest(
                "Pipeline is not ready".to_string(),
            ));
        }
        let request_id = request.request_id;
        {
            let mut pending = self.pending_requests.write().await;
            pending.entry(pipeline_id).or_default().push_back(request);
        }

        tracing::debug!(%request_id, %pipeline_id, "Request queued for pipeline");
        Ok(())
    }

    pub async fn get_pending_request(&self, pipeline_id: PipelineId) -> Option<InferenceRequest> {
        let mut pending = self.pending_requests.write().await;
        pending.get_mut(&pipeline_id).and_then(|queue| {
            let request = queue.pop_front();
            if let Some(ref req) = request {
                tracing::debug!(request_id = %req.request_id, "Dequeued request for first stage");
            }
            request
        })
    }

    pub async fn pending_request_count(&self, pipeline_id: PipelineId) -> usize {
        let pending = self.pending_requests.read().await;
        pending.get(&pipeline_id).map_or(0, VecDeque::len)
    }

    pub async fn start_streaming_session(
        &self,
        request_id: RequestId,
        token_tx: mpsc::UnboundedSender<String>,
        complete_tx: oneshot::Sender<UsageStats>,
    ) {
        let session = StreamingSession {
            token_tx,
            complete_tx,
        };
        self.streaming_sessions
            .write()
            .await
            .insert(request_id, session);
        tracing::debug!(%request_id, "Streaming session started");
    }

    #[allow(clippy::significant_drop_tightening)]
    pub async fn forward_token(&self, request_id: RequestId, token_text: String) -> bool {
        let sessions = self.streaming_sessions.read().await;
        if let Some(session) = sessions.get(&request_id) {
            if session.token_tx.send(token_text).is_ok() {
                return true;
            }
            warn!(%request_id, "Failed to forward token (receiver dropped)");
        }
        false
    }

    pub async fn complete_streaming_session(
        &self,
        request_id: RequestId,
        usage: UsageStats,
    ) -> bool {
        let session = {
            let mut sessions = self.streaming_sessions.write().await;
            sessions.remove(&request_id)
        };

        if let Some(session) = session {
            drop(session.token_tx);
            if session.complete_tx.send(usage).is_ok() {
                tracing::debug!(%request_id, "Streaming session completed");
                return true;
            }
            warn!(%request_id, "Failed to send streaming completion (receiver dropped)");
        }
        false
    }

    pub async fn store_generation_decision(
        &self,
        request_id: RequestId,
        decision: GenerationDecision,
    ) {
        self.generation_decisions
            .write()
            .await
            .insert(request_id, decision);
        tracing::trace!(%request_id, "Generation decision stored");
    }

    pub async fn take_generation_decision(
        &self,
        request_id: RequestId,
    ) -> Option<GenerationDecision> {
        let mut decisions = self.generation_decisions.write().await;
        decisions.remove(&request_id)
    }

    #[allow(clippy::significant_drop_tightening)]
    pub async fn register_multi_stage_request(
        &self,
        request_id: RequestId,
    ) -> Result<(), CoordError> {
        let mut active = self.active_multi_stage_requests.write().await;

        if active.contains(&request_id) {
            return Err(CoordError::InvalidRequest(format!(
                "Request {request_id} is already registered as active"
            )));
        }

        active.insert(request_id);
        tracing::debug!(%request_id, "Registered multi-stage request");
        Ok(())
    }

    #[allow(clippy::significant_drop_tightening)]
    pub async fn unregister_multi_stage_request(&self, request_id: RequestId) -> bool {
        let mut active = self.active_multi_stage_requests.write().await;
        let removed = active.remove(&request_id);
        if removed {
            tracing::debug!(%request_id, "Unregistered multi-stage request");
        }
        removed
    }

    pub async fn is_multi_stage_active(&self, request_id: RequestId) -> bool {
        let active = self.active_multi_stage_requests.read().await;
        active.contains(&request_id)
    }

    pub async fn start_generation_session(
        &self,
        request_id: RequestId,
        params: &GenerationParams,
        eos_token: u32,
        prompt_tokens: usize,
        seed: u64,
    ) {
        let session = GenerationSession::new(request_id, params, eos_token, prompt_tokens, seed);
        self.generation_sessions
            .write()
            .await
            .insert(request_id, session);
        tracing::debug!(%request_id, prompt_tokens, eos_token, "Generation session started");
    }

    #[allow(clippy::significant_drop_tightening)]
    pub async fn on_token_sampled(
        &self,
        request_id: RequestId,
        token: u32,
    ) -> Result<GenerationStatus, CoordError> {
        let mut sessions = self.generation_sessions.write().await;
        let session = sessions.get_mut(&request_id).ok_or_else(|| {
            CoordError::InvalidRequest(format!(
                "No generation session found for request {request_id}"
            ))
        })?;

        let status = session.on_token(token);
        tracing::trace!(
            %request_id,
            token,
            generated_count = session.generated_tokens().len(),
            "Token sampled"
        );
        Ok(status)
    }

    pub async fn get_generation_session(
        &self,
        request_id: RequestId,
    ) -> Result<tokio::sync::RwLockReadGuard<'_, HashMap<RequestId, GenerationSession>>, CoordError>
    {
        let sessions = self.generation_sessions.read().await;
        if !sessions.contains_key(&request_id) {
            return Err(CoordError::InvalidRequest(format!(
                "No generation session found for request {request_id}"
            )));
        }
        Ok(sessions)
    }

    #[allow(clippy::significant_drop_tightening)]
    pub async fn get_session_finish_data(
        &self,
        request_id: RequestId,
    ) -> Result<(Vec<u32>, u64), CoordError> {
        let sessions = self.generation_sessions.read().await;
        let session = sessions.get(&request_id).ok_or_else(|| {
            CoordError::InvalidRequest(format!(
                "No generation session found for request {request_id}"
            ))
        })?;
        Ok((
            session.generated_tokens().to_vec(),
            session.time_to_first_token_ms(),
        ))
    }

    pub async fn end_generation_session(&self, request_id: RequestId) -> bool {
        let removed = self
            .generation_sessions
            .write()
            .await
            .remove(&request_id)
            .is_some();
        if removed {
            tracing::debug!(%request_id, "Generation session ended");
        }
        removed
    }

    pub async fn has_generation_session(&self, request_id: RequestId) -> bool {
        self.generation_sessions
            .read()
            .await
            .contains_key(&request_id)
    }

    pub async fn nodes(&self) -> tokio::sync::RwLockReadGuard<'_, HashMap<NodeId, NodeRecord>> {
        self.nodes.read().await
    }

    pub async fn node_count(&self) -> usize {
        let nodes = self.nodes.read().await;
        nodes.len()
    }

    pub async fn pipeline_count(&self) -> usize {
        let pipelines = self.pipelines.read().await;
        pipelines.len()
    }

    pub async fn get_node_info(&self, node_id: NodeId) -> Option<NodeInfo> {
        let nodes = self.nodes.read().await;
        nodes.get(&node_id).map(|r| r.info.clone())
    }

    pub async fn build_cluster_status(&self) -> rig_core::ClusterStatusResponse {
        let nodes = self.nodes.read().await;
        let pipelines = self.pipelines.read().await;

        let healthy_nodes = nodes.values().filter(|r| r.status.is_online()).count();
        let ready_pipelines = pipelines
            .values()
            .filter(|p| p.status == PipelineStatus::Ready)
            .count();

        let node_infos: Vec<rig_core::NodeStatusInfo> = nodes
            .values()
            .map(|record| {
                rig_core::NodeStatusInfo::from_node_info(&record.info, record.status.clone())
            })
            .collect();

        let pipeline_infos: Vec<rig_core::PipelineInfoResponse> = pipelines
            .iter()
            .map(|(pipeline_id, record)| Self::build_pipeline_info_internal(*pipeline_id, record))
            .collect();

        rig_core::ClusterStatusResponse {
            total_nodes: nodes.len(),
            healthy_nodes,
            total_pipelines: pipelines.len(),
            ready_pipelines,
            nodes: node_infos,
            pipelines: pipeline_infos,
        }
    }

    #[allow(clippy::significant_drop_tightening)]
    pub async fn get_pipeline_info(
        &self,
        pipeline_id: PipelineId,
    ) -> Result<rig_core::PipelineInfoResponse, CoordError> {
        let pipelines = self.pipelines.read().await;
        let record = pipelines
            .get(&pipeline_id)
            .ok_or(CoordError::PipelineNotFound(pipeline_id))?;

        Ok(Self::build_pipeline_info_internal(pipeline_id, record))
    }

    pub async fn list_pipelines(&self) -> Vec<rig_core::PipelineInfoResponse> {
        let pipelines = self.pipelines.read().await;
        pipelines
            .iter()
            .map(|(pipeline_id, record)| Self::build_pipeline_info_internal(*pipeline_id, record))
            .collect()
    }

    fn build_pipeline_info_internal(
        pipeline_id: PipelineId,
        record: &PipelineRecord,
    ) -> rig_core::PipelineInfoResponse {
        let status = match record.status {
            PipelineStatus::Creating => "creating".to_string(),
            PipelineStatus::Ready => "ready".to_string(),
            PipelineStatus::Error => "error".to_string(),
        };

        let stages: Vec<rig_core::StageInfoResponse> = record
            .stages
            .iter()
            .map(|stage| rig_core::StageInfoResponse {
                stage_id: stage.stage_id.0,
                node_id: stage.node_id,
                layer_start: stage.layer_range.start,
                layer_end: stage.layer_range.end,
                ready: stage.ready,
            })
            .collect();

        rig_core::PipelineInfoResponse {
            pipeline_id,
            model_id: record.config.model_id.clone(),
            status,
            stages,
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use rig_core::{DType, ModelId, RuntimeCapabilities};
    use std::net::SocketAddr;

    fn test_config() -> CoordinatorConfig {
        CoordinatorConfig::default().with_max_nodes(10)
    }

    fn test_node_info(node_id: NodeId, port: u16) -> NodeInfo {
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        NodeInfo::new(
            node_id,
            vec![Address::tcp(addr)],
            NodeStatus::Healthy,
            RuntimeCapabilities::new("candle", 0, vec![]),
        )
    }

    #[tokio::test]
    async fn test_register_node() {
        let state = CoordinatorState::new(&test_config());
        let node_id = NodeId::new();
        let info = test_node_info(node_id, 5000);

        let result = state.register_node(info, Vec::new()).await;
        assert!(result.is_ok());
        assert_eq!(result.ok(), Some(node_id));
        assert!(state.is_registered(node_id).await);
    }

    #[tokio::test]
    async fn test_max_nodes_reached() {
        let config = CoordinatorConfig::default().with_max_nodes(2);
        let state = CoordinatorState::new(&config);

        for i in 0..2 {
            let node_id = NodeId::new();
            let info = test_node_info(node_id, 5000 + i);
            let result = state.register_node(info, Vec::new()).await;
            assert!(result.is_ok());
        }

        let node_id = NodeId::new();
        let info = test_node_info(node_id, 5010);
        let result = state.register_node(info, Vec::new()).await;
        assert!(matches!(
            result,
            Err(CoordError::MaxNodesReached { max: 2 })
        ));
    }

    #[tokio::test]
    async fn test_heartbeat() {
        let state = CoordinatorState::new(&test_config());
        let node_id = NodeId::new();
        let info = test_node_info(node_id, 5000);

        state.register_node(info, Vec::new()).await.ok();

        let result = state.heartbeat(node_id, NodeStatus::Healthy).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_heartbeat_unregistered() {
        let state = CoordinatorState::new(&test_config());
        let node_id = NodeId::new();

        let result = state.heartbeat(node_id, NodeStatus::Healthy).await;
        assert!(matches!(result, Err(CoordError::NodeNotFound(_))));
    }

    #[tokio::test]
    async fn test_create_pipeline() {
        let state = CoordinatorState::new(&test_config());

        let node1 = NodeId::new();
        let node2 = NodeId::new();
        state
            .register_node(test_node_info(node1, 5000), Vec::new())
            .await
            .ok();
        state
            .register_node(test_node_info(node2, 5001), Vec::new())
            .await
            .ok();

        let config = PipelineConfig::new(ModelId::new("test", "v1"), "/models/test", 2, DType::F16);

        let result = state
            .create_pipeline(config, vec![(node1, 0..10), (node2, 10..20)], None)
            .await;
        assert!(result.is_ok());

        let assignment1 = state.get_assignment(node1).await.ok().flatten();
        assert!(assignment1.is_some());
        let a1 = assignment1.as_ref().unwrap_or_else(|| {
            panic!("assignment1 should be some");
        });
        assert!(a1.neighbors.prev.is_none());
        assert!(a1.neighbors.next.is_some());

        let assignment2 = state.get_assignment(node2).await.ok().flatten();
        assert!(assignment2.is_some());
        let a2 = assignment2.as_ref().unwrap_or_else(|| {
            panic!("assignment2 should be some");
        });
        assert!(a2.neighbors.prev.is_some());
        assert!(a2.neighbors.next.is_none());
    }

    #[tokio::test]
    async fn test_deregister_node() {
        let state = CoordinatorState::new(&test_config());
        let node_id = NodeId::new();
        let info = test_node_info(node_id, 5000);

        state.register_node(info, Vec::new()).await.ok();
        assert!(state.is_registered(node_id).await);

        state.deregister_node(node_id).await;
        assert!(!state.is_registered(node_id).await);
    }

    #[tokio::test]
    async fn test_find_dead_nodes() {
        let state = CoordinatorState::new(&test_config());
        let node_id = NodeId::new();
        let info = test_node_info(node_id, 5000);

        state.register_node(info, Vec::new()).await.ok();

        let dead = state
            .find_dead_nodes(Instant::now(), std::time::Duration::from_secs(0))
            .await;
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0], node_id);

        let dead = state
            .find_dead_nodes(Instant::now(), std::time::Duration::from_secs(3600))
            .await;
        assert!(dead.is_empty());
    }

    #[tokio::test]
    async fn test_is_pipeline_ready() {
        let state = CoordinatorState::new(&test_config());

        let node1 = NodeId::new();
        let node2 = NodeId::new();
        state
            .register_node(test_node_info(node1, 5000), Vec::new())
            .await
            .ok();
        state
            .register_node(test_node_info(node2, 5001), Vec::new())
            .await
            .ok();

        let config = PipelineConfig::new(ModelId::new("test", "v1"), "/models/test", 2, DType::F16);
        let pipeline_id = state
            .create_pipeline(config, vec![(node1, 0..10), (node2, 10..20)], None)
            .await
            .expect("pipeline creation should succeed");

        assert!(
            !state
                .is_pipeline_ready(pipeline_id)
                .await
                .expect("should succeed")
        );

        state.mark_ready(node1, pipeline_id).await.ok();
        assert!(
            !state
                .is_pipeline_ready(pipeline_id)
                .await
                .expect("should succeed")
        );

        state.mark_ready(node2, pipeline_id).await.ok();
        assert!(
            state
                .is_pipeline_ready(pipeline_id)
                .await
                .expect("should succeed")
        );
    }

    #[tokio::test]
    async fn test_is_pipeline_ready_not_found() {
        let state = CoordinatorState::new(&test_config());
        let fake_pipeline_id = rig_core::PipelineId::new();

        let result = state.is_pipeline_ready(fake_pipeline_id).await;
        assert!(matches!(result, Err(CoordError::PipelineNotFound(_))));
    }

    #[tokio::test]
    async fn test_get_pipeline_first_stage() {
        let state = CoordinatorState::new(&test_config());

        let node1 = NodeId::new();
        let node2 = NodeId::new();
        state
            .register_node(test_node_info(node1, 5000), Vec::new())
            .await
            .ok();
        state
            .register_node(test_node_info(node2, 5001), Vec::new())
            .await
            .ok();

        let config = PipelineConfig::new(ModelId::new("test", "v1"), "/models/test", 2, DType::F16);
        let pipeline_id = state
            .create_pipeline(config, vec![(node1, 0..10), (node2, 10..20)], None)
            .await
            .expect("pipeline creation should succeed");

        let first_stage = state
            .get_pipeline_first_stage(pipeline_id)
            .await
            .expect("should succeed");
        assert_eq!(first_stage, node1);
    }

    #[tokio::test]
    async fn test_get_pipeline_last_stage() {
        let state = CoordinatorState::new(&test_config());

        let node1 = NodeId::new();
        let node2 = NodeId::new();
        state
            .register_node(test_node_info(node1, 5000), Vec::new())
            .await
            .ok();
        state
            .register_node(test_node_info(node2, 5001), Vec::new())
            .await
            .ok();

        let config = PipelineConfig::new(ModelId::new("test", "v1"), "/models/test", 2, DType::F16);
        let pipeline_id = state
            .create_pipeline(config, vec![(node1, 0..10), (node2, 10..20)], None)
            .await
            .expect("pipeline creation should succeed");

        let last_stage = state
            .get_pipeline_last_stage(pipeline_id)
            .await
            .expect("should succeed");
        assert_eq!(last_stage, node2);
    }
}
