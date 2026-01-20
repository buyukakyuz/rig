use std::sync::Arc;

use rig_core::{CoordError, InferenceRequest, PipelineId};
use tracing::instrument;

use crate::state::CoordinatorState;

pub struct RequestHandler {
    state: Arc<CoordinatorState>,
}

impl RequestHandler {
    #[must_use]
    pub const fn new(state: Arc<CoordinatorState>) -> Self {
        Self { state }
    }

    #[instrument(skip(self, request), fields(request_id = %request.request_id, pipeline_id = %pipeline_id))]
    pub async fn submit_request(
        &self,
        pipeline_id: PipelineId,
        request: InferenceRequest,
    ) -> Result<(), CoordError> {
        self.state.submit_request(pipeline_id, request).await
    }

    pub async fn pending_count(&self, pipeline_id: PipelineId) -> usize {
        self.state.pending_request_count(pipeline_id).await
    }

    #[must_use]
    pub const fn state(&self) -> &Arc<CoordinatorState> {
        &self.state
    }
}

impl std::fmt::Debug for RequestHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestHandler")
            .field("state", &"...")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::config::CoordinatorConfig;
    use rig_core::{DType, ModelId, PipelineConfig};

    fn test_request(model_id: ModelId) -> InferenceRequest {
        InferenceRequest::text(model_id, "test input")
    }

    #[tokio::test]
    async fn test_request_handler_creation() {
        let config = CoordinatorConfig::default();
        let state = Arc::new(CoordinatorState::new(&config));
        let _handler = RequestHandler::new(state);
    }

    #[tokio::test]
    async fn test_submit_to_non_ready_pipeline() {
        let config = CoordinatorConfig::default();
        let state = Arc::new(CoordinatorState::new(&config));
        let handler = RequestHandler::new(Arc::clone(&state));

        let node_id = rig_core::NodeId::new();
        let node_info = rig_core::NodeInfo::new(
            node_id,
            vec![rig_core::Address::tcp("127.0.0.1:5000".parse().unwrap())],
            rig_core::NodeStatus::Healthy,
            rig_core::RuntimeCapabilities::new("test", 0, vec![]),
        );
        state.register_node(node_info, Vec::new()).await.ok();

        let pipeline_config =
            PipelineConfig::new(ModelId::new("test", "v1"), "/models/test", 1, DType::F16);
        let pipeline_id = state
            .create_pipeline(pipeline_config, vec![(node_id, 0..10)], None)
            .await
            .unwrap();

        let request = test_request(ModelId::new("test", "v1"));
        let result = handler.submit_request(pipeline_id, request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_submit_and_get_request() {
        let config = CoordinatorConfig::default();
        let state = Arc::new(CoordinatorState::new(&config));
        let handler = RequestHandler::new(Arc::clone(&state));

        let node_id = rig_core::NodeId::new();
        let node_info = rig_core::NodeInfo::new(
            node_id,
            vec![rig_core::Address::tcp("127.0.0.1:5000".parse().unwrap())],
            rig_core::NodeStatus::Healthy,
            rig_core::RuntimeCapabilities::new("test", 0, vec![]),
        );
        state.register_node(node_info, Vec::new()).await.ok();

        let pipeline_config =
            PipelineConfig::new(ModelId::new("test", "v1"), "/models/test", 1, DType::F16);
        let pipeline_id = state
            .create_pipeline(pipeline_config, vec![(node_id, 0..10)], None)
            .await
            .unwrap();

        state.mark_ready(node_id, pipeline_id).await.ok();

        let request = test_request(ModelId::new("test", "v1"));
        let request_id = request.request_id;
        handler.submit_request(pipeline_id, request).await.unwrap();

        assert_eq!(handler.pending_count(pipeline_id).await, 1);

        let pending = state.get_pending_request(pipeline_id).await;
        assert!(pending.is_some());
        assert_eq!(pending.unwrap().request_id, request_id);
    }
}
