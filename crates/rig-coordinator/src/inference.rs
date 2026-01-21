use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use rig_core::{
    CoordError, GenerationParams, InferenceOutput, InferenceRequest, InferenceResult, PipelineId,
    RequestId, UsageStats,
};
use rig_inference::{Sampler, StopChecker, StopReason};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::state::CoordinatorState;

#[allow(dead_code)]
#[derive(Debug)]
struct GenerationState {
    request_id: RequestId,
    pipeline_id: PipelineId,
    sampler: Sampler,
    stop_checker: StopChecker,
    generated_tokens: Vec<u32>,
    prompt_tokens: usize,
    start_time: Instant,
    time_to_first_token_ms: Option<u64>,
    params: GenerationParams,
}

impl GenerationState {
    fn new(
        request_id: RequestId,
        pipeline_id: PipelineId,
        params: GenerationParams,
        eos_token: u32,
        prompt_tokens: usize,
        seed: Option<u64>,
    ) -> Self {
        let sampler = Sampler::new(&params, seed);
        let stop_checker = StopChecker::new(eos_token, params.max_tokens);

        Self {
            request_id,
            pipeline_id,
            sampler,
            stop_checker,
            generated_tokens: Vec::new(),
            prompt_tokens,
            start_time: Instant::now(),
            time_to_first_token_ms: None,
            params,
        }
    }

    fn sample_token(&mut self, logits: &[f32]) -> (u32, StopReason) {
        let token = self.sampler.sample(logits);
        self.generated_tokens.push(token);

        if self.time_to_first_token_ms.is_none() {
            #[allow(clippy::cast_possible_truncation)]
            let ttft = self.start_time.elapsed().as_millis() as u64;
            self.time_to_first_token_ms = Some(ttft);
        }

        let stop_reason = self.stop_checker.should_stop(&self.generated_tokens);
        (token, stop_reason)
    }

    #[allow(clippy::cast_possible_truncation, clippy::missing_const_for_fn)]
    fn current_position(&self) -> u32 {
        (self.prompt_tokens + self.generated_tokens.len()) as u32
    }

    #[allow(clippy::cast_possible_truncation)]
    fn usage_stats(&self) -> UsageStats {
        let total_time_ms = self.start_time.elapsed().as_millis() as u64;
        UsageStats::new(
            self.prompt_tokens,
            self.generated_tokens.len(),
            total_time_ms,
            self.time_to_first_token_ms.unwrap_or(0),
        )
    }
}

#[derive(Debug)]
pub enum GenerationDecision {
    Continue {
        request_id: RequestId,
        token: u32,
        position: u32,
    },
    Finish {
        request_id: RequestId,
        generated_tokens: Vec<u32>,
        reason: StopReason,
        time_to_first_token_ms: u64,
    },
}

pub struct InferenceEngine {
    state: Arc<CoordinatorState>,
    generations: RwLock<HashMap<RequestId, GenerationState>>,
}

impl InferenceEngine {
    #[must_use]
    pub fn new(state: Arc<CoordinatorState>) -> Self {
        Self {
            state,
            generations: RwLock::new(HashMap::new()),
        }
    }

    #[allow(clippy::significant_drop_tightening)]
    pub async fn start_generation(
        &self,
        pipeline_id: PipelineId,
        request: &InferenceRequest,
        eos_token: u32,
        prompt_tokens: usize,
        seed: Option<u64>,
    ) -> Result<(), CoordError> {
        let request_id = request.request_id;

        let mut generations = self.generations.write().await;

        if generations.contains_key(&request_id) {
            return Err(CoordError::InvalidRequest(format!(
                "Generation already active for request {request_id}"
            )));
        }

        let state = GenerationState::new(
            request_id,
            pipeline_id,
            request.params.clone(),
            eos_token,
            prompt_tokens,
            seed,
        );

        generations.insert(request_id, state);
        Ok(())
    }

    #[allow(clippy::significant_drop_tightening)]
    pub async fn process_logits(
        &self,
        request_id: RequestId,
        logits: &[f32],
        eos_token: u32,
    ) -> Result<GenerationDecision, CoordError> {
        let mut generations = self.generations.write().await;

        let state = generations.get_mut(&request_id).ok_or_else(|| {
            CoordError::InvalidRequest(format!("No active generation for request {request_id}"))
        })?;

        if state.generated_tokens.is_empty() {
            state.stop_checker.set_eos_token(eos_token);
            debug!(%request_id, eos_token, "Updated EOS token from worker");
        }

        let (token, stop_reason) = state.sample_token(logits);
        let position = state.current_position();
        let tokens_generated = state.generated_tokens.len();

        debug!(
            %request_id,
            token,
            position,
            tokens_generated,
            "Sampled token"
        );

        if stop_reason.is_stopped() {
            debug!(
                %request_id,
                tokens_generated,
                reason = %stop_reason,
                "Generation complete"
            );

            let generated_tokens = state.generated_tokens.clone();
            let time_to_first_token_ms = state.time_to_first_token_ms.unwrap_or(0);

            Ok(GenerationDecision::Finish {
                request_id,
                generated_tokens,
                reason: stop_reason,
                time_to_first_token_ms,
            })
        } else {
            Ok(GenerationDecision::Continue {
                request_id,
                token,
                position,
            })
        }
    }

    #[allow(clippy::significant_drop_tightening)]
    pub async fn complete_generation(
        &self,
        request_id: RequestId,
        text: String,
    ) -> Result<InferenceResult, CoordError> {
        let mut generations = self.generations.write().await;

        let state = generations.remove(&request_id).ok_or_else(|| {
            CoordError::InvalidRequest(format!(
                "No generation to complete for request {request_id}"
            ))
        })?;

        let usage = state.usage_stats();

        debug!(
            %request_id,
            prompt_tokens = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
            total_time_ms = usage.total_time_ms,
            "Generation completed"
        );

        Ok(InferenceResult::new(
            request_id,
            InferenceOutput::text(text),
            usage,
        ))
    }

    pub async fn cancel_generation(&self, request_id: RequestId) {
        let mut generations = self.generations.write().await;
        if generations.remove(&request_id).is_some() {
            warn!(%request_id, "Generation cancelled");
        }
    }

    pub async fn has_active_generation(&self, request_id: RequestId) -> bool {
        let generations = self.generations.read().await;
        generations.contains_key(&request_id)
    }

    pub async fn active_generation_count(&self) -> usize {
        let generations = self.generations.read().await;
        generations.len()
    }

    #[must_use]
    pub const fn state(&self) -> &Arc<CoordinatorState> {
        &self.state
    }
}

impl std::fmt::Debug for InferenceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceEngine")
            .field("state", &"...")
            .field("active_generations", &"...")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::config::CoordinatorConfig;
    use rig_core::ModelId;

    fn test_config() -> CoordinatorConfig {
        CoordinatorConfig::default()
    }

    fn test_request() -> InferenceRequest {
        InferenceRequest::text(ModelId::new("test", "v1"), "Hello world")
            .with_params(GenerationParams::default().with_max_tokens(10))
    }

    #[tokio::test]
    async fn test_start_generation() {
        let state = Arc::new(CoordinatorState::new(&test_config()));
        let engine = InferenceEngine::new(state);

        let request = test_request();
        let pipeline_id = PipelineId::new();

        let result = engine
            .start_generation(pipeline_id, &request, 2, 10, Some(42))
            .await;

        assert!(result.is_ok());
        assert!(engine.has_active_generation(request.request_id).await);
    }

    #[tokio::test]
    async fn test_duplicate_generation_rejected() {
        let state = Arc::new(CoordinatorState::new(&test_config()));
        let engine = InferenceEngine::new(state);

        let request = test_request();
        let pipeline_id = PipelineId::new();

        engine
            .start_generation(pipeline_id, &request, 2, 10, Some(42))
            .await
            .unwrap();

        let result = engine
            .start_generation(pipeline_id, &request, 2, 10, Some(42))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_process_logits_continue() {
        let state = Arc::new(CoordinatorState::new(&test_config()));
        let engine = InferenceEngine::new(state);

        let request = test_request();
        let pipeline_id = PipelineId::new();

        engine
            .start_generation(pipeline_id, &request, 2, 10, Some(42))
            .await
            .unwrap();

        let logits = vec![0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0];
        let eos_token = 2u32;

        let decision = engine
            .process_logits(request.request_id, &logits, eos_token)
            .await
            .unwrap();

        match decision {
            GenerationDecision::Continue { token, .. } => {
                assert_eq!(token, 5);
            }
            GenerationDecision::Finish { .. } => panic!("Expected Continue"),
        }
    }

    #[tokio::test]
    async fn test_process_logits_eos_stops() {
        let state = Arc::new(CoordinatorState::new(&test_config()));
        let engine = InferenceEngine::new(state);

        let mut request = test_request();
        request.params = GenerationParams::default()
            .with_temperature(0.0)
            .with_max_tokens(100);
        let pipeline_id = PipelineId::new();
        let eos_token = 2u32;

        engine
            .start_generation(pipeline_id, &request, eos_token, 10, Some(42))
            .await
            .unwrap();

        let mut logits = vec![0.0; 10];
        logits[eos_token as usize] = 10.0;

        let decision = engine
            .process_logits(request.request_id, &logits, eos_token)
            .await
            .unwrap();

        match decision {
            GenerationDecision::Finish {
                reason,
                time_to_first_token_ms,
                ..
            } => {
                assert_eq!(reason, StopReason::EosToken);
                let _ = time_to_first_token_ms;
            }
            GenerationDecision::Continue { .. } => panic!("Expected Finish due to EOS"),
        }
    }

    #[tokio::test]
    async fn test_process_logits_max_tokens_stops() {
        let state = Arc::new(CoordinatorState::new(&test_config()));
        let engine = InferenceEngine::new(state);

        let mut request = test_request();
        request.params = GenerationParams::default()
            .with_temperature(0.0)
            .with_max_tokens(3);
        let pipeline_id = PipelineId::new();

        engine
            .start_generation(pipeline_id, &request, 999, 10, Some(42))
            .await
            .unwrap();

        let logits = vec![10.0, 0.0, 0.0, 0.0, 0.0];
        let eos_token = 999u32;

        for i in 0..3 {
            let decision = engine
                .process_logits(request.request_id, &logits, eos_token)
                .await
                .unwrap();

            if i < 2 {
                assert!(matches!(decision, GenerationDecision::Continue { .. }));
            } else {
                match decision {
                    GenerationDecision::Finish {
                        reason,
                        time_to_first_token_ms,
                        ..
                    } => {
                        assert_eq!(reason, StopReason::MaxTokens);
                        let _ = time_to_first_token_ms;
                    }
                    GenerationDecision::Continue { .. } => {
                        panic!("Expected Finish due to MaxTokens")
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_complete_generation() {
        let state = Arc::new(CoordinatorState::new(&test_config()));
        let engine = InferenceEngine::new(state);

        let request = test_request();
        let pipeline_id = PipelineId::new();

        engine
            .start_generation(pipeline_id, &request, 2, 5, Some(42))
            .await
            .unwrap();

        let logits = vec![10.0, 0.0, 0.0];
        let eos_token = 2u32;
        engine
            .process_logits(request.request_id, &logits, eos_token)
            .await
            .unwrap();

        let result = engine
            .complete_generation(request.request_id, "Generated text".to_string())
            .await
            .unwrap();

        assert_eq!(result.request_id, request.request_id);
        assert!(result.is_success());
        assert!(!engine.has_active_generation(request.request_id).await);
    }

    #[tokio::test]
    async fn test_cancel_generation() {
        let state = Arc::new(CoordinatorState::new(&test_config()));
        let engine = InferenceEngine::new(state);

        let request = test_request();
        let pipeline_id = PipelineId::new();

        engine
            .start_generation(pipeline_id, &request, 2, 10, Some(42))
            .await
            .unwrap();

        assert!(engine.has_active_generation(request.request_id).await);

        engine.cancel_generation(request.request_id).await;

        assert!(!engine.has_active_generation(request.request_id).await);
    }
}
