use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use rig_core::{
    Activation, Assignment, GenerationDecision, GenerationParams, LoadedPartition, Partition,
    RequestId, StopReasonProto, Tokenizer,
};
use rig_inference::{Sampler, StopChecker, StopReason};
use tokio::sync::{Mutex, broadcast};
use tracing::{debug, info, instrument, warn};

use crate::coordinator_client::CoordinatorClient;
use crate::error::WorkerError;
use crate::peer_connection::PeerConnection;

struct MultiStageGenerationState {
    sampler: Sampler,
    stop_checker: StopChecker,
    generated_tokens: Vec<u32>,
    prompt_tokens: usize,
    start_time: Instant,
    time_to_first_token_ms: Option<u64>,
}

impl MultiStageGenerationState {
    fn new(
        params: &GenerationParams,
        eos_token: u32,
        prompt_tokens: usize,
        seed: Option<u64>,
    ) -> Self {
        let sampler = Sampler::new(params, seed);
        let stop_checker = StopChecker::new(eos_token, params.max_tokens);

        Self {
            sampler,
            stop_checker,
            generated_tokens: Vec::new(),
            prompt_tokens,
            start_time: Instant::now(),
            time_to_first_token_ms: None,
        }
    }

    fn sample_and_check(&mut self, logits: &[f32]) -> GenerationDecision {
        let token = self.sampler.sample(logits);
        self.generated_tokens.push(token);

        if self.time_to_first_token_ms.is_none() {
            #[allow(clippy::cast_possible_truncation)]
            let ttft = self.start_time.elapsed().as_millis() as u64;
            self.time_to_first_token_ms = Some(ttft);
        }

        let stop_reason = self.stop_checker.should_stop(&self.generated_tokens);

        if stop_reason.is_stopped() {
            GenerationDecision::Finish {
                generated_tokens: self.generated_tokens.clone(),
                stop_reason: Self::convert_stop_reason(stop_reason),
                time_to_first_token_ms: self.time_to_first_token_ms.unwrap_or(0),
            }
        } else {
            #[allow(clippy::cast_possible_truncation)]
            let position = (self.prompt_tokens + self.generated_tokens.len()) as u32;
            GenerationDecision::Continue { token, position }
        }
    }

    fn convert_stop_reason(reason: StopReason) -> StopReasonProto {
        match reason {
            StopReason::MaxTokens => StopReasonProto::MaxTokens,
            StopReason::EosToken => StopReasonProto::EosToken,
            StopReason::StopSequence(seq) => StopReasonProto::StopSequence(seq),
            StopReason::NotStopped => StopReasonProto::MaxTokens,
        }
    }
}

pub struct PipelineStage {
    partition: Box<dyn Partition>,
    tokenizer: Option<Box<dyn Tokenizer>>,
    assignment: Assignment,
    prev_peer: Option<PeerConnection>,
    next_peer: Option<PeerConnection>,
    coord_client: Option<Arc<Mutex<CoordinatorClient>>>,
    generation_states: HashMap<RequestId, MultiStageGenerationState>,
}

impl PipelineStage {
    #[must_use]
    pub fn from_loaded(
        loaded: LoadedPartition,
        assignment: Assignment,
        prev_peer: Option<PeerConnection>,
        next_peer: Option<PeerConnection>,
    ) -> Self {
        let (partition, tokenizer) = loaded.into_parts();
        Self {
            partition,
            tokenizer,
            assignment,
            prev_peer,
            next_peer,
            coord_client: None,
            generation_states: HashMap::new(),
        }
    }

    #[must_use]
    pub fn new(
        partition: Box<dyn Partition>,
        tokenizer: Option<Box<dyn Tokenizer>>,
        assignment: Assignment,
        prev_peer: Option<PeerConnection>,
        next_peer: Option<PeerConnection>,
    ) -> Self {
        Self {
            partition,
            tokenizer,
            assignment,
            prev_peer,
            next_peer,
            coord_client: None,
            generation_states: HashMap::new(),
        }
    }

    #[must_use]
    pub fn partition(&self) -> &dyn Partition {
        &*self.partition
    }

    #[must_use]
    pub fn partition_mut(&mut self) -> &mut Box<dyn Partition> {
        &mut self.partition
    }

    #[must_use]
    pub fn tokenizer(&self) -> Option<&dyn Tokenizer> {
        self.tokenizer.as_deref()
    }

    #[must_use]
    pub fn partition_and_tokenizer(&mut self) -> (&mut dyn Partition, Option<&dyn Tokenizer>) {
        (self.partition.as_mut(), self.tokenizer.as_deref())
    }

    #[must_use]
    pub const fn assignment(&self) -> &Assignment {
        &self.assignment
    }

    #[must_use]
    pub const fn is_first_stage(&self) -> bool {
        self.assignment.neighbors.prev.is_none()
    }

    #[must_use]
    pub const fn is_last_stage(&self) -> bool {
        self.assignment.neighbors.next.is_none()
    }

    pub fn set_prev_peer(&mut self, peer: PeerConnection) {
        self.prev_peer = Some(peer);
    }

    pub fn set_coordinator_client(&mut self, client: Arc<Mutex<CoordinatorClient>>) {
        self.coord_client = Some(client);
    }

    #[must_use]
    pub const fn is_multi_stage_last(&self) -> bool {
        self.is_last_stage() && !self.is_first_stage()
    }

    #[must_use]
    pub const fn has_prev_peer(&self) -> bool {
        self.prev_peer.is_some()
    }

    #[must_use]
    pub const fn has_next_peer(&self) -> bool {
        self.next_peer.is_some()
    }

    #[instrument(skip(self))]
    pub async fn recv_activation(&self) -> Result<Activation, WorkerError> {
        match &self.prev_peer {
            Some(peer) => peer.recv_activation().await,
            None => Err(WorkerError::peer_connection(
                "No previous peer connection available",
            )),
        }
    }

    #[instrument(skip(self, activation))]
    pub async fn send_activation(&self, activation: &Activation) -> Result<(), WorkerError> {
        if let Some(peer) = &self.next_peer {
            peer.send_activation(activation).await
        } else {
            Err(WorkerError::peer_connection(
                "No next peer configured for send_activation",
            ))
        }
    }

    #[instrument(skip(self, activation), fields(
        request_id = %activation.metadata.request_id,
        stage_id = %self.assignment.stage_id
    ))]
    pub fn forward(&mut self, activation: Activation) -> Result<Activation, WorkerError> {
        debug!(
            layer_range = ?self.assignment.layer_range,
            "Processing activation through stage"
        );

        let output = self.partition.forward(activation)?;

        debug!("Forward pass complete");
        Ok(output)
    }

    async fn process_multi_stage_last(
        &mut self,
        activation: Activation,
    ) -> Result<(), WorkerError> {
        let request_id = activation.metadata.request_id;
        let eos_token = self.tokenizer().map_or(2, Tokenizer::eos_token);

        if !self.generation_states.contains_key(&request_id) {
            let params = activation
                .metadata
                .generation_params
                .clone()
                .unwrap_or_default();

            let prompt_tokens = activation.metadata.positions.len();

            let state = MultiStageGenerationState::new(&params, eos_token, prompt_tokens, None);
            self.generation_states.insert(request_id, state);

            debug!(
                %request_id,
                prompt_tokens,
                eos_token,
                "Initialized generation state for request"
            );
        }

        let output = self.forward(activation)?;
        let logits = extract_logits(&output);

        if logits.is_empty() {
            return Err(WorkerError::partition_processing(
                "Multi-stage last stage: forward pass returned no logits",
            ));
        }

        let state = self.generation_states.get_mut(&request_id).ok_or_else(|| {
            WorkerError::partition_processing("Generation state not found for request")
        })?;

        let decision = state.sample_and_check(&logits);

        let is_finished = matches!(decision, GenerationDecision::Finish { .. });

        debug!(
            %request_id,
            logits_len = logits.len(),
            is_finished,
            "Multi-stage last stage: sampled token locally"
        );

        let coord_client = self.coord_client.as_ref().ok_or_else(|| {
            WorkerError::config(
                "Multi-stage last stage requires coordinator client to be configured",
            )
        })?;

        coord_client
            .lock()
            .await
            .send_generation_decision(request_id, decision)
            .await?;

        if is_finished {
            self.generation_states.remove(&request_id);
            debug!(%request_id, "Cleaned up generation state after completion");
        }

        debug!(%request_id, "Generation decision sent to coordinator");
        Ok(())
    }

    pub async fn step(&mut self) -> Result<(), WorkerError> {
        let input = self.recv_activation().await?;
        let output = self.forward(input)?;
        self.send_activation(&output).await?;

        Ok(())
    }

    #[instrument(skip(self, shutdown_rx), fields(
        stage_id = %self.assignment.stage_id,
        pipeline_id = %self.assignment.pipeline_id
    ))]
    pub async fn run(
        &mut self,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) -> Result<(), WorkerError> {
        info!(
            layer_range = ?self.assignment.layer_range,
            is_first = self.is_first_stage(),
            is_last = self.is_last_stage(),
            "Starting stage processing loop"
        );

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("Shutdown signal received, stopping stage");
                    break;
                }

                result = self.recv_activation() => {
                    match result {
                        Ok(activation) => {
                            debug!(
                                request_id = %activation.metadata.request_id,
                                seq = activation.metadata.sequence_num,
                                "Received activation"
                            );

                            if self.is_multi_stage_last() {
                                debug!(
                                    request_id = %activation.metadata.request_id,
                                    "Multi-stage last stage: processing and sampling locally"
                                );
                                self.process_multi_stage_last(activation).await?;
                            } else {
                                let output = self.forward(activation)?;
                                self.send_activation(&output).await?;
                            }
                        }
                        Err(WorkerError::Transport(rig_core::TransportError::Closed)) => {
                            info!("Previous peer connection closed, stopping stage");
                            break;
                        }
                        Err(e) => {
                            warn!(error = %e, "Error receiving activation");
                            return Err(e);
                        }
                    }
                }
            }
        }

        info!("Stage processing loop stopped");
        Ok(())
    }
}

fn extract_logits(activation: &Activation) -> Vec<f32> {
    use rig_core::DType;

    let bytes = activation.data.as_bytes();
    let dtype = activation.dtype();

    match dtype {
        DType::F32 => {
            if bytes.len() < 4 {
                return Vec::new();
            }
            bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                    f32::from_le_bytes(arr)
                })
                .collect()
        }
        DType::F16 => {
            if bytes.len() < 2 {
                return Vec::new();
            }
            bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let arr: [u8; 2] = chunk.try_into().unwrap_or([0; 2]);
                    half::f16::from_le_bytes(arr).to_f32()
                })
                .collect()
        }
        DType::BF16 => {
            if bytes.len() < 2 {
                return Vec::new();
            }
            bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let arr: [u8; 2] = chunk.try_into().unwrap_or([0; 2]);
                    half::bf16::from_le_bytes(arr).to_f32()
                })
                .collect()
        }
        _ => Vec::new(),
    }
}

impl std::fmt::Debug for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineStage")
            .field("stage_id", &self.assignment.stage_id)
            .field("pipeline_id", &self.assignment.pipeline_id)
            .field("layer_range", &self.assignment.layer_range)
            .field("is_first_stage", &self.is_first_stage())
            .field("is_last_stage", &self.is_last_stage())
            .finish_non_exhaustive()
    }
}

pub struct PipelineStageBuilder {
    partition: Option<Box<dyn Partition>>,
    tokenizer: Option<Box<dyn Tokenizer>>,
    assignment: Option<Assignment>,
    prev_peer: Option<PeerConnection>,
    next_peer: Option<PeerConnection>,
    coord_client: Option<Arc<Mutex<CoordinatorClient>>>,
}

impl PipelineStageBuilder {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            partition: None,
            tokenizer: None,
            assignment: None,
            prev_peer: None,
            next_peer: None,
            coord_client: None,
        }
    }

    #[must_use]
    pub fn with_loaded_partition(mut self, loaded: LoadedPartition) -> Self {
        let (partition, tokenizer) = loaded.into_parts();
        self.partition = Some(partition);
        self.tokenizer = tokenizer;
        self
    }

    #[must_use]
    pub fn with_partition(mut self, partition: Box<dyn Partition>) -> Self {
        self.partition = Some(partition);
        self
    }

    #[must_use]
    pub fn with_tokenizer(mut self, tokenizer: Box<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    #[must_use]
    pub fn with_assignment(mut self, assignment: Assignment) -> Self {
        self.assignment = Some(assignment);
        self
    }

    #[must_use]
    pub fn with_prev_peer(mut self, peer: PeerConnection) -> Self {
        self.prev_peer = Some(peer);
        self
    }

    #[must_use]
    pub fn with_next_peer(mut self, peer: PeerConnection) -> Self {
        self.next_peer = Some(peer);
        self
    }

    #[must_use]
    pub fn with_coordinator_client(mut self, client: Arc<Mutex<CoordinatorClient>>) -> Self {
        self.coord_client = Some(client);
        self
    }

    pub fn build(self) -> Result<PipelineStage, WorkerError> {
        let partition = self
            .partition
            .ok_or_else(|| WorkerError::config("Partition is required"))?;
        let assignment = self
            .assignment
            .ok_or_else(|| WorkerError::config("Assignment is required"))?;

        let mut stage = PipelineStage::new(
            partition,
            self.tokenizer,
            assignment,
            self.prev_peer,
            self.next_peer,
        );

        if let Some(client) = self.coord_client {
            stage.set_coordinator_client(client);
        }

        Ok(stage)
    }
}

impl Default for PipelineStageBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_requires_partition() {
        let result = PipelineStageBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_requires_assignment() {
        use rig_core::{Activation, DType, MemoryUsage, PartitionError, PartitionSpec};

        struct MockPartition {
            spec: PartitionSpec,
        }

        impl MockPartition {
            fn new() -> Self {
                Self {
                    spec: PartitionSpec::new(0..10, DType::F16),
                }
            }
        }

        impl Partition for MockPartition {
            fn spec(&self) -> &PartitionSpec {
                &self.spec
            }

            fn forward(&self, _activation: Activation) -> Result<Activation, PartitionError> {
                Err(PartitionError::ForwardFailed("mock".to_string()))
            }

            fn memory_usage(&self) -> MemoryUsage {
                MemoryUsage::new()
            }
        }

        let result = PipelineStageBuilder::new()
            .with_partition(Box::new(MockPartition::new()))
            .build();

        assert!(result.is_err());
    }
}
