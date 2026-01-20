use std::sync::Arc;
use std::time::Instant;

use rig_core::{
    Activation, ActivationMetadata, Assignment, DType, GenerationParams, Partition, RequestId,
    Sampler, Shape, StopChecker, TensorData, UsageStats,
};
use tokio::sync::{Mutex, broadcast, mpsc};
use tracing::{debug, info, instrument, trace, warn};

use crate::coordinator_client::CoordinatorClient;
use crate::error::WorkerError;
use crate::peer_connection::PeerConnection;

pub struct PipelineStage {
    partition: Box<dyn Partition>,
    assignment: Assignment,
    prev_peer: Option<PeerConnection>,
    next_peer: Option<PeerConnection>,
    coord_client: Option<Arc<Mutex<CoordinatorClient>>>,
}

impl PipelineStage {
    #[must_use]
    pub fn new(
        partition: Box<dyn Partition>,
        assignment: Assignment,
        prev_peer: Option<PeerConnection>,
        next_peer: Option<PeerConnection>,
    ) -> Self {
        Self {
            partition,
            assignment,
            prev_peer,
            next_peer,
            coord_client: None,
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
        debug!(
            has_next_peer = self.next_peer.is_some(),
            is_last_stage = self.is_last_stage(),
            "send_activation called"
        );
        if let Some(peer) = &self.next_peer {
            debug!("Sending activation to next peer");
            peer.send_activation(activation).await
        } else {
            Err(WorkerError::peer_connection(
                "No next peer configured for send_activation",
            ))
        }
    }

    #[instrument(skip(self, initial_activation, params, token_tx), fields(
        request_id = %initial_activation.metadata.request_id
    ))]
    #[allow(clippy::too_many_lines)]
    pub async fn generate(
        &mut self,
        initial_activation: Activation,
        params: &GenerationParams,
        token_tx: mpsc::UnboundedSender<String>,
    ) -> Result<UsageStats, WorkerError> {
        let request_id = initial_activation.metadata.request_id;
        let start = Instant::now();

        let eos_token = self
            .partition
            .tokenizer()
            .ok_or_else(|| {
                WorkerError::partition_processing("Model does not support tokenization")
            })?
            .eos_token();

        let prompt_tokens = initial_activation.metadata.positions.len();
        debug!(prompt_tokens, "Prompt token count");

        debug!("Starting prefill forward pass");
        let output = self.forward(initial_activation)?;
        let logits = extract_logits(&output);

        if logits.is_empty() {
            return Err(WorkerError::partition_processing(
                "Forward pass returned no logits",
            ));
        }

        let mut sampler = Sampler::new(params, None);
        let stop_checker = StopChecker::with_stop_sequences(
            eos_token,
            params.max_tokens,
            params.stop_sequences.clone(),
        );
        debug!(
            eos_token,
            max_tokens = params.max_tokens,
            stop_sequences = ?params.stop_sequences,
            "StopChecker configured"
        );

        if self.partition.tokenizer().is_none() {
            return Err(WorkerError::partition_processing(
                "Model does not support tokenization",
            ));
        }

        let decode_tokens = |partition: &dyn Partition, tokens: &[u32]| -> Option<String> {
            partition.tokenizer().and_then(|t| t.decode(tokens).ok())
        };

        let first_token = sampler.sample(&logits);
        let mut generated_tokens = vec![first_token];
        #[allow(clippy::cast_possible_truncation)]
        let time_to_first_token = start.elapsed().as_millis() as u64;
        debug!(token = first_token, "Sampled first token");

        let mut current_decoded_text = String::new();

        if let Some(decoded_text) = decode_tokens(self.partition.as_ref(), &generated_tokens) {
            let new_text = &decoded_text[current_decoded_text.len()..];
            if !new_text.is_empty() {
                let _ = token_tx.send(new_text.to_string());
            }
            current_decoded_text = decoded_text;
        }

        while stop_checker
            .should_stop_with_text(&generated_tokens, &current_decoded_text)
            .should_continue()
        {
            let last_token = *generated_tokens
                .last()
                .ok_or_else(|| WorkerError::partition_processing("No tokens generated"))?;
            let position = prompt_tokens + generated_tokens.len() - 1;
            trace!(last_token, position, "Decode step");

            let decode_activation = create_decode_activation(request_id, last_token, position);
            let output = self.forward(decode_activation)?;
            let logits = extract_logits(&output);

            if logits.is_empty() {
                return Err(WorkerError::partition_processing(
                    "Decode forward pass returned no logits",
                ));
            }

            let token = sampler.sample(&logits);
            generated_tokens.push(token);

            if let Some(decoded_text) = decode_tokens(self.partition.as_ref(), &generated_tokens) {
                let new_text = &decoded_text[current_decoded_text.len()..];
                if !new_text.is_empty() {
                    let _ = token_tx.send(new_text.to_string());
                }
                current_decoded_text = decoded_text;
            }

            if generated_tokens.len() % 10 == 0 {
                debug!(count = generated_tokens.len(), "Generated tokens");
            }
        }

        let stop_reason =
            stop_checker.should_stop_with_text(&generated_tokens, &current_decoded_text);
        debug!(
            tokens_generated = generated_tokens.len(),
            %stop_reason,
            "Streaming generation complete"
        );

        self.partition.release_request_cache(request_id);

        #[allow(clippy::cast_possible_truncation)]
        let total_time = start.elapsed().as_millis() as u64;

        info!(
            completion_tokens = generated_tokens.len(),
            %stop_reason,
            total_time_ms = total_time,
            "Streaming generation complete"
        );

        Ok(UsageStats {
            prompt_tokens,
            completion_tokens: generated_tokens.len(),
            total_time_ms: total_time,
            time_to_first_token_ms: time_to_first_token,
        })
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

        let eos_token = self
            .partition
            .tokenizer()
            .map_or(2, rig_core::Tokenizer::eos_token);

        let output = self.forward(activation)?;

        let logits = extract_logits(&output);

        if logits.is_empty() {
            return Err(WorkerError::partition_processing(
                "Multi-stage last stage: forward pass returned no logits",
            ));
        }

        debug!(
            %request_id,
            logits_len = logits.len(),
            "Multi-stage last stage: sending logits to coordinator"
        );

        let coord_client = self.coord_client.as_ref().ok_or_else(|| {
            WorkerError::config(
                "Multi-stage last stage requires coordinator client to be configured",
            )
        })?;

        coord_client
            .lock()
            .await
            .send_logits(request_id, logits, eos_token)
            .await?;

        debug!(%request_id, "Logits sent to coordinator");
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
                                    "Multi-stage last stage: forwarding and sending logits"
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

fn create_decode_activation(request_id: RequestId, token: u32, position: usize) -> Activation {
    let bytes = token.to_le_bytes().to_vec();
    trace!(
        token,
        position,
        bytes_len = bytes.len(),
        "Creating decode activation"
    );
    let data = TensorData::cpu(bytes, DType::I8);

    let shape = Shape::new(vec![1, 1, 1]);

    #[allow(clippy::cast_possible_truncation)]
    let position_u32 = position as u32;

    let metadata = ActivationMetadata::new(request_id, position_u32, vec![position_u32], false);

    Activation::new(data, shape, metadata)
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
            assignment: None,
            prev_peer: None,
            next_peer: None,
            coord_client: None,
        }
    }

    #[must_use]
    pub fn with_partition(mut self, partition: Box<dyn Partition>) -> Self {
        self.partition = Some(partition);
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

        let mut stage = PipelineStage::new(partition, assignment, self.prev_peer, self.next_peer);

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
