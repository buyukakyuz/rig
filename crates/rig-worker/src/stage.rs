use std::sync::Arc;

use rig_core::{Activation, Assignment, LoadedPartition, Partition, SamplingParams, Tokenizer};
use tokio::sync::{Mutex, broadcast};
use tracing::{debug, instrument, trace, warn};

use crate::coordinator_client::CoordinatorClient;
use crate::error::WorkerError;
use crate::peer_connection::PeerConnection;

pub struct PipelineStage {
    partition: Box<dyn Partition>,
    tokenizer: Option<Box<dyn Tokenizer>>,
    assignment: Assignment,
    prev_peer: Option<PeerConnection>,
    next_peer: Option<PeerConnection>,
    coord_client: Option<Arc<Mutex<CoordinatorClient>>>,
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

        let params = activation
            .metadata
            .generation_params
            .clone()
            .unwrap_or_default();
        let seed = params.seed.unwrap_or_else(rand::random::<u64>);
        let sampling_params =
            SamplingParams::new(params.temperature, params.top_p, params.top_k, seed);

        let sample_result = self
            .partition
            .forward_sample(activation, &sampling_params)?
            .ok_or_else(|| {
                WorkerError::partition_processing(
                    "Multi-stage last stage: forward_sample returned None (not final stage?)",
                )
            })?;

        debug!(
            %request_id,
            token = sample_result.token,
            "Multi-stage last stage: sampled token, sending to coordinator"
        );

        let coord_client = self.coord_client.as_ref().ok_or_else(|| {
            WorkerError::config(
                "Multi-stage last stage requires coordinator client to be configured",
            )
        })?;

        coord_client
            .lock()
            .await
            .send_token_sampled(request_id, sample_result.token)
            .await?;

        trace!(%request_id, "Token sent to coordinator");
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
        debug!(
            layer_range = ?self.assignment.layer_range,
            is_first = self.is_first_stage(),
            is_last = self.is_last_stage(),
            "Starting stage processing loop"
        );

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    debug!("Shutdown signal received, stopping stage");
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
                                trace!(
                                    request_id = %activation.metadata.request_id,
                                    positions = ?activation.metadata.positions,
                                    shape = ?activation.shape.dims(),
                                    is_prefill = activation.metadata.is_prefill,
                                    "Multi-stage last stage: processing and sampling locally"
                                );
                                self.process_multi_stage_last(activation).await?;
                            } else {
                                let output = self.forward(activation)?;
                                self.send_activation(&output).await?;
                            }
                        }
                        Err(WorkerError::Transport(rig_core::TransportError::Closed)) => {
                            debug!("Previous peer connection closed, stopping stage");
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

        debug!("Stage processing loop stopped");
        Ok(())
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
