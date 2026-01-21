use std::path::Path;
use std::time::{Duration, Instant};

use rig_core::{
    Activation, ActivationMetadata, Address, Assignment, CoordinatorMessage, DType, InferenceInput,
    InferenceRequest, LoadedPartition, ModelId, ModelInfo, ModelSpec, NodeId, NodeInfo, NodeStatus,
    Partition, PartitionSpec, RequestId, Runtime, RuntimeCapabilities, RuntimeError, Shape,
    TensorData, Tokenizer, UsageStats,
};
use rig_runtime_candle::CandleRuntime;
use tokio::sync::broadcast;
use tracing::{debug, info, instrument, warn};

use crate::config::{RuntimeConfig, WorkerConfig};
use crate::coordinator_client::{CoordinatorClient, create_heartbeat_client};
use crate::error::WorkerError;
use crate::peer_connection::{PeerConnection, PeerListener};
use crate::stage::PipelineStage;

enum RuntimeWrapper {
    Candle(CandleRuntime),
}

impl RuntimeWrapper {
    fn capabilities(&self) -> RuntimeCapabilities {
        match self {
            Self::Candle(rt) => rt.capabilities(),
        }
    }

    fn discover_model(&self, model_id: ModelId, path: &Path) -> Result<ModelSpec, RuntimeError> {
        match self {
            Self::Candle(rt) => rt.discover_model(model_id, path),
        }
    }

    async fn load_partition(
        &self,
        model: &ModelSpec,
        partition: &PartitionSpec,
    ) -> Result<LoadedPartition, RuntimeError> {
        match self {
            Self::Candle(rt) => rt.load_partition(model, partition).await,
        }
    }
}
pub struct WorkerNode {
    node_id: Option<NodeId>,
    config: WorkerConfig,
    runtime: Option<RuntimeWrapper>,
    coordinator_client: Option<CoordinatorClient>,
    stage: Option<PipelineStage>,
    peer_listener: Option<PeerListener>,
    shutdown_tx: broadcast::Sender<()>,
    model_info: Option<ModelInfo>,
}

impl WorkerNode {
    #[must_use]
    pub fn new(config: WorkerConfig) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);

        Self {
            node_id: None,
            config,
            runtime: None,
            coordinator_client: None,
            stage: None,
            peer_listener: None,
            shutdown_tx,
            model_info: None,
        }
    }

    pub fn set_model_info(&mut self, model_info: ModelInfo) {
        self.model_info = Some(model_info);
    }

    #[must_use]
    pub const fn node_id(&self) -> Option<NodeId> {
        self.node_id
    }

    #[must_use]
    pub const fn config(&self) -> &WorkerConfig {
        &self.config
    }

    #[must_use]
    pub fn shutdown_receiver(&self) -> broadcast::Receiver<()> {
        self.shutdown_tx.subscribe()
    }

    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }

    #[instrument(skip(self))]
    pub fn init_runtime(&mut self) -> Result<(), WorkerError> {
        info!("Initializing runtime");

        let runtime = match &self.config.runtime_config {
            RuntimeConfig::Candle(config) => {
                info!(device = %config.device, "Using Candle runtime");
                let rt = match config.device.as_str() {
                    "cpu" => CandleRuntime::cpu()?,
                    _ => CandleRuntime::new()?,
                };
                RuntimeWrapper::Candle(rt)
            }
        };

        self.runtime = Some(runtime);
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn start_peer_listener(&mut self) -> Result<(), WorkerError> {
        let listener = PeerListener::bind(self.config.listen_addr).await?;
        let local_addr = listener.local_addr()?;
        info!(%local_addr, "Peer listener started");
        self.peer_listener = Some(listener);
        Ok(())
    }

    pub fn peer_listen_addr(&self) -> Option<std::net::SocketAddr> {
        self.peer_listener.as_ref()?.local_addr().ok()
    }

    #[instrument(skip(self), fields(addr = %self.config.coordinator_addr))]
    pub async fn connect_to_coordinator(&mut self) -> Result<(), WorkerError> {
        info!("Connecting to coordinator");
        let client = CoordinatorClient::connect(&self.config.coordinator_addr).await?;
        self.coordinator_client = Some(client);
        info!("Connected to coordinator");
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn register(&mut self) -> Result<NodeId, WorkerError> {
        let capabilities = self
            .runtime
            .as_ref()
            .ok_or_else(|| WorkerError::config("Runtime not initialized"))?
            .capabilities();

        let listen_addr = self
            .peer_listen_addr()
            .ok_or_else(|| WorkerError::config("Peer listener not started"))?;

        let node_id = NodeId::new();
        let info = NodeInfo::new(
            node_id,
            vec![Address::tcp(listen_addr)],
            NodeStatus::Healthy,
            capabilities,
        );

        let models: Vec<ModelInfo> = self.model_info.iter().cloned().collect();

        let client = self
            .coordinator_client
            .as_mut()
            .ok_or(WorkerError::NotRegistered)?;

        info!(%node_id, num_models = models.len(), "Registering with coordinator");
        let registered_id = client.register_with_models(info, models).await?;
        self.node_id = Some(registered_id);

        info!(%registered_id, "Registration successful");
        Ok(registered_id)
    }

    #[instrument(skip(self))]
    pub async fn start_heartbeat_task(&self) -> Result<(), WorkerError> {
        let node_id = self.node_id.ok_or(WorkerError::NotRegistered)?;

        let coordinator_addr = self.config.coordinator_addr.clone();
        let interval = self.config.heartbeat_interval;
        let mut shutdown_rx = self.shutdown_receiver();

        tokio::spawn(async move {
            if let Err(e) =
                run_heartbeat_loop(node_id, &coordinator_addr, interval, &mut shutdown_rx).await
            {
                warn!(%node_id, error = %e, "Heartbeat task failed");
            }
        });

        info!(%node_id, interval = ?interval, "Heartbeat task started");
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn wait_for_assignment(&mut self) -> Result<Assignment, WorkerError> {
        let client = self
            .coordinator_client
            .as_mut()
            .ok_or(WorkerError::NotRegistered)?;

        loop {
            if let Some(assignment) = client.get_assignment().await? {
                info!(
                    pipeline_id = %assignment.pipeline_id,
                    stage_id = %assignment.stage_id,
                    layer_range = ?assignment.layer_range,
                    "Received assignment"
                );
                return Ok(assignment);
            }
            debug!("No assignment yet, waiting...");
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }

    #[instrument(skip(self, model_spec), fields(
        pipeline_id = %assignment.pipeline_id,
        stage_id = %assignment.stage_id,
        layer_range = ?assignment.layer_range
    ))]
    pub async fn load_partition(
        &self,
        assignment: &Assignment,
        model_spec: &ModelSpec,
    ) -> Result<LoadedPartition, WorkerError> {
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| WorkerError::config("Runtime not initialized"))?;

        let partition_spec =
            PartitionSpec::new(assignment.layer_range.clone(), rig_core::DType::F16);

        info!(
            model = %model_spec.model_id,
            path = %model_spec.path.display(),
            "Loading partition"
        );

        let loaded = runtime.load_partition(model_spec, &partition_spec).await?;

        info!("Partition loaded successfully");
        Ok(loaded)
    }

    #[instrument(skip(self, assignment))]
    pub async fn establish_peer_connections(
        &mut self,
        assignment: &Assignment,
    ) -> Result<(Option<PeerConnection>, Option<PeerConnection>), WorkerError> {
        let next_peer = if let Some(ref next_addr) = assignment.neighbors.next {
            let addr = next_addr
                .first_address()
                .ok_or_else(|| WorkerError::peer_connection("Next peer has no address"))?;

            info!(peer_node_id = %next_addr.node_id, addr = %addr, "Connecting to next stage");
            Some(PeerConnection::connect(addr, next_addr.node_id).await?)
        } else {
            None
        };

        let prev_peer = if let Some(ref prev_addr) = assignment.neighbors.prev {
            let listener = self
                .peer_listener
                .as_ref()
                .ok_or_else(|| WorkerError::peer_connection("Peer listener not started"))?;

            info!(peer_node_id = %prev_addr.node_id, "Waiting for connection from previous stage");
            Some(listener.accept(prev_addr.node_id).await?)
        } else {
            None
        };

        debug!(
            has_prev_peer = prev_peer.is_some(),
            has_next_peer = next_peer.is_some(),
            "Peer connections established"
        );
        Ok((prev_peer, next_peer))
    }

    #[instrument(skip(self))]
    pub async fn report_ready(&mut self, assignment: &Assignment) -> Result<(), WorkerError> {
        let client = self
            .coordinator_client
            .as_mut()
            .ok_or(WorkerError::NotRegistered)?;

        info!(pipeline_id = %assignment.pipeline_id, "Reporting ready");
        client.report_ready(assignment.pipeline_id).await?;
        info!("Ready status acknowledged");
        Ok(())
    }

    pub fn set_stage(&mut self, stage: PipelineStage) {
        self.stage = Some(stage);
    }

    #[must_use]
    pub const fn stage(&self) -> Option<&PipelineStage> {
        self.stage.as_ref()
    }

    #[must_use]
    pub const fn stage_mut(&mut self) -> Option<&mut PipelineStage> {
        self.stage.as_mut()
    }

    #[instrument(skip(self))]
    pub async fn run(&mut self, model_id: ModelId) -> Result<(), WorkerError> {
        self.init_runtime()?;

        let model_path = self
            .config
            .get_model_path(&model_id)
            .ok_or_else(|| WorkerError::ModelNotFound(model_id.to_string()))?;

        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| WorkerError::config("Runtime not initialized"))?;

        let model_spec = runtime.discover_model(model_id, model_path)?;
        info!(
            num_layers = model_spec.num_layers,
            hidden_dim = model_spec.hidden_dim,
            "Discovered model metadata"
        );

        self.start_peer_listener().await?;

        self.connect_to_coordinator().await?;

        let model_info = ModelInfo::new(
            model_spec.model_id.clone(),
            model_spec.path.clone(),
            model_spec.num_layers,
            model_spec.hidden_dim,
        );
        self.set_model_info(model_info);

        self.register().await?;

        self.start_heartbeat_task().await?;

        let assignment = self.wait_for_assignment().await?;

        let loaded = self.load_partition(&assignment, &model_spec).await?;

        if self.config.enable_warmup {
            run_warmup(loaded.partition())?;
        }

        let (prev_peer, next_peer) = self.establish_peer_connections(&assignment).await?;

        debug!(
            has_prev_peer = prev_peer.is_some(),
            has_next_peer = next_peer.is_some(),
            "Building PipelineStage"
        );
        let mut stage =
            PipelineStage::from_loaded(loaded, assignment.clone(), prev_peer, next_peer);
        debug!(
            is_last_stage = stage.is_last_stage(),
            has_next_peer_in_stage = stage.has_next_peer(),
            "PipelineStage built"
        );

        if stage.is_multi_stage_last() {
            let coordinator_addr = self.config.coordinator_addr.clone();
            let node_id = self.node_id.ok_or(WorkerError::NotRegistered)?;
            let coord_client =
                CoordinatorClient::connect_for_node(&coordinator_addr, node_id).await?;
            stage
                .set_coordinator_client(std::sync::Arc::new(tokio::sync::Mutex::new(coord_client)));
            info!("Multi-stage last stage coordinator client connected");
        }

        self.set_stage(stage);

        self.report_ready(&assignment).await?;

        info!("Worker ready, starting inference loop");
        let shutdown_rx = self.shutdown_receiver();

        if let Some(stage) = self.stage.as_mut() {
            if stage.is_first_stage() {
                self.run_first_stage_loop(&assignment, shutdown_rx).await?;
            } else {
                stage.run(shutdown_rx).await?;
            }
        }

        info!("Shutdown signal received, cleaning up");
        Ok(())
    }

    async fn run_first_stage_loop(
        &mut self,
        assignment: &Assignment,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) -> Result<(), WorkerError> {
        info!("Starting first stage polling loop");

        let pipeline_id = assignment.pipeline_id;

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("Shutdown signal received, stopping first stage");
                    break;
                }

                () = tokio::time::sleep(Duration::from_millis(100)) => {
                    let request = self.poll_for_request(pipeline_id).await?;

                    if let Some(req) = request {
                        self.process_first_stage_request(req).await?;
                    }
                }
            }
        }

        Ok(())
    }

    async fn poll_for_request(
        &mut self,
        pipeline_id: rig_core::PipelineId,
    ) -> Result<Option<rig_core::InferenceRequest>, WorkerError> {
        let client = self
            .coordinator_client
            .as_mut()
            .ok_or(WorkerError::NotRegistered)?;

        client.get_pending_request(pipeline_id).await
    }

    async fn process_first_stage_request(
        &mut self,
        request: rig_core::InferenceRequest,
    ) -> Result<(), WorkerError> {
        let request_id = request.request_id;
        debug!(%request_id, "Processing request");

        if self.stage.is_none() {
            return Err(WorkerError::NoAssignment);
        }

        let activation = {
            let stage = self.stage.as_ref().ok_or(WorkerError::NoAssignment)?;

            let tokenizer = stage
                .tokenizer()
                .ok_or_else(|| WorkerError::config("Model does not support tokenization"))?;

            create_initial_activation(&request, tokenizer)?
        };

        let prompt_tokens = activation.metadata.positions.len();
        let start_time = Instant::now();

        let stage = self.stage.as_mut().ok_or(WorkerError::NoAssignment)?;
        let output = stage.forward(activation)?;
        stage.send_activation(&output).await?;

        self.run_multi_stage_generation_loop(request_id, prompt_tokens, start_time)
            .await?;

        debug!(%request_id, "Request processed");
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    async fn run_multi_stage_generation_loop(
        &mut self,
        request_id: RequestId,
        prompt_tokens: usize,
        start_time: Instant,
    ) -> Result<(), WorkerError> {
        debug!(%request_id, "Entering multi-stage generation loop");

        let mut streaming_client = CoordinatorClient::connect_for_node(
            &self.config.coordinator_addr,
            self.node_id.ok_or(WorkerError::NotRegistered)?,
        )
        .await?;

        let mut decode_stream = self
            .stage
            .as_ref()
            .and_then(|s| s.tokenizer())
            .and_then(|t| t.create_decode_stream(true).ok());

        loop {
            let client = self
                .coordinator_client
                .as_mut()
                .ok_or(WorkerError::NotRegistered)?;

            let control = client.get_generation_control(request_id).await?;

            match control {
                CoordinatorMessage::ContinueGeneration {
                    request_id: _,
                    token,
                    position,
                } => {
                    debug!(%request_id, %token, %position, "Continuing generation");

                    if let Some(ref mut stream) = decode_stream {
                        if let Ok(Some(new_text)) = stream.step(token) {
                            if let Err(e) = streaming_client.send_token(request_id, new_text).await
                            {
                                warn!(%request_id, error = %e, "Failed to stream token");
                            }
                        }
                    }

                    let activation = create_decode_activation(request_id, token, position);

                    let stage = self.stage.as_mut().ok_or(WorkerError::NoAssignment)?;
                    let output = stage.forward(activation)?;

                    stage.send_activation(&output).await?;
                }

                CoordinatorMessage::FinishGeneration {
                    request_id: _,
                    generated_tokens,
                    time_to_first_token_ms,
                } => {
                    debug!(
                        %request_id,
                        num_tokens = generated_tokens.len(),
                        "Generation finished"
                    );

                    let elapsed = start_time.elapsed();
                    #[allow(clippy::cast_possible_truncation)]
                    let usage = UsageStats {
                        prompt_tokens,
                        completion_tokens: generated_tokens.len(),
                        total_time_ms: elapsed.as_millis() as u64,
                        time_to_first_token_ms,
                    };

                    streaming_client
                        .send_streaming_complete(request_id, usage)
                        .await?;
                    break;
                }

                CoordinatorMessage::GenerationPending { request_id: _ } => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }

                other => {
                    warn!(
                        %request_id,
                        message = ?other,
                        "Unexpected message in generation loop"
                    );
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        }

        Ok(())
    }
}

impl std::fmt::Debug for WorkerNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerNode")
            .field("node_id", &self.node_id)
            .field("config", &self.config)
            .field("has_runtime", &self.runtime.is_some())
            .field("has_coordinator_client", &self.coordinator_client.is_some())
            .field("has_stage", &self.stage.is_some())
            .finish_non_exhaustive()
    }
}

async fn run_heartbeat_loop(
    node_id: NodeId,
    coordinator_addr: &Address,
    interval: Duration,
    shutdown_rx: &mut broadcast::Receiver<()>,
) -> Result<(), WorkerError> {
    let mut client = create_heartbeat_client(coordinator_addr, node_id).await?;

    let mut heartbeat_interval = tokio::time::interval(interval);

    loop {
        tokio::select! {
            _ = heartbeat_interval.tick() => {
                match client.heartbeat(NodeStatus::Healthy).await {
                    Ok(()) => {},
                    Err(e) => {
                        warn!(%node_id, error = %e, "Heartbeat failed");
                    }
                }
            }
            _ = shutdown_rx.recv() => {
                debug!(%node_id, "Heartbeat loop shutting down");
                break;
            }
        }
    }

    Ok(())
}

fn create_initial_activation(
    request: &InferenceRequest,
    tokenizer: &dyn Tokenizer,
) -> Result<Activation, WorkerError> {
    let request_id = request.request_id;

    let tokens: Vec<u32> = match &request.input {
        InferenceInput::Tokens(tokens) => tokens.clone(),
        InferenceInput::Text(text) => {
            let full_text = if request.params.use_chat_template {
                let mut messages = Vec::new();
                if let Some(sp) = &request.params.system_prompt {
                    messages.push(rig_core::ChatMessage::system(sp));
                }
                messages.push(rig_core::ChatMessage::user(text));
                tokenizer
                    .apply_chat_template(&messages, true)
                    .map_err(|e| WorkerError::config(format!("Chat template failed: {e}")))?
            } else {
                request
                    .params
                    .system_prompt
                    .as_ref()
                    .map_or_else(|| text.clone(), |sp| format!("{sp}\n\n{text}"))
            };

            tokenizer
                .encode(&full_text, true)
                .map_err(|e| WorkerError::config(format!("Tokenization failed: {e}")))?
        }
    };

    let seq_len = tokens.len();

    #[allow(clippy::cast_possible_truncation)]
    let positions: Vec<u32> = (0..seq_len).map(|i| i as u32).collect();

    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for token in &tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }

    let shape = Shape::new(vec![1, seq_len, 1]);
    let data = TensorData::cpu(bytes, DType::I8);

    let metadata = ActivationMetadata::new(request_id, 0, positions, true)
        .with_generation_params(request.params.clone());

    Ok(Activation::new(data, shape, metadata))
}

fn create_decode_activation(request_id: RequestId, token: u32, position: u32) -> Activation {
    let bytes = token.to_le_bytes().to_vec();

    let shape = Shape::new(vec![1, 1, 1]);
    let data = TensorData::cpu(bytes, DType::I8);

    let metadata = ActivationMetadata::new(request_id, 0, vec![position], false);

    Activation::new(data, shape, metadata)
}

#[instrument(skip(partition))]
fn run_warmup(partition: &dyn Partition) -> Result<(), WorkerError> {
    let is_first_stage = partition.spec().layer_range.start == 0;
    if !is_first_stage {
        info!("Skipping warm-up for non-first stage partition");
        return Ok(());
    }

    let start = Instant::now();
    info!("Running warm-up pass to initialize KV cache and compile GPU kernels");

    let warmup_token: u32 = 1;
    let bytes = warmup_token.to_le_bytes().to_vec();
    let shape = Shape::new(vec![1, 1, 1]);
    let data = TensorData::cpu(bytes, DType::I8);

    let warmup_request_id = RequestId::new();
    let metadata = ActivationMetadata::new(warmup_request_id, 0, vec![0], true);

    let warmup_activation = Activation::new(data, shape, metadata);

    let _ = partition
        .forward(warmup_activation)
        .map_err(|e| WorkerError::partition_processing(format!("Warm-up forward failed: {e}")))?;

    partition.release_request_cache(warmup_request_id);

    let elapsed = start.elapsed();
    info!(elapsed_ms = elapsed.as_millis(), "Warm-up complete");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_node_creation() {
        let config = WorkerConfig::default();
        let node = WorkerNode::new(config);

        assert!(node.node_id().is_none());
        assert!(node.stage().is_none());
    }

    #[test]
    fn worker_node_shutdown() {
        let config = WorkerConfig::default();
        let node = WorkerNode::new(config);

        let _rx1 = node.shutdown_receiver();
        node.shutdown();
        let _rx2 = node.shutdown_receiver();
    }
}
