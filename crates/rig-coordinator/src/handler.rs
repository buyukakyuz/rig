use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use rig_core::{
    CliMessage, CliResponse, CliSubmitRequest, Codec, CoordError, CoordinatorIncoming,
    CoordinatorMessage, CoordinatorOutgoing, FramedTransport, InferenceRequest, ModelId, NodeId,
    RegisterRequest, RegisterResponse, TransportError, WorkerMessage, types::protocol::error_codes,
};
use rig_message_bincode::BincodeCodec;
use rig_transport_tcp::TcpTransport;
use tokio::time::timeout;

use crate::inference::{GenerationDecision, InferenceEngine};
use crate::state::CoordinatorState;

pub struct ConnectionHandler {
    state: Arc<CoordinatorState>,
    engine: Arc<InferenceEngine>,
    transport: TcpTransport,
    codec: BincodeCodec,
    node_id: Option<NodeId>,
    is_registered: bool,
    remote_addr: SocketAddr,
}

impl ConnectionHandler {
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(
        state: Arc<CoordinatorState>,
        engine: Arc<InferenceEngine>,
        transport: TcpTransport,
        remote_addr: SocketAddr,
    ) -> Self {
        Self {
            state,
            engine,
            transport,
            codec: BincodeCodec::new(),
            node_id: None,
            is_registered: false,
            remote_addr,
        }
    }

    pub async fn run(mut self) -> Result<(), CoordError> {
        let result = self.message_loop().await;

        if let Some(node_id) = self.node_id
            && self.is_registered
        {
            tracing::debug!(
                %node_id,
                addr = %self.remote_addr,
                "Control connection closed (node stays registered)"
            );
        }

        result
    }

    async fn message_loop(&mut self) -> Result<(), CoordError> {
        loop {
            let frame = match self.transport.recv_frame().await {
                Ok(frame) => frame,
                Err(TransportError::Closed) => {
                    tracing::debug!(addr = %self.remote_addr, "Connection closed by peer");
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!(addr = %self.remote_addr, error = %e, "Transport error");
                    return Err(e.into());
                }
            };

            let msg: CoordinatorIncoming = match self.codec.decode(&frame) {
                Ok(msg) => msg,
                Err(e) => {
                    tracing::warn!(addr = %self.remote_addr, error = %e, "Failed to decode message");
                    let response = CoordinatorOutgoing::Worker(CoordinatorMessage::Error {
                        code: error_codes::INVALID_REQUEST,
                        message: format!("Failed to decode message: {e}"),
                    });
                    self.send_response(&response).await?;
                    continue;
                }
            };

            let response = match self.handle_incoming(msg).await {
                Ok(resp) => resp,
                Err(e) => {
                    tracing::warn!(addr = %self.remote_addr, error = %e, "Error handling message");
                    Self::error_to_response(&e)
                }
            };

            self.send_response(&response).await?;
        }
    }

    async fn handle_incoming(
        &mut self,
        msg: CoordinatorIncoming,
    ) -> Result<CoordinatorOutgoing, CoordError> {
        match msg {
            CoordinatorIncoming::Worker(worker_msg) => {
                let resp = self.handle_worker_message(worker_msg).await?;
                Ok(CoordinatorOutgoing::Worker(resp))
            }
            CoordinatorIncoming::Cli(cli_msg) => {
                let resp = self.handle_cli_message(cli_msg).await?;
                Ok(CoordinatorOutgoing::Cli(resp))
            }
        }
    }

    async fn handle_worker_message(
        &mut self,
        msg: WorkerMessage,
    ) -> Result<CoordinatorMessage, CoordError> {
        match msg {
            WorkerMessage::Register(req) => self.handle_register(req).await,
            WorkerMessage::Heartbeat(req) => self.handle_heartbeat(req).await,
            WorkerMessage::GetAssignment => self.handle_get_assignment().await,
            WorkerMessage::ReportReady { pipeline_id } => {
                self.handle_report_ready(pipeline_id).await
            }
            WorkerMessage::Deregister => self.handle_deregister().await,
            WorkerMessage::GetPendingRequest { pipeline_id } => {
                self.handle_get_pending_request(pipeline_id).await
            }
            WorkerMessage::ReturnLogits {
                request_id,
                logits,
                eos_token,
            } => {
                self.handle_return_logits(request_id, logits, eos_token)
                    .await
            }
            WorkerMessage::TokenGenerated {
                request_id,
                token_text,
            } => self.handle_token_generated(request_id, token_text).await,
            WorkerMessage::StreamingComplete { request_id, usage } => {
                self.handle_streaming_complete(request_id, usage).await
            }
            WorkerMessage::GetGenerationControl { request_id } => {
                self.handle_get_generation_control(request_id).await
            }
        }
    }

    async fn handle_cli_message(&self, msg: CliMessage) -> Result<CliResponse, CoordError> {
        match msg {
            CliMessage::GetStatus => self.handle_get_status().await,
            CliMessage::SubmitRequest(req) => self.handle_cli_submit_request(req).await,
            CliMessage::CreatePipeline(req) => self.handle_cli_create_pipeline(req).await,
            CliMessage::CreatePipelineAuto(req) => self.handle_cli_create_pipeline_auto(req).await,
            CliMessage::GetPipeline { pipeline_id } => {
                self.handle_cli_get_pipeline(pipeline_id).await
            }
            CliMessage::ListPipelines => self.handle_cli_list_pipelines().await,
        }
    }

    async fn handle_register(
        &mut self,
        req: RegisterRequest,
    ) -> Result<CoordinatorMessage, CoordError> {
        tracing::info!(
            addr = %self.remote_addr,
            node_id = %req.node_info.node_id,
            num_models = req.available_models.len(),
            "Registration request"
        );

        match self
            .state
            .register_node(req.node_info, req.available_models)
            .await
        {
            Ok(node_id) => {
                self.node_id = Some(node_id);
                self.is_registered = true;
                Ok(CoordinatorMessage::RegisterResponse(
                    RegisterResponse::accepted(node_id),
                ))
            }
            Err(CoordError::MaxNodesReached { max }) => Ok(CoordinatorMessage::RegisterResponse(
                RegisterResponse::rejected(format!("Cluster full: maximum {max} nodes")),
            )),
            Err(CoordError::InvalidRequest(msg)) => Ok(CoordinatorMessage::RegisterResponse(
                RegisterResponse::rejected(msg),
            )),
            Err(e) => Err(e),
        }
    }

    async fn handle_heartbeat(
        &self,
        req: rig_core::HeartbeatRequest,
    ) -> Result<CoordinatorMessage, CoordError> {
        if let Some(registered_id) = self.node_id
            && registered_id != req.node_id
        {
            return Err(CoordError::InvalidRequest(
                "Heartbeat node ID doesn't match registered node".to_string(),
            ));
        }
        self.state.heartbeat(req.node_id, req.status).await?;
        Ok(CoordinatorMessage::HeartbeatAck)
    }

    async fn handle_get_assignment(&self) -> Result<CoordinatorMessage, CoordError> {
        let node_id = self.node_id.ok_or(CoordError::NotRegistered)?;
        let assignment = self.state.get_assignment(node_id).await?;
        Ok(CoordinatorMessage::Assignment(assignment))
    }

    async fn handle_report_ready(
        &self,
        pipeline_id: rig_core::PipelineId,
    ) -> Result<CoordinatorMessage, CoordError> {
        let node_id = self.node_id.ok_or(CoordError::NotRegistered)?;
        self.state.mark_ready(node_id, pipeline_id).await?;
        Ok(CoordinatorMessage::ReadyAck)
    }

    async fn handle_deregister(&mut self) -> Result<CoordinatorMessage, CoordError> {
        if let Some(node_id) = self.node_id.take() {
            self.state.deregister_node(node_id).await;
            tracing::info!(%node_id, addr = %self.remote_addr, "Node deregistered gracefully");
        }

        Err(CoordError::ReportFailed(
            "Deregistration complete, closing connection".to_string(),
        ))
    }

    async fn handle_get_pending_request(
        &self,
        pipeline_id: rig_core::PipelineId,
    ) -> Result<CoordinatorMessage, CoordError> {
        let node_id = self.node_id.ok_or(CoordError::NotRegistered)?;

        let first_stage = self.state.get_pipeline_first_stage(pipeline_id).await?;
        if first_stage != node_id {
            return Err(CoordError::InvalidRequest(
                "Only the first stage can poll for requests".to_string(),
            ));
        }

        let request = self.state.get_pending_request(pipeline_id).await;
        Ok(CoordinatorMessage::PendingRequest(request))
    }

    async fn handle_return_logits(
        &self,
        request_id: rig_core::RequestId,
        logits: Vec<f32>,
        eos_token: u32,
    ) -> Result<CoordinatorMessage, CoordError> {
        tracing::debug!(
            %request_id,
            logits_len = logits.len(),
            eos_token,
            "Received logits from last stage"
        );

        let decision = self
            .engine
            .process_logits(request_id, &logits, eos_token)
            .await?;

        self.state
            .store_generation_decision(request_id, decision)
            .await;

        Ok(CoordinatorMessage::ResultAck)
    }

    async fn handle_get_generation_control(
        &self,
        request_id: rig_core::RequestId,
    ) -> Result<CoordinatorMessage, CoordError> {
        tracing::trace!(%request_id, "Generation control poll");

        match self.state.take_generation_decision(request_id).await {
            Some(GenerationDecision::Continue {
                request_id,
                token,
                position,
            }) => {
                tracing::debug!(%request_id, token, position, "Returning ContinueGeneration");
                Ok(CoordinatorMessage::ContinueGeneration {
                    request_id,
                    token,
                    position,
                })
            }
            Some(GenerationDecision::Finish {
                request_id,
                generated_tokens,
                reason,
            }) => {
                tracing::debug!(%request_id, reason = %reason, "Returning FinishGeneration");
                Ok(CoordinatorMessage::FinishGeneration {
                    request_id,
                    generated_tokens,
                })
            }
            None => Ok(CoordinatorMessage::GenerationPending { request_id }),
        }
    }

    async fn handle_token_generated(
        &self,
        request_id: rig_core::RequestId,
        token_text: String,
    ) -> Result<CoordinatorMessage, CoordError> {
        tracing::trace!(%request_id, token_len = token_text.len(), "Token received");

        if !self.state.forward_token(request_id, token_text).await {
            tracing::warn!(%request_id, "No streaming session found for token");
        }

        Ok(CoordinatorMessage::ResultAck)
    }

    async fn handle_streaming_complete(
        &self,
        request_id: rig_core::RequestId,
        usage: rig_core::UsageStats,
    ) -> Result<CoordinatorMessage, CoordError> {
        tracing::debug!(
            %request_id,
            prompt_tokens = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
            "Streaming complete"
        );

        if !self
            .state
            .complete_streaming_session(request_id, usage)
            .await
        {
            tracing::warn!(%request_id, "No streaming session found to complete");
        }

        Ok(CoordinatorMessage::ResultAck)
    }

    async fn handle_get_status(&self) -> Result<CliResponse, CoordError> {
        tracing::debug!(addr = %self.remote_addr, "CLI status request");

        let status = self.state.build_cluster_status().await;
        Ok(CliResponse::Status(status))
    }

    async fn handle_cli_submit_request(
        &self,
        req: CliSubmitRequest,
    ) -> Result<CliResponse, CoordError> {
        let pipeline_info = self.state.get_pipeline_info(req.pipeline_id).await?;
        let is_multi_stage = pipeline_info.stages.len() > 1;

        let inference_request =
            InferenceRequest::new(ModelId::new("default", "v1"), req.input.clone())
                .with_params(req.params.clone());

        if is_multi_stage {
            let placeholder_eos = 0u32;

            self.engine
                .start_generation(
                    req.pipeline_id,
                    &inference_request,
                    placeholder_eos,
                    0,
                    None,
                )
                .await?;
        }

        self.handle_streaming_request(req.pipeline_id, inference_request, req.timeout_ms)
            .await
    }

    async fn handle_streaming_request(
        &self,
        pipeline_id: rig_core::PipelineId,
        inference_request: InferenceRequest,
        timeout_ms: Option<u64>,
    ) -> Result<CliResponse, CoordError> {
        use tokio::sync::{mpsc, oneshot};

        let request_id = inference_request.request_id;
        let idle_timeout = Duration::from_millis(timeout_ms.unwrap_or(60_000));

        let (token_tx, mut token_rx) = mpsc::unbounded_channel::<String>();
        let (complete_tx, complete_rx) = oneshot::channel::<rig_core::UsageStats>();

        self.state
            .start_streaming_session(request_id, token_tx, complete_tx)
            .await;

        self.state
            .submit_request(pipeline_id, inference_request)
            .await?;

        let start_response = CliResponse::StreamStart { request_id };
        self.send_response(&CoordinatorOutgoing::Cli(start_response))
            .await?;

        loop {
            match timeout(idle_timeout, token_rx.recv()).await {
                Ok(Some(token_text)) => {
                    let token_response = CliResponse::Token {
                        request_id,
                        token_text,
                    };
                    if let Err(e) = self
                        .send_response(&CoordinatorOutgoing::Cli(token_response))
                        .await
                    {
                        tracing::warn!(%request_id, error = %e, "Failed to send token");
                        break;
                    }
                }
                Ok(None) => {
                    break;
                }
                Err(_) => {
                    tracing::warn!(
                        %request_id,
                        idle_timeout_ms = idle_timeout.as_millis(),
                        "Idle timeout - no token received"
                    );
                    return Ok(CliResponse::error(
                        error_codes::INTERNAL_ERROR,
                        format!(
                            "Idle timeout: no token received for {}ms",
                            idle_timeout.as_millis()
                        ),
                    ));
                }
            }
        }

        match timeout(idle_timeout, complete_rx).await {
            Ok(Ok(usage)) => {
                tracing::debug!(%request_id, "Streaming request completed");
                Ok(CliResponse::StreamComplete { request_id, usage })
            }
            Ok(Err(_recv_error)) => {
                tracing::warn!(%request_id, "Streaming completion channel closed unexpectedly");
                Ok(CliResponse::error(
                    error_codes::INTERNAL_ERROR,
                    "Streaming completion failed",
                ))
            }
            Err(_) => {
                tracing::warn!(%request_id, "Timeout waiting for completion signal");
                Ok(CliResponse::error(
                    error_codes::INTERNAL_ERROR,
                    format!(
                        "Idle timeout: no completion signal for {}ms",
                        idle_timeout.as_millis()
                    ),
                ))
            }
        }
    }

    async fn handle_cli_create_pipeline(
        &self,
        req: rig_core::CliCreatePipelineRequest,
    ) -> Result<CliResponse, CoordError> {
        tracing::info!(
            addr = %self.remote_addr,
            model = %req.config.model_id,
            stages = req.assignments.len(),
            "CLI create pipeline request"
        );

        let assignments = req.assignments_as_ranges();
        let pipeline_id = self
            .state
            .create_pipeline(req.config, assignments, req.pipeline_id)
            .await?;

        Ok(CliResponse::PipelineCreated { pipeline_id })
    }

    async fn handle_cli_create_pipeline_auto(
        &self,
        req: rig_core::CliCreatePipelineAutoRequest,
    ) -> Result<CliResponse, CoordError> {
        use crate::state::CoordinatorState;

        let model_id = rig_core::ModelId::new(&req.model_name, &req.model_version);

        tracing::info!(
            addr = %self.remote_addr,
            model = %model_id,
            num_stages = ?req.num_stages,
            "CLI create pipeline auto request"
        );

        let nodes_with_model = self.state.get_nodes_with_model(&model_id).await;

        if nodes_with_model.is_empty() {
            return Err(CoordError::InvalidRequest(format!(
                "No nodes have model '{model_id}' available"
            )));
        }

        let (num_layers, _hidden_dim) =
            self.state.get_model_info(&model_id).await.ok_or_else(|| {
                CoordError::InvalidRequest(format!("Model '{model_id}' not found in registry"))
            })?;

        let num_stages = req.num_stages.unwrap_or(nodes_with_model.len());

        if num_stages == 0 {
            return Err(CoordError::InvalidRequest(
                "At least one stage is required".to_string(),
            ));
        }

        if num_stages > nodes_with_model.len() {
            return Err(CoordError::InvalidRequest(format!(
                "Requested {} stages but only {} nodes have model '{}'",
                num_stages,
                nodes_with_model.len(),
                model_id
            )));
        }

        let ranges = CoordinatorState::partition_layers(num_layers, num_stages);

        let assignments: Vec<(rig_core::NodeId, std::ops::Range<usize>)> = nodes_with_model
            .iter()
            .take(num_stages)
            .zip(ranges)
            .map(|((node_id, _), range)| (*node_id, range))
            .collect();

        let model_path = nodes_with_model[0].1.model_path.clone();

        let config = rig_core::PipelineConfig::new(model_id, model_path, num_stages, req.dtype);

        let pipeline_id = self
            .state
            .create_pipeline(config, assignments, req.pipeline_id)
            .await?;

        tracing::info!(
            %pipeline_id,
            num_stages,
            num_layers,
            "Auto-partitioned pipeline created"
        );

        Ok(CliResponse::PipelineCreated { pipeline_id })
    }

    async fn handle_cli_get_pipeline(
        &self,
        pipeline_id: rig_core::PipelineId,
    ) -> Result<CliResponse, CoordError> {
        tracing::debug!(
            addr = %self.remote_addr,
            %pipeline_id,
            "CLI get pipeline request"
        );

        let info = self.state.get_pipeline_info(pipeline_id).await?;
        Ok(CliResponse::PipelineInfo(info))
    }

    async fn handle_cli_list_pipelines(&self) -> Result<CliResponse, CoordError> {
        tracing::debug!(addr = %self.remote_addr, "CLI list pipelines request");

        let pipelines = self.state.list_pipelines().await;
        Ok(CliResponse::PipelineList(pipelines))
    }

    async fn send_response(&self, response: &CoordinatorOutgoing) -> Result<(), CoordError> {
        let frame = self.codec.encode(response)?;
        self.transport.send_frame(&frame).await?;
        Ok(())
    }

    fn error_to_response(error: &CoordError) -> CoordinatorOutgoing {
        let (code, message) = match error {
            CoordError::NotRegistered => (error_codes::NOT_REGISTERED, "Node not registered"),
            CoordError::MaxNodesReached { .. } => {
                (error_codes::MAX_NODES_REACHED, "Maximum nodes reached")
            }
            CoordError::NodeNotFound(_) => (error_codes::NODE_NOT_FOUND, "Node not found"),
            CoordError::PipelineNotFound(_) => {
                (error_codes::PIPELINE_NOT_FOUND, "Pipeline not found")
            }
            CoordError::InvalidRequest(_) => (error_codes::INVALID_REQUEST, "Invalid request"),
            _ => (error_codes::INTERNAL_ERROR, "Internal error"),
        };

        CoordinatorOutgoing::Worker(CoordinatorMessage::Error {
            code,
            message: format!("{message}: {error}"),
        })
    }
}
