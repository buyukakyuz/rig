use rig_core::{
    Address, Assignment, Codec, CoordinatorIncoming, CoordinatorMessage, CoordinatorOutgoing,
    FramedTransport, HeartbeatRequest, InferenceRequest, ModelInfo, NodeId, NodeInfo, NodeStatus,
    PipelineId, RegisterRequest, RequestId, TransportFactory, UsageStats, WorkerMessage,
};
use rig_message_bincode::BincodeCodec;
use rig_transport_tcp::{TcpConfig, TcpTransport, TcpTransportFactory};

use crate::error::WorkerError;

pub struct CoordinatorClient {
    transport: TcpTransport,
    codec: BincodeCodec,
    node_id: Option<NodeId>,
}

impl CoordinatorClient {
    pub async fn connect(addr: &Address) -> Result<Self, WorkerError> {
        let config = TcpConfig::default().with_read_timeout(None);
        let factory = TcpTransportFactory::with_config(config);
        let transport = factory.connect(addr).await?;

        Ok(Self {
            transport,
            codec: BincodeCodec::new(),
            node_id: None,
        })
    }

    pub async fn connect_for_node(addr: &Address, node_id: NodeId) -> Result<Self, WorkerError> {
        let mut client = Self::connect(addr).await?;
        client.node_id = Some(node_id);
        Ok(client)
    }

    #[must_use]
    pub const fn node_id(&self) -> Option<NodeId> {
        self.node_id
    }

    #[must_use]
    pub const fn is_registered(&self) -> bool {
        self.node_id.is_some()
    }

    pub async fn register(&mut self, info: NodeInfo) -> Result<NodeId, WorkerError> {
        self.register_with_models(info, Vec::new()).await
    }

    pub async fn register_with_models(
        &mut self,
        info: NodeInfo,
        models: Vec<ModelInfo>,
    ) -> Result<NodeId, WorkerError> {
        let msg = WorkerMessage::Register(RegisterRequest::with_models(info, models));
        self.send(&msg).await?;

        let response: CoordinatorMessage = self.recv().await?;
        match response {
            CoordinatorMessage::RegisterResponse(r) if r.accepted => {
                self.node_id = Some(r.node_id);
                Ok(r.node_id)
            }
            CoordinatorMessage::RegisterResponse(r) => Err(WorkerError::RegistrationRejected(
                r.reason.unwrap_or_default(),
            )),
            CoordinatorMessage::Error { code, message } => {
                Err(WorkerError::CoordinatorError { code, message })
            }
            _ => Err(WorkerError::UnexpectedResponse(
                "Expected RegisterResponse".to_string(),
            )),
        }
    }

    pub async fn heartbeat(&mut self, status: NodeStatus) -> Result<(), WorkerError> {
        let node_id = self.node_id.ok_or(WorkerError::NotRegistered)?;

        let msg = WorkerMessage::Heartbeat(HeartbeatRequest::new(node_id, status));
        self.send(&msg).await?;

        let response: CoordinatorMessage = self.recv().await?;
        match response {
            CoordinatorMessage::HeartbeatAck => Ok(()),
            CoordinatorMessage::Error { code, message } => {
                Err(WorkerError::CoordinatorError { code, message })
            }
            _ => Err(WorkerError::UnexpectedResponse(
                "Expected HeartbeatAck".to_string(),
            )),
        }
    }

    pub async fn get_assignment(&mut self) -> Result<Option<Assignment>, WorkerError> {
        if !self.is_registered() {
            return Err(WorkerError::NotRegistered);
        }

        self.send(&WorkerMessage::GetAssignment).await?;

        let response: CoordinatorMessage = self.recv().await?;
        match response {
            CoordinatorMessage::Assignment(assignment) => Ok(assignment),
            CoordinatorMessage::Error { code, message } => {
                Err(WorkerError::CoordinatorError { code, message })
            }
            _ => Err(WorkerError::UnexpectedResponse(
                "Expected Assignment".to_string(),
            )),
        }
    }

    pub async fn report_ready(&mut self, pipeline_id: PipelineId) -> Result<(), WorkerError> {
        if !self.is_registered() {
            return Err(WorkerError::NotRegistered);
        }

        self.send(&WorkerMessage::ReportReady { pipeline_id })
            .await?;

        let response: CoordinatorMessage = self.recv().await?;
        match response {
            CoordinatorMessage::ReadyAck => Ok(()),
            CoordinatorMessage::Error { code, message } => {
                Err(WorkerError::CoordinatorError { code, message })
            }
            _ => Err(WorkerError::UnexpectedResponse(
                "Expected ReadyAck".to_string(),
            )),
        }
    }

    pub async fn get_pending_request(
        &mut self,
        pipeline_id: PipelineId,
    ) -> Result<Option<InferenceRequest>, WorkerError> {
        if !self.is_registered() {
            return Err(WorkerError::NotRegistered);
        }

        self.send(&WorkerMessage::GetPendingRequest { pipeline_id })
            .await?;

        let response: CoordinatorMessage = self.recv().await?;
        match response {
            CoordinatorMessage::PendingRequest(request) => Ok(request),
            CoordinatorMessage::Error { code, message } => {
                Err(WorkerError::CoordinatorError { code, message })
            }
            _ => Err(WorkerError::UnexpectedResponse(
                "Expected PendingRequest".to_string(),
            )),
        }
    }

    pub async fn deregister(&mut self) -> Result<(), WorkerError> {
        self.send(&WorkerMessage::Deregister).await?;
        self.node_id = None;
        Ok(())
    }

    pub async fn send_token(
        &mut self,
        request_id: RequestId,
        token_text: String,
    ) -> Result<(), WorkerError> {
        if !self.is_registered() {
            return Err(WorkerError::NotRegistered);
        }

        self.send(&WorkerMessage::TokenGenerated {
            request_id,
            token_text,
        })
        .await?;

        let response: CoordinatorMessage = self.recv().await?;
        match response {
            CoordinatorMessage::ResultAck => Ok(()),
            CoordinatorMessage::Error { code, message } => {
                Err(WorkerError::CoordinatorError { code, message })
            }
            _ => Err(WorkerError::UnexpectedResponse(
                "Expected ResultAck".to_string(),
            )),
        }
    }

    pub async fn send_streaming_complete(
        &mut self,
        request_id: RequestId,
        usage: UsageStats,
    ) -> Result<(), WorkerError> {
        if !self.is_registered() {
            return Err(WorkerError::NotRegistered);
        }

        self.send(&WorkerMessage::StreamingComplete { request_id, usage })
            .await?;

        let response: CoordinatorMessage = self.recv().await?;
        match response {
            CoordinatorMessage::ResultAck => Ok(()),
            CoordinatorMessage::Error { code, message } => {
                Err(WorkerError::CoordinatorError { code, message })
            }
            _ => Err(WorkerError::UnexpectedResponse(
                "Expected ResultAck".to_string(),
            )),
        }
    }

    pub async fn send_logits(
        &mut self,
        request_id: RequestId,
        logits: Vec<f32>,
        eos_token: u32,
    ) -> Result<(), WorkerError> {
        if !self.is_registered() {
            return Err(WorkerError::NotRegistered);
        }

        self.send(&WorkerMessage::ReturnLogits {
            request_id,
            logits,
            eos_token,
        })
        .await?;

        let response: CoordinatorMessage = self.recv().await?;
        match response {
            CoordinatorMessage::ResultAck => Ok(()),
            CoordinatorMessage::Error { code, message } => {
                Err(WorkerError::CoordinatorError { code, message })
            }
            _ => Err(WorkerError::UnexpectedResponse(
                "Expected ResultAck".to_string(),
            )),
        }
    }

    pub async fn get_generation_control(
        &mut self,
        request_id: RequestId,
    ) -> Result<CoordinatorMessage, WorkerError> {
        if !self.is_registered() {
            return Err(WorkerError::NotRegistered);
        }

        self.send(&WorkerMessage::GetGenerationControl { request_id })
            .await?;

        let response: CoordinatorMessage = self.recv().await?;
        match response {
            CoordinatorMessage::ContinueGeneration { .. }
            | CoordinatorMessage::FinishGeneration { .. }
            | CoordinatorMessage::GenerationPending { .. } => Ok(response),
            CoordinatorMessage::Error { code, message } => {
                Err(WorkerError::CoordinatorError { code, message })
            }
            _ => Err(WorkerError::UnexpectedResponse(
                "Expected generation control response".to_string(),
            )),
        }
    }

    async fn send(&self, msg: &WorkerMessage) -> Result<(), WorkerError> {
        let incoming = CoordinatorIncoming::Worker(msg.clone());
        let frame = self.codec.encode(&incoming)?;
        self.transport.send_frame(&frame).await?;
        Ok(())
    }

    async fn recv(&self) -> Result<CoordinatorMessage, WorkerError> {
        let frame = self.transport.recv_frame().await?;
        let outgoing: CoordinatorOutgoing = self.codec.decode(&frame)?;

        match outgoing {
            CoordinatorOutgoing::Worker(msg) => Ok(msg),
            CoordinatorOutgoing::Cli(_) => Err(WorkerError::UnexpectedResponse(
                "Received CLI response on worker connection".to_string(),
            )),
        }
    }
}

impl std::fmt::Debug for CoordinatorClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoordinatorClient")
            .field("node_id", &self.node_id)
            .field("registered", &self.is_registered())
            .finish_non_exhaustive()
    }
}

pub async fn create_heartbeat_client(
    addr: &Address,
    node_id: NodeId,
) -> Result<CoordinatorClient, WorkerError> {
    CoordinatorClient::connect_for_node(addr, node_id).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn connect_fails_on_invalid_address() {
        let addr = Address::tcp(std::net::SocketAddr::from(([0, 0, 0, 0], 0)));
        let result = CoordinatorClient::connect(&addr).await;

        assert!(result.is_err());
    }
}
