use std::net::SocketAddr;

use anyhow::{Context, Result};
use rig_core::{
    Codec, CoordinatorIncoming, CoordinatorOutgoing, FramedTransport, GenerationParams,
    InferenceInput, PipelineConfig, PipelineId, TransportFactory, UsageStats,
    types::protocol::{
        CliCreatePipelineAutoRequest, CliCreatePipelineRequest, CliMessage, CliResponse,
        CliSubmitRequest, ClusterStatusResponse, PipelineInfoResponse,
    },
};
use rig_message_bincode::BincodeCodec;
use rig_transport_tcp::{TcpTransport, TcpTransportFactory};

pub struct CliClient {
    transport: TcpTransport,
    codec: BincodeCodec,
}

impl CliClient {
    pub async fn connect(addr: SocketAddr) -> Result<Self> {
        let factory = TcpTransportFactory::new();
        let address = rig_core::Address::tcp(addr);
        let transport = factory
            .connect(&address)
            .await
            .context("Failed to connect to coordinator")?;

        Ok(Self {
            transport,
            codec: BincodeCodec::new(),
        })
    }

    pub async fn get_status(&self) -> Result<ClusterStatusResponse> {
        let msg = CliMessage::GetStatus;
        let response = self.send_and_recv(msg).await?;

        match response {
            CliResponse::Status(status) => Ok(status),
            CliResponse::Error { code, message } => {
                anyhow::bail!("Coordinator error ({code}): {message}")
            }
            _ => anyhow::bail!("Unexpected response type"),
        }
    }

    pub async fn generate<F>(
        &self,
        pipeline_id: PipelineId,
        input: InferenceInput,
        params: GenerationParams,
        timeout_ms: Option<u64>,
        mut on_token: F,
    ) -> Result<UsageStats>
    where
        F: FnMut(&str),
    {
        let msg = CliMessage::SubmitRequest(CliSubmitRequest {
            pipeline_id,
            input,
            params,
            timeout_ms,
        });

        let incoming = CoordinatorIncoming::Cli(msg);
        let frame = self
            .codec
            .encode(&incoming)
            .context("Failed to encode message")?;
        self.transport
            .send_frame(&frame)
            .await
            .context("Failed to send message")?;

        loop {
            let response_frame = self
                .transport
                .recv_frame()
                .await
                .context("Failed to receive response")?;
            let outgoing: CoordinatorOutgoing = self
                .codec
                .decode(&response_frame)
                .context("Failed to decode response")?;

            match outgoing {
                CoordinatorOutgoing::Cli(CliResponse::StreamStart { .. }) => {}
                CoordinatorOutgoing::Cli(CliResponse::Token { token_text, .. }) => {
                    on_token(&token_text);
                }
                CoordinatorOutgoing::Cli(CliResponse::StreamComplete { usage, .. }) => {
                    return Ok(usage);
                }
                CoordinatorOutgoing::Cli(CliResponse::Error { code, message }) => {
                    anyhow::bail!("Coordinator error ({code}): {message}")
                }
                _ => anyhow::bail!("Unexpected response type during streaming"),
            }
        }
    }

    pub async fn create_pipeline(
        &self,
        config: PipelineConfig,
        assignments: Vec<(rig_core::NodeId, usize, usize)>,
        pipeline_id: Option<PipelineId>,
    ) -> Result<PipelineId> {
        let mut request = CliCreatePipelineRequest::new(config, assignments);
        if let Some(id) = pipeline_id {
            request = request.with_pipeline_id(id);
        }
        let msg = CliMessage::CreatePipeline(request);
        let response = self.send_and_recv(msg).await?;

        match response {
            CliResponse::PipelineCreated { pipeline_id } => Ok(pipeline_id),
            CliResponse::Error { code, message } => {
                anyhow::bail!("Coordinator error ({code}): {message}")
            }
            _ => anyhow::bail!("Unexpected response type"),
        }
    }

    pub async fn create_pipeline_auto(
        &self,
        request: CliCreatePipelineAutoRequest,
    ) -> Result<PipelineId> {
        let msg = CliMessage::CreatePipelineAuto(request);
        let response = self.send_and_recv(msg).await?;

        match response {
            CliResponse::PipelineCreated { pipeline_id } => Ok(pipeline_id),
            CliResponse::Error { code, message } => {
                anyhow::bail!("Coordinator error ({code}): {message}")
            }
            _ => anyhow::bail!("Unexpected response type"),
        }
    }

    pub async fn get_pipeline(&self, pipeline_id: PipelineId) -> Result<PipelineInfoResponse> {
        let msg = CliMessage::GetPipeline { pipeline_id };
        let response = self.send_and_recv(msg).await?;

        match response {
            CliResponse::PipelineInfo(info) => Ok(info),
            CliResponse::Error { code, message } => {
                anyhow::bail!("Coordinator error ({code}): {message}")
            }
            _ => anyhow::bail!("Unexpected response type"),
        }
    }

    pub async fn list_pipelines(&self) -> Result<Vec<PipelineInfoResponse>> {
        let msg = CliMessage::ListPipelines;
        let response = self.send_and_recv(msg).await?;

        match response {
            CliResponse::PipelineList(pipelines) => Ok(pipelines),
            CliResponse::Error { code, message } => {
                anyhow::bail!("Coordinator error ({code}): {message}")
            }
            _ => anyhow::bail!("Unexpected response type"),
        }
    }

    async fn send_and_recv(&self, msg: CliMessage) -> Result<CliResponse> {
        let incoming = CoordinatorIncoming::Cli(msg);
        let frame = self
            .codec
            .encode(&incoming)
            .context("Failed to encode message")?;
        self.transport
            .send_frame(&frame)
            .await
            .context("Failed to send message")?;

        let response_frame = self
            .transport
            .recv_frame()
            .await
            .context("Failed to receive response")?;
        let outgoing: CoordinatorOutgoing = self
            .codec
            .decode(&response_frame)
            .context("Failed to decode response")?;

        match outgoing {
            CoordinatorOutgoing::Cli(response) => Ok(response),
            CoordinatorOutgoing::Worker(_) => {
                anyhow::bail!("Received worker response on CLI connection")
            }
        }
    }
}

impl std::fmt::Debug for CliClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CliClient").finish_non_exhaustive()
    }
}
