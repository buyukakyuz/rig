use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use rig_core::{CliCreatePipelineAutoRequest, DType, ModelId, NodeId, PipelineConfig, PipelineId};

use crate::client::CliClient;

#[derive(Debug, Args)]
pub struct PipelineArgs {
    /// Pipeline subcommand.
    #[command(subcommand)]
    pub command: PipelineCommand,

    /// Coordinator address.
    #[arg(
        long,
        env = "RIG_COORDINATOR_ADDR",
        global = true,
        default_value = "127.0.0.1:50051"
    )]
    pub coordinator: String,
}

#[derive(Debug, Subcommand)]
pub enum PipelineCommand {
    /// Create a new pipeline.
    Create(CreatePipelineArgs),
    /// Get information about a pipeline.
    Get {
        /// Pipeline ID.
        pipeline_id: String,
    },
    /// List all pipelines.
    List,
}

#[derive(Debug, Args)]
pub struct CreatePipelineArgs {
    /// Pipeline ID (auto-generated if not provided).
    #[arg(long = "pipeline")]
    pub pipeline_id: Option<String>,

    /// Model name.
    #[arg(long)]
    pub model_name: String,

    /// Model version.
    #[arg(long, default_value = "v1")]
    pub model_version: String,

    /// Number of stages to use (auto-partitioning mode).
    /// If not specified, uses all available nodes with the model.
    #[arg(long)]
    pub stages: Option<usize>,

    /// Data type (f32, f16, bf16, i8, i4).
    #[arg(long, default_value = "f16")]
    pub dtype: String,

    /// Path to the model file (manual mode).
    #[arg(long)]
    pub model_path: Option<PathBuf>,

    /// Number of model layers (manual mode).
    #[arg(long)]
    pub num_layers: Option<usize>,

    /// Stage assignments in format "node_id:start-end" (manual mode, can be repeated).
    #[arg(long = "stage", value_name = "NODE_ID:START-END")]
    pub manual_stages: Vec<String>,
}

fn parse_stage_spec(spec: &str) -> Result<(NodeId, usize, usize)> {
    let (node_id_str, range_str) = spec
        .split_once(':')
        .context("Stage spec must be in format 'node_id:start-end'")?;

    let node_id: NodeId = node_id_str
        .parse()
        .context("Invalid node ID (expected UUID)")?;

    let (start_str, end_str) = range_str
        .split_once('-')
        .context("Layer range must be in format 'start-end'")?;

    let start: usize = start_str.parse().context("Invalid start layer")?;
    let end: usize = end_str.parse().context("Invalid end layer")?;

    Ok((node_id, start, end))
}

fn parse_dtype(s: &str) -> Result<DType> {
    match s.to_lowercase().as_str() {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        "i8" => Ok(DType::I8),
        "i4" => Ok(DType::I4),
        _ => anyhow::bail!("Unknown dtype: {s}. Valid options: f32, f16, bf16, i8, i4"),
    }
}

pub async fn run_pipeline(args: PipelineArgs) -> Result<()> {
    let coordinator_addr: SocketAddr = args
        .coordinator
        .parse()
        .context("Invalid coordinator address")?;

    let client = CliClient::connect(coordinator_addr).await?;

    match args.command {
        PipelineCommand::Create(create_args) => run_pipeline_create(&client, create_args).await,
        PipelineCommand::Get { pipeline_id } => run_pipeline_get(&client, &pipeline_id).await,
        PipelineCommand::List => run_pipeline_list(&client).await,
    }
}

async fn run_pipeline_create(client: &CliClient, args: CreatePipelineArgs) -> Result<()> {
    let dtype = parse_dtype(&args.dtype)?;

    let pipeline_id = args
        .pipeline_id
        .map(|s| s.parse::<PipelineId>())
        .transpose()
        .context("Invalid pipeline ID (expected UUID)")?;

    let is_manual_mode = !args.manual_stages.is_empty();
    tracing::info!("Creating pipeline");

    if is_manual_mode {
        let model_path = args
            .model_path
            .as_ref()
            .context("--model-path is required in manual mode (when using --stage)")?;

        let num_layers = args
            .num_layers
            .context("--num-layers is required in manual mode (when using --stage)")?;

        let model_id = ModelId::new(&args.model_name, &args.model_version);

        let mut assignments = Vec::new();
        for spec in &args.manual_stages {
            let (node_id, start, end) = parse_stage_spec(spec)?;
            assignments.push((node_id, start, end));
        }

        let config = PipelineConfig::new(model_id.clone(), model_path, num_layers, dtype);

        let created_pipeline_id = client
            .create_pipeline(config, assignments, pipeline_id)
            .await?;

        println!("Pipeline created: {created_pipeline_id}");
    } else {
        let mut req = CliCreatePipelineAutoRequest::new(&args.model_name)
            .with_version(&args.model_version)
            .with_dtype(dtype);

        if let Some(num_stages) = args.stages {
            req = req.with_stages(num_stages);
        }

        if let Some(pid) = pipeline_id {
            req = req.with_pipeline_id(pid);
        }

        let created_pipeline_id = client.create_pipeline_auto(req).await?;

        println!("Pipeline created: {created_pipeline_id}");
    }

    Ok(())
}

async fn run_pipeline_get(client: &CliClient, pipeline_id_str: &str) -> Result<()> {
    let pipeline_id: PipelineId = pipeline_id_str
        .parse()
        .context("Invalid pipeline ID (expected UUID)")?;

    let info = client.get_pipeline(pipeline_id).await?;

    println!("Pipeline: {}", info.pipeline_id);
    println!("Model:    {}:{}", info.model_id.name, info.model_id.version);
    println!("Status:   {}", info.status);
    println!();
    println!("Stages:");
    for stage in &info.stages {
        let ready = if stage.ready { "ready" } else { "pending" };
        println!(
            "  Stage {}: layers {}-{} on {} ({})",
            stage.stage_id, stage.layer_start, stage.layer_end, stage.node_id, ready
        );
    }

    Ok(())
}

async fn run_pipeline_list(client: &CliClient) -> Result<()> {
    let pipelines = client.list_pipelines().await?;

    if pipelines.is_empty() {
        println!("No pipelines found.");
        return Ok(());
    }

    println!("Pipelines:");
    println!("-----------");
    for pipeline in &pipelines {
        println!(
            "{} - {}:{} [{}] ({} stages)",
            pipeline.pipeline_id,
            pipeline.model_id.name,
            pipeline.model_id.version,
            pipeline.status,
            pipeline.stages.len()
        );
    }

    Ok(())
}
