use std::collections::HashMap;
use std::io::Write as IoWrite;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Args;
use rig_coordinator::{CoordinatorConfig, CoordinatorServer, HeartbeatMonitor};
use rig_core::types::protocol::CliCreatePipelineAutoRequest;
use rig_core::{Address, GenerationParams, InferenceInput, ModelId, PipelineId};
use rig_message_bincode::BincodeCodec;
use rig_transport_tcp::{TcpConfig, TcpTransportFactory};
use rig_worker::{WorkerConfig, WorkerNode};
use tokio::sync::broadcast;

use crate::client::CliClient;
use crate::output::{GenerateContent, GenerateOutput};

#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum OutputFormat {
    /// Plain text output.
    #[default]
    Text,
    /// JSON output with full details.
    Json,
}

#[derive(Debug, Args)]
pub struct InferenceArgs {
    /// Path to the model directory.
    #[arg(short, long)]
    pub model: PathBuf,

    /// The prompt to run inference on.
    #[arg(required = true)]
    pub prompt: String,

    /// Number of workers (stages in the pipeline).
    #[arg(short = 'n', long, default_value = "1")]
    pub workers: usize,

    /// Device: "cpu", "metal", "cuda", or "auto".
    #[arg(long, env = "RIG_DEVICE", default_value = "auto")]
    pub device: String,

    /// System prompt.
    #[arg(long)]
    pub system_prompt: Option<String>,

    /// Maximum tokens to generate.
    #[arg(long, default_value = "256")]
    pub max_tokens: usize,

    /// Temperature for sampling.
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,

    /// Top-p sampling threshold.
    #[arg(long, default_value = "0.9")]
    pub top_p: f32,

    /// Top-k sampling.
    #[arg(long, default_value = "40")]
    pub top_k: usize,

    /// Stop sequences (comma-separated).
    #[arg(long)]
    pub stop: Option<String>,

    /// Random seed for reproducible generation.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Disable chat template.
    #[arg(long)]
    pub no_chat: bool,

    /// Request timeout in seconds.
    #[arg(long, default_value = "300")]
    pub timeout: u64,

    /// Output format.
    #[arg(short, long, value_enum, default_value = "text")]
    pub format: OutputFormat,

    /// Quiet mode.
    #[arg(short, long)]
    pub quiet: bool,

    /// Coordinator port.
    #[arg(long, default_value = "50052")]
    pub coordinator_port: u16,

    /// Base port for workers.
    #[arg(long, default_value = "5101")]
    pub worker_base_port: u16,
}

fn get_model_name(path: &Path) -> Result<String> {
    let config_path = path.join("config.json");
    let is_gguf = path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));

    if !config_path.exists() && !is_gguf {
        anyhow::bail!(
            "Model not found: {}\nEnsure the path points to a directory containing config.json or a .gguf file",
            path.display()
        );
    }

    path.file_name()
        .and_then(|n| n.to_str())
        .map(|s| {
            s.strip_suffix(".gguf")
                .or_else(|| s.strip_suffix(".GGUF"))
                .unwrap_or(s)
                .to_string()
        })
        .ok_or_else(|| {
            anyhow::anyhow!("Could not extract model name from path: {}", path.display())
        })
}

struct LocalCluster {
    coordinator_addr: SocketAddr,
    shutdown_tx: broadcast::Sender<()>,
    pipeline_id: Option<PipelineId>,
}

impl LocalCluster {
    async fn start(model_name: &str, model_path: &Path, args: &InferenceArgs) -> Result<Self> {
        let (shutdown_tx, _) = broadcast::channel::<()>(1);

        let coordinator_addr: SocketAddr =
            format!("127.0.0.1:{}", args.coordinator_port).parse()?;

        let coord_config = CoordinatorConfig::default()
            .with_listen_addr(coordinator_addr)
            .with_heartbeat_interval(Duration::from_secs(30))
            .with_heartbeat_timeout(Duration::from_secs(60));

        let mut coordinator = CoordinatorServer::new(coord_config.clone());
        let coord_state = coordinator.state();

        let heartbeat_monitor = HeartbeatMonitor::new(
            coord_state.clone(),
            coord_config.heartbeat_timeout,
            coord_config.heartbeat_check_interval,
            coordinator.shutdown_receiver(),
        );
        tokio::spawn(heartbeat_monitor.run());

        let coord_shutdown_rx = shutdown_tx.subscribe();
        tokio::spawn(async move {
            let mut shutdown_rx = coord_shutdown_rx;
            tokio::select! {
                result = coordinator.run() => {
                    if let Err(e) = result {
                        tracing::error!(error = %e, "Coordinator error");
                    }
                }
                _ = shutdown_rx.recv() => {
                    coordinator.shutdown();
                }
            }
        });

        tokio::time::sleep(Duration::from_millis(100)).await;

        let model_id = ModelId::new(model_name, "v1");
        let mut model_paths = HashMap::new();
        model_paths.insert(model_id.clone(), model_path.to_path_buf());

        for i in 0..args.workers {
            #[allow(clippy::cast_possible_truncation)]
            let worker_port = args.worker_base_port + i as u16;
            let worker_addr: SocketAddr = format!("127.0.0.1:{worker_port}").parse()?;

            let worker_config = WorkerConfig::default()
                .with_coordinator_addr(Address::tcp(coordinator_addr))
                .with_listen_addr(worker_addr)
                .with_heartbeat_interval(Duration::from_secs(30))
                .with_model_paths(model_paths.clone());

            let runtime = crate::runtime::create_runtime(&args.device)?;

            let tcp_config = TcpConfig::default().with_read_timeout(None);
            let transport_factory = TcpTransportFactory::with_config(tcp_config);
            let codec = BincodeCodec::new();

            let mut worker = WorkerNode::new(worker_config, runtime, transport_factory, codec);
            let worker_shutdown_rx = shutdown_tx.subscribe();
            let worker_model_id = model_id.clone();

            tokio::spawn(async move {
                let mut shutdown_rx = worker_shutdown_rx;
                tokio::select! {
                    result = worker.run(worker_model_id) => {
                        if let Err(e) = result {
                            tracing::error!(error = %e, "Worker error");
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        worker.shutdown();
                    }
                }
            });
        }

        Self::wait_for_workers(&coord_state, args.workers).await?;

        Ok(Self {
            coordinator_addr,
            shutdown_tx,
            pipeline_id: None,
        })
    }

    async fn wait_for_workers(
        state: &Arc<rig_coordinator::CoordinatorState>,
        expected: usize,
    ) -> Result<()> {
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(120);

        loop {
            let count = state.node_count().await;
            if count >= expected {
                tokio::time::sleep(Duration::from_millis(200)).await;
                return Ok(());
            }

            if start.elapsed() > timeout {
                anyhow::bail!("Timeout waiting for workers: expected {expected}, got {count}");
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn create_pipeline(&mut self, model_name: &str, stages: usize) -> Result<()> {
        let client = CliClient::connect(self.coordinator_addr).await?;

        let request = CliCreatePipelineAutoRequest::new(model_name)
            .with_version("v1")
            .with_stages(stages);

        let pipeline_id = client.create_pipeline_auto(request).await?;
        self.pipeline_id = Some(pipeline_id);
        self.wait_for_pipeline_ready().await?;

        Ok(())
    }

    async fn wait_for_pipeline_ready(&self) -> Result<()> {
        let pipeline_id = self
            .pipeline_id
            .ok_or_else(|| anyhow::anyhow!("No pipeline created"))?;

        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(120);

        loop {
            let client = CliClient::connect(self.coordinator_addr).await?;
            let info = client.get_pipeline(pipeline_id).await?;

            if info.status.eq_ignore_ascii_case("ready") {
                return Ok(());
            }

            if start.elapsed() > timeout {
                anyhow::bail!(
                    "Timeout waiting for pipeline to be ready (status: {})",
                    info.status
                );
            }

            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }
}

fn build_generation_params(args: &InferenceArgs) -> GenerationParams {
    let mut params = GenerationParams::new()
        .with_max_tokens(args.max_tokens)
        .with_temperature(args.temperature)
        .with_top_p(args.top_p)
        .with_top_k(args.top_k)
        .with_chat_template(!args.no_chat);

    if let Some(stop) = &args.stop {
        for seq in stop.split(',') {
            params = params.with_stop_sequence(seq.trim());
        }
    }

    if let Some(sp) = &args.system_prompt {
        params = params.with_system_prompt(sp);
    }

    if let Some(seed) = args.seed {
        params = params.with_seed(seed);
    }

    params
}

pub async fn run_inference(args: InferenceArgs) -> Result<()> {
    let model_name = get_model_name(&args.model)?;
    let quiet = args.quiet;
    let is_json = matches!(args.format, OutputFormat::Json);

    if !quiet && !is_json {
        eprintln!("Loading model: {} ({})", model_name, args.model.display());
    }

    let mut cluster = LocalCluster::start(&model_name, &args.model, &args).await?;

    if !quiet && !is_json {
        eprintln!("Creating pipeline...");
    }

    cluster.create_pipeline(&model_name, args.workers).await?;

    let pipeline_id = cluster
        .pipeline_id
        .ok_or_else(|| anyhow::anyhow!("Pipeline was not created"))?;

    if !quiet && !is_json {
        eprintln!("Running inference...");
        eprintln!();
    }

    let client = CliClient::connect(cluster.coordinator_addr).await?;
    let params = build_generation_params(&args);
    let timeout_ms = args.timeout * 1000;

    let mut output_text = String::new();

    let usage = client
        .generate(
            pipeline_id,
            InferenceInput::text(&args.prompt),
            params,
            Some(timeout_ms),
            |token_text| {
                if is_json {
                    output_text.push_str(token_text);
                } else {
                    print!("{token_text}");
                    let _ = std::io::stdout().flush();
                }
            },
        )
        .await?;

    if !is_json {
        println!();
    }

    if is_json {
        let output = GenerateOutput {
            output: GenerateContent::text(output_text),
            usage: usage.into(),
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
    }

    cluster.shutdown();
    tokio::time::sleep(Duration::from_millis(100)).await;

    Ok(())
}
