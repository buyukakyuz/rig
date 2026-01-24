use std::collections::HashMap;
use std::io::{self, BufRead, Write as IoWrite};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Args;
use rig_coordinator::{CoordinatorConfig, CoordinatorServer, HeartbeatMonitor};
use rig_core::types::protocol::CliCreatePipelineAutoRequest;
use rig_core::{Address, GenerationParams, InferenceInput, ModelId, PipelineId, RigConfig};
use rig_worker::{WorkerConfig, WorkerNode};
use tokio::sync::broadcast;

use crate::client::CliClient;

#[derive(Debug, Args)]
pub struct DemoArgs {
    /// Path to the model directory.
    #[arg(short, long)]
    pub model: PathBuf,

    /// Number of workers.
    #[arg(short = 'n', long, default_value = "2")]
    pub workers: usize,

    /// Device: "cpu", "metal", "cuda", or "auto".
    #[arg(long, env = "RIG_DEVICE")]
    pub device: Option<String>,

    /// System prompt for the chat.
    #[arg(long, default_value = "You are a helpful assistant.")]
    pub system_prompt: String,

    /// Maximum tokens to generate per response.
    #[arg(long)]
    pub max_tokens: Option<usize>,

    /// Temperature for sampling.
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Top-p sampling threshold.
    #[arg(long)]
    pub top_p: Option<f32>,

    /// Random seed for reproducible generation.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Coordinator port.
    #[arg(long, default_value = "50051")]
    pub coordinator_port: u16,

    /// Base port for workers (workers use consecutive ports).
    #[arg(long, default_value = "5001")]
    pub worker_base_port: u16,

    /// Start cluster only, don't enter interactive chat.
    #[arg(long)]
    pub no_chat: bool,
}

fn get_model_name(path: &Path) -> Result<String> {
    let config_path = path.join("config.json");
    if !config_path.exists() {
        anyhow::bail!(
            "Model config not found: {}\nEnsure the path points to a directory containing config.json",
            config_path.display()
        );
    }

    path.file_name()
        .and_then(|n| n.to_str())
        .map(String::from)
        .ok_or_else(|| {
            anyhow::anyhow!("Could not extract model name from path: {}", path.display())
        })
}

struct DemoCluster {
    coordinator_addr: SocketAddr,
    shutdown_tx: broadcast::Sender<()>,
    pipeline_id: Option<PipelineId>,
}

impl DemoCluster {
    async fn start(
        model_name: &str,
        model_path: &Path,
        args: &DemoArgs,
        config: &RigConfig,
    ) -> Result<Self> {
        let (shutdown_tx, _) = broadcast::channel::<()>(1);

        let coordinator_addr: SocketAddr =
            format!("127.0.0.1:{}", args.coordinator_port).parse()?;

        let coord_config = CoordinatorConfig::default()
            .with_listen_addr(coordinator_addr)
            .with_heartbeat_interval(Duration::from_secs(
                config.coordinator.heartbeat_interval_secs,
            ))
            .with_heartbeat_timeout(Duration::from_secs(
                config.coordinator.heartbeat_timeout_secs,
            ));

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

        tokio::time::sleep(Duration::from_millis(200)).await;
        println!("  Coordinator: {coordinator_addr}");

        let model_id = ModelId::new(model_name, "v1");
        let mut model_paths = HashMap::new();
        model_paths.insert(model_id.clone(), model_path.to_path_buf());

        let device = args.device.as_deref().unwrap_or(&config.runtime.device);

        for i in 0..args.workers {
            #[allow(clippy::cast_possible_truncation)]
            let worker_port = args.worker_base_port + i as u16;
            let worker_addr: SocketAddr = format!("127.0.0.1:{worker_port}").parse()?;

            let worker_config = WorkerConfig::default()
                .with_coordinator_addr(Address::tcp(coordinator_addr))
                .with_listen_addr(worker_addr)
                .with_heartbeat_interval(Duration::from_secs(config.worker.heartbeat_interval_secs))
                .with_model_paths(model_paths.clone());

            let runtime = crate::runtime::create_runtime(device)?;

            let mut worker = WorkerNode::new(worker_config, runtime);
            let worker_shutdown_rx = shutdown_tx.subscribe();
            let worker_model_id = model_id.clone();
            let worker_num = i + 1;

            tokio::spawn(async move {
                let mut shutdown_rx = worker_shutdown_rx;
                tokio::select! {
                    result = worker.run(worker_model_id) => {
                        if let Err(e) = result {
                            tracing::error!(error = %e, "Worker {worker_num} error");
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        worker.shutdown();
                    }
                }
            });

            println!("  Worker {}: {worker_addr}", i + 1);
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
                tokio::time::sleep(Duration::from_millis(500)).await;
                return Ok(());
            }

            if start.elapsed() > timeout {
                anyhow::bail!("Timeout waiting for workers: expected {expected}, got {count}");
            }

            tokio::time::sleep(Duration::from_millis(200)).await;
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

            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }
}

struct GenerationConfig {
    system_prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: Option<u64>,
}

async fn run_interactive_chat(
    coordinator_addr: SocketAddr,
    pipeline_id: PipelineId,
    config: &GenerationConfig,
) -> Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    println!();
    println!("========================================");
    println!("  Rig Demo - Interactive Chat");
    println!("========================================");
    println!();
    println!("Type your message and press Enter.");
    println!("Commands: /quit to exit, /status to check cluster");
    println!();

    loop {
        print!("You> ");
        stdout.flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            break;
        }

        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input.starts_with('/') {
            match input.to_lowercase().as_str() {
                "/quit" | "/exit" | "/q" => {
                    println!("Goodbye!");
                    break;
                }
                "/status" => {
                    let client = CliClient::connect(coordinator_addr).await?;
                    let status = client.get_status().await?;
                    println!();
                    println!(
                        "Nodes: {}/{} healthy",
                        status.healthy_nodes, status.total_nodes
                    );
                    println!(
                        "Pipelines: {}/{} ready",
                        status.ready_pipelines, status.total_pipelines
                    );
                    println!();
                    continue;
                }
                _ => {
                    println!("Unknown command: {input}");
                    println!("Available: /quit, /status");
                    continue;
                }
            }
        }

        print!("\nAssistant> ");
        stdout.flush()?;

        let client = CliClient::connect(coordinator_addr).await?;

        let mut params = GenerationParams::new()
            .with_max_tokens(config.max_tokens)
            .with_temperature(config.temperature)
            .with_top_p(config.top_p)
            .with_system_prompt(&config.system_prompt)
            .with_chat_template(true);

        if let Some(seed) = config.seed {
            params = params.with_seed(seed);
        }

        let result = client
            .generate(
                pipeline_id,
                InferenceInput::text(input),
                params,
                Some(120_000),
                |token| {
                    print!("{token}");
                    let _ = io::stdout().flush();
                },
            )
            .await;

        println!("\n");

        if let Err(e) = result {
            eprintln!("Error: {e}");
        }
    }

    Ok(())
}

pub async fn run_demo(args: DemoArgs, config: &RigConfig) -> Result<()> {
    let model_name = get_model_name(&args.model)?;

    let device = args.device.as_deref().unwrap_or(&config.runtime.device);
    let max_tokens = args.max_tokens.unwrap_or(config.generation.max_tokens);
    let temperature = args.temperature.unwrap_or(config.generation.temperature);
    let top_p = args.top_p.unwrap_or(config.generation.top_p);

    println!();
    println!("Starting Rig Demo");
    println!("=================");
    println!("Model: {} ({})", model_name, args.model.display());
    println!("Workers: {}", args.workers);
    println!("Device: {device}");
    println!();

    println!("Starting coordinator and workers...");
    let mut cluster = DemoCluster::start(&model_name, &args.model, &args, config).await?;

    print!("Creating pipeline... ");
    io::stdout().flush()?;
    cluster.create_pipeline(&model_name, args.workers).await?;
    println!("done");

    let pipeline_id = cluster
        .pipeline_id
        .ok_or_else(|| anyhow::anyhow!("Pipeline was not created"))?;
    println!("Pipeline ID: {pipeline_id}");
    println!();
    println!("Cluster ready!");

    if args.no_chat {
        println!();
        println!("Cluster is running. Press Ctrl+C to stop.");
        println!();
        println!("In another terminal, you can run:");
        println!(
            "  rig generate --coordinator 127.0.0.1:{} --pipeline {pipeline_id} --chat --text \"Hello\"",
            args.coordinator_port
        );

        tokio::signal::ctrl_c().await?;
    } else {
        let gen_config = GenerationConfig {
            system_prompt: args.system_prompt.clone(),
            max_tokens,
            temperature,
            top_p,
            seed: args.seed,
        };
        tokio::select! {
            result = run_interactive_chat(cluster.coordinator_addr, pipeline_id, &gen_config) => {
                if let Err(e) = result {
                    eprintln!("Chat error: {e}");
                }
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\nInterrupted");
            }
        }
    }

    println!("\nShutting down...");
    cluster.shutdown();
    tokio::time::sleep(Duration::from_millis(300)).await;
    println!("Done.");

    Ok(())
}
