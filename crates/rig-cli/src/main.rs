use std::path::PathBuf;

use clap::{Parser, Subcommand};
use rig_core::RigConfig;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

mod client;
mod commands;
mod config;

use config::RigConfigLoader;
mod output;
mod runtime;

use commands::{
    benchmark::BenchmarkArgs, config::ConfigArgs, coordinator::CoordinatorArgs, demo::DemoArgs,
    generate::GenerateArgs, inference::InferenceArgs, pipeline::PipelineArgs, status::StatusArgs,
    worker::WorkerArgs,
};

#[derive(Debug, Parser)]
#[command(name = "rig", version, about, long_about = None)]
struct Cli {
    /// Path to config file (default: rig.toml or ~/.config/rig/config.toml).
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Set log level.
    #[arg(long, global = true, env = "RIG_LOG")]
    log_level: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Start a demo cluster and chat interactively.
    #[command(name = "demo")]
    Demo(DemoArgs),

    /// Run a single inference without interactive mode.
    #[command(name = "inference")]
    Inference(InferenceArgs),

    /// Start the coordinator server.
    #[command(name = "coordinator")]
    Coordinator(CoordinatorArgs),

    /// Start a worker node.
    #[command(name = "worker")]
    Worker(WorkerArgs),

    /// Submit an inference request.
    #[command(name = "generate")]
    Generate(GenerateArgs),

    /// Check cluster status.
    #[command(name = "status")]
    Status(StatusArgs),

    /// Manage pipelines.
    #[command(name = "pipeline")]
    Pipeline(PipelineArgs),

    /// Run inference benchmarks.
    #[command(name = "bench")]
    Benchmark(BenchmarkArgs),

    /// Configuration management (generate, show, validate).
    #[command(name = "config")]
    Config(ConfigArgs),
}

fn setup_logging(log_level: &str) {
    use std::io::IsTerminal;

    let env_filter = EnvFilter::try_new(log_level).unwrap_or_else(|_| EnvFilter::new("info"));

    let is_terminal = std::io::stdout().is_terminal();

    tracing_subscriber::registry()
        .with(fmt::layer().with_ansi(is_terminal).with_target(true))
        .with(env_filter)
        .init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let rig_config = RigConfig::load(cli.config.as_deref())?;

    let log_level = cli
        .log_level
        .as_deref()
        .unwrap_or(&rig_config.logging.level);
    setup_logging(log_level);

    match cli.command {
        Commands::Config(args) => commands::run_config(&args),
        Commands::Demo(args) => commands::run_demo(args, &rig_config).await,
        Commands::Inference(args) => commands::run_inference(args).await,
        Commands::Coordinator(args) => commands::run_coordinator(args, &rig_config).await,
        Commands::Worker(args) => commands::run_worker(args, &rig_config).await,
        Commands::Generate(args) => commands::run_generate(args, &rig_config).await,
        Commands::Status(args) => commands::run_status(args).await,
        Commands::Pipeline(args) => commands::run_pipeline(args).await,
        Commands::Benchmark(args) => commands::run_benchmark(args).await,
    }
}
