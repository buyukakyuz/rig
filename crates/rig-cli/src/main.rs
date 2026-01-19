use clap::{Parser, Subcommand};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

mod client;
mod commands;
mod output;

use commands::{
    coordinator::CoordinatorArgs, generate::GenerateArgs, pipeline::PipelineArgs,
    status::StatusArgs, worker::WorkerArgs,
};

#[derive(Debug, Parser)]
#[command(name = "rig", version, about, long_about = None)]
struct Cli {
    /// Set log level.
    #[arg(long, global = true, env = "RIG_LOG", default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
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

    setup_logging(&cli.log_level);

    match cli.command {
        Commands::Coordinator(args) => commands::run_coordinator(args).await,
        Commands::Worker(args) => commands::run_worker(args).await,
        Commands::Generate(args) => commands::run_generate(args).await,
        Commands::Status(args) => commands::run_status(args).await,
        Commands::Pipeline(args) => commands::run_pipeline(args).await,
    }
}
