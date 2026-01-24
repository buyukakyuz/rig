use std::io::Write;
use std::net::SocketAddr;

use anyhow::{Context, Result};
use clap::Args;
use rig_core::{GenerationParams, InferenceInput, PipelineId, RigConfig};

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
pub struct GenerateArgs {
    /// Pipeline ID to submit to.
    #[arg(short, long)]
    pub pipeline: String,

    /// Text prompt to generate from.
    #[arg(short, long, conflicts_with = "tokens")]
    pub text: Option<String>,

    /// Token IDs to generate from (comma-separated).
    #[arg(long)]
    pub tokens: Option<String>,

    /// Coordinator address.
    #[arg(long, env = "RIG_COORDINATOR_ADDR")]
    pub coordinator: Option<String>,

    /// Maximum tokens to generate.
    #[arg(long)]
    pub max_tokens: Option<usize>,

    /// Sampling temperature.
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Top-p sampling.
    #[arg(long)]
    pub top_p: Option<f32>,

    /// Top-k sampling.
    #[arg(long)]
    pub top_k: Option<usize>,

    /// Stop sequences (comma-separated).
    #[arg(long)]
    pub stop: Option<String>,

    /// System prompt to prepend to input.
    #[arg(long)]
    pub system_prompt: Option<String>,

    /// Use the model's chat template for formatting.
    /// When enabled, system_prompt becomes a system message and text becomes a user message.
    #[arg(long)]
    pub chat: bool,

    /// Random seed for reproducible generation.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Request timeout in seconds.
    #[arg(long)]
    pub timeout: Option<u64>,

    /// Output format.
    #[arg(short, long, value_enum, default_value = "text")]
    pub format: OutputFormat,
}

fn parse_input(args: &GenerateArgs) -> Result<InferenceInput> {
    if let Some(text) = &args.text {
        Ok(InferenceInput::text(text))
    } else if let Some(tokens_str) = &args.tokens {
        let tokens: Vec<u32> = tokens_str
            .split(',')
            .map(|s| s.trim().parse::<u32>())
            .collect::<Result<Vec<_>, _>>()
            .context("Invalid token IDs")?;
        Ok(InferenceInput::tokens(tokens))
    } else {
        anyhow::bail!("Either --text or --tokens must be provided")
    }
}

fn build_generation_params(args: &GenerateArgs, config: &RigConfig) -> GenerationParams {
    let max_tokens = args.max_tokens.unwrap_or(config.generation.max_tokens);
    let temperature = args.temperature.unwrap_or(config.generation.temperature);
    let top_p = args.top_p.unwrap_or(config.generation.top_p);
    let top_k = args.top_k.unwrap_or(config.generation.top_k);

    let mut params = GenerationParams::new()
        .with_max_tokens(max_tokens)
        .with_temperature(temperature)
        .with_top_p(top_p)
        .with_top_k(top_k);

    if let Some(stop) = &args.stop {
        for seq in stop.split(',') {
            params = params.with_stop_sequence(seq.trim());
        }
    }

    if let Some(sp) = &args.system_prompt {
        params = params.with_system_prompt(sp);
    }

    if args.chat {
        params = params.with_chat_template(true);
    }

    if let Some(seed) = args.seed {
        params = params.with_seed(seed);
    }

    params
}

pub async fn run_generate(args: GenerateArgs, config: &RigConfig) -> Result<()> {
    let coordinator_str = args
        .coordinator
        .as_deref()
        .unwrap_or(&config.worker.coordinator_addr);
    let coordinator_addr: SocketAddr = coordinator_str
        .parse()
        .context("Invalid coordinator address")?;

    let pipeline_id: PipelineId = args
        .pipeline
        .parse()
        .context("Invalid pipeline ID (expected UUID)")?;

    let timeout_secs = args.timeout.unwrap_or(config.generation.timeout_secs);

    let input = parse_input(&args)?;
    let params = build_generation_params(&args, config);
    let timeout_ms = timeout_secs * 1000;

    tracing::debug!(
        coordinator = %coordinator_addr,
        pipeline = %pipeline_id,
        max_tokens = params.max_tokens,
        temperature = params.temperature,
        "Submitting generation request"
    );

    let client = CliClient::connect(coordinator_addr).await?;

    let mut output_text = String::new();
    let is_json = matches!(args.format, OutputFormat::Json);

    let usage = client
        .generate(pipeline_id, input, params, Some(timeout_ms), |token_text| {
            if is_json {
                output_text.push_str(token_text);
            } else {
                print!("{token_text}");
                let _ = std::io::stdout().flush();
            }
        })
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

    Ok(())
}
