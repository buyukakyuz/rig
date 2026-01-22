use std::io::Write;
use std::net::SocketAddr;

use anyhow::{Context, Result};
use clap::Args;
use rig_core::{GenerationParams, InferenceInput, PipelineId};

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
    #[arg(long, env = "RIG_COORDINATOR_ADDR", default_value = "127.0.0.1:50051")]
    pub coordinator: String,

    /// Maximum tokens to generate.
    #[arg(long, default_value = "256")]
    pub max_tokens: usize,

    /// Sampling temperature.
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,

    /// Top-p sampling.
    #[arg(long, default_value = "0.9")]
    pub top_p: f32,

    /// Top-k sampling.
    #[arg(long, default_value = "40")]
    pub top_k: usize,

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
    #[arg(long, default_value = "60")]
    pub timeout: u64,

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

fn build_generation_params(args: &GenerateArgs) -> GenerationParams {
    let mut params = GenerationParams::new()
        .with_max_tokens(args.max_tokens)
        .with_temperature(args.temperature)
        .with_top_p(args.top_p)
        .with_top_k(args.top_k);

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

pub async fn run_generate(args: GenerateArgs) -> Result<()> {
    let coordinator_addr: SocketAddr = args
        .coordinator
        .parse()
        .context("Invalid coordinator address")?;

    let pipeline_id: PipelineId = args
        .pipeline
        .parse()
        .context("Invalid pipeline ID (expected UUID)")?;

    let input = parse_input(&args)?;
    let params = build_generation_params(&args);
    let timeout_ms = args.timeout * 1000;

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
