use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use tracing::info;

use rig_core::{
    Activation, ActivationMetadata, Assignment, DType, ModelId, Neighbors, PartitionSpec,
    PipelineId, RequestId, Runtime, SamplingParams, Shape, StageId, StopChecker, TensorData,
    UsageStats,
};
use rig_runtime_candle::{CandleRuntime, Device, memory::query_device_memory};
use rig_worker::stage::PipelineStage;

use crate::output::{
    AggregatedMetrics, BenchmarkMetadata, BenchmarkOutput, BenchmarkSummary, MemoryMetrics,
    SequenceLengthResult, SingleRunMetrics,
};

#[derive(Debug, Args)]
pub struct BenchmarkArgs {
    /// Path to the model directory.
    #[arg(short, long)]
    pub model: PathBuf,

    /// Device to use: "auto", "cpu", "metal", or "cuda".
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Comma-separated prompt sizes (in tokens) to benchmark.
    #[arg(long, default_value = "128,256,512")]
    pub prompt_sizes: String,

    /// Number of tokens to generate per benchmark run.
    #[arg(long, default_value = "64")]
    pub generation_length: usize,

    /// Number of benchmark runs per configuration.
    #[arg(long, default_value = "3")]
    pub runs: usize,

    /// Number of warmup runs before benchmarking.
    #[arg(long, default_value = "1")]
    pub warmup: usize,

    /// Temperature for sampling.
    #[arg(long, default_value = "0.0")]
    pub temperature: f32,

    /// Suppress progress output.
    #[arg(long)]
    pub quiet: bool,

    /// Save results to .benchmarks/<model>_<timestamp>.json
    #[arg(long)]
    pub save: bool,

    /// Save results to a specific file path.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

fn parse_prompt_sizes(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .with_context(|| format!("Invalid prompt size: {part}"))
        })
        .collect()
}

fn create_device(device_str: &str) -> Result<Device> {
    match device_str.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        #[cfg(feature = "metal")]
        "metal" => Device::new_metal(0).context("Failed to create Metal device"),
        #[cfg(not(feature = "metal"))]
        "metal" => anyhow::bail!("Metal support not compiled in"),
        #[cfg(feature = "cuda")]
        "cuda" => Device::new_cuda(0).context("Failed to create CUDA device"),
        #[cfg(not(feature = "cuda"))]
        "cuda" => anyhow::bail!("CUDA support not compiled in"),
        "auto" => {
            #[cfg(feature = "metal")]
            if let Ok(device) = Device::new_metal(0) {
                return Ok(device);
            }
            #[cfg(feature = "cuda")]
            if let Ok(device) = Device::new_cuda(0) {
                return Ok(device);
            }
            Ok(Device::Cpu)
        }
        other => anyhow::bail!("Unknown device: {other}"),
    }
}

#[allow(clippy::missing_const_for_fn)]
fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    }
}

fn generate_test_tokens(bos_token: u32, vocab_size: usize, target_length: usize) -> Vec<u32> {
    let mut tokens = vec![bos_token];

    let safe_start = 1000.min(vocab_size / 4);
    let safe_end = (vocab_size.saturating_sub(1000)).max(vocab_size * 3 / 4);
    let range = safe_end.saturating_sub(safe_start).max(1);

    for i in 0..(target_length.saturating_sub(1)) {
        #[allow(clippy::cast_possible_truncation)]
        let token = (safe_start + (i % range)) as u32;
        tokens.push(token);
    }

    tokens
}

fn create_benchmark_activation(request_id: RequestId, tokens: &[u32]) -> Activation {
    let seq_len = tokens.len();

    #[allow(clippy::cast_possible_truncation)]
    let positions: Vec<u32> = (0..seq_len).map(|i| i as u32).collect();

    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }

    let shape = Shape::new(vec![1, seq_len, 1]);
    let data = TensorData::cpu(bytes, DType::I8);

    let metadata = ActivationMetadata::new(request_id, 0, positions, true);

    Activation::new(data, shape, metadata)
}

fn create_decode_activation(request_id: RequestId, token: u32, position: usize) -> Activation {
    let bytes = token.to_le_bytes().to_vec();
    let data = TensorData::cpu(bytes, DType::I8);
    let shape = Shape::new(vec![1, 1, 1]);

    #[allow(clippy::cast_possible_truncation)]
    let position_u32 = position as u32;
    let metadata = ActivationMetadata::new(request_id, position_u32, vec![position_u32], false);

    Activation::new(data, shape, metadata)
}

fn create_standalone_assignment(num_layers: usize) -> Assignment {
    Assignment::new(
        PipelineId::new(),
        StageId::new(0),
        0..num_layers,
        Neighbors::none(),
    )
}

fn run_single_benchmark(
    stage: &mut PipelineStage,
    tokens: &[u32],
    generation_length: usize,
    temperature: f32,
) -> Result<UsageStats> {
    let request_id = RequestId::new();
    let prompt_tokens = tokens.len();
    let start = Instant::now();

    let (partition, tokenizer) = stage.partition_and_tokenizer();
    let tokenizer = tokenizer.ok_or_else(|| anyhow::anyhow!("Model does not have a tokenizer"))?;
    let eos_token = tokenizer.eos_token();

    let seed = rand::random::<u64>();
    let sampling = SamplingParams::new(temperature, 1.0, 0, seed);
    let stop_checker = StopChecker::new(eos_token, generation_length);

    let prefill_activation = create_benchmark_activation(request_id, tokens);
    let first_result = partition
        .forward_sample(prefill_activation, &sampling)?
        .ok_or_else(|| anyhow::anyhow!("forward_sample returned None"))?;

    let mut generated_tokens = vec![first_result.token];

    #[allow(clippy::cast_possible_truncation)]
    let time_to_first_token = start.elapsed().as_millis() as u64;

    while stop_checker
        .should_stop(&generated_tokens)
        .should_continue()
    {
        let last_token = *generated_tokens
            .last()
            .ok_or_else(|| anyhow::anyhow!("No tokens"))?;
        let position = prompt_tokens + generated_tokens.len() - 1;

        let decode_activation = create_decode_activation(request_id, last_token, position);
        let result = partition
            .forward_sample(decode_activation, &sampling)?
            .ok_or_else(|| anyhow::anyhow!("forward_sample returned None"))?;

        generated_tokens.push(result.token);
    }

    partition.release_request_cache(request_id);

    #[allow(clippy::cast_possible_truncation)]
    let total_time = start.elapsed().as_millis() as u64;

    Ok(UsageStats {
        prompt_tokens,
        completion_tokens: generated_tokens.len(),
        total_time_ms: total_time,
        time_to_first_token_ms: time_to_first_token,
    })
}

fn compute_single_run_metrics(usage: &UsageStats, run_index: usize) -> SingleRunMetrics {
    SingleRunMetrics {
        run_index,
        prefill_time_ms: usage.time_to_first_token_ms,
        decode_time_ms: usage.decode_time_ms(),
        total_time_ms: usage.total_time_ms,
        time_to_first_token_ms: usage.time_to_first_token_ms,
        prefill_tokens_per_second: usage.prefill_tokens_per_second(),
        decode_tokens_per_second: usage.decode_tokens_per_second(),
    }
}

#[allow(clippy::cast_precision_loss)]
fn aggregate_metrics(runs: &[SingleRunMetrics]) -> AggregatedMetrics {
    let n = runs.len() as f64;
    if runs.is_empty() {
        return AggregatedMetrics {
            mean_prefill_tok_s: 0.0,
            std_prefill_tok_s: 0.0,
            mean_decode_tok_s: 0.0,
            std_decode_tok_s: 0.0,
            mean_ttft_ms: 0.0,
            std_ttft_ms: 0.0,
            mean_total_time_ms: 0.0,
        };
    }

    let mean_prefill: f64 = runs
        .iter()
        .map(|r| r.prefill_tokens_per_second)
        .sum::<f64>()
        / n;
    let mean_decode: f64 = runs.iter().map(|r| r.decode_tokens_per_second).sum::<f64>() / n;
    let mean_ttft: f64 = runs
        .iter()
        .map(|r| r.time_to_first_token_ms as f64)
        .sum::<f64>()
        / n;
    let mean_total: f64 = runs.iter().map(|r| r.total_time_ms as f64).sum::<f64>() / n;

    let std_prefill = if n > 1.0 {
        (runs
            .iter()
            .map(|r| (r.prefill_tokens_per_second - mean_prefill).powi(2))
            .sum::<f64>()
            / (n - 1.0))
            .sqrt()
    } else {
        0.0
    };

    let std_decode = if n > 1.0 {
        (runs
            .iter()
            .map(|r| (r.decode_tokens_per_second - mean_decode).powi(2))
            .sum::<f64>()
            / (n - 1.0))
            .sqrt()
    } else {
        0.0
    };

    let std_ttft = if n > 1.0 {
        (runs
            .iter()
            .map(|r| (r.time_to_first_token_ms as f64 - mean_ttft).powi(2))
            .sum::<f64>()
            / (n - 1.0))
            .sqrt()
    } else {
        0.0
    };

    AggregatedMetrics {
        mean_prefill_tok_s: mean_prefill,
        std_prefill_tok_s: std_prefill,
        mean_decode_tok_s: mean_decode,
        std_decode_tok_s: std_decode,
        mean_ttft_ms: mean_ttft,
        std_ttft_ms: std_ttft,
        mean_total_time_ms: mean_total,
    }
}

#[allow(clippy::cast_precision_loss)]
fn compute_summary(results: &[SequenceLengthResult], total_time_seconds: f64) -> BenchmarkSummary {
    let mut peak_prefill = 0.0_f64;
    let mut peak_decode = 0.0_f64;
    let mut min_ttft: Option<f64> = None;

    for result in results {
        for run in &result.runs {
            peak_prefill = peak_prefill.max(run.prefill_tokens_per_second);
            peak_decode = peak_decode.max(run.decode_tokens_per_second);
            let ttft = run.time_to_first_token_ms as f64;
            min_ttft = Some(min_ttft.map_or(ttft, |m| m.min(ttft)));
        }
    }

    let min_ttft = min_ttft.unwrap_or(0.0);

    BenchmarkSummary {
        peak_prefill_tok_s: peak_prefill,
        peak_decode_tok_s: peak_decode,
        min_ttft_ms: min_ttft,
        total_benchmark_time_seconds: total_time_seconds,
    }
}

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
pub async fn run_benchmark(args: BenchmarkArgs) -> Result<()> {
    let benchmark_start = Instant::now();

    let prompt_sizes = parse_prompt_sizes(&args.prompt_sizes)?;

    if !args.quiet {
        eprintln!("Benchmark configuration:");
        eprintln!("  Model: {}", args.model.display());
        eprintln!("  Device: {}", args.device);
        eprintln!("  Prompt sizes: {prompt_sizes:?}");
        eprintln!("  Generation length: {}", args.generation_length);
        eprintln!("  Runs per config: {}", args.runs);
        eprintln!("  Warmup runs: {}", args.warmup);
        eprintln!();
    }

    let device = create_device(&args.device)?;
    let device_type = device_name(&device);

    if !args.quiet {
        eprintln!("Using device: {device_type}");
    }

    let (baseline_free, device_total) = query_device_memory(&device).unwrap_or((0, 0));

    if !args.quiet && device_total > 0 {
        eprintln!(
            "Device memory: {:.2} GB total, {:.2} GB free",
            device_total as f64 / 1e9,
            baseline_free as f64 / 1e9
        );
    }

    if !args.quiet {
        eprintln!("Loading model...");
    }

    let runtime = CandleRuntime::with_device(device.clone()).context("Failed to create runtime")?;

    let model_name = args
        .model
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model");

    let model_spec = runtime
        .discover_model(ModelId::new(model_name, "v1"), &args.model)
        .context("Failed to discover model")?;

    let partition_spec = PartitionSpec::new(0..model_spec.num_layers, DType::F16);

    let loaded = runtime
        .load_partition(&model_spec, &partition_spec)
        .await
        .context("Failed to load model partition")?;

    let (post_load_free, _) = query_device_memory(&device).unwrap_or((0, 0));
    let model_memory = baseline_free.saturating_sub(post_load_free);

    if !args.quiet {
        eprintln!(
            "Model loaded. Memory used: {:.2} GB",
            model_memory as f64 / 1e9
        );
    }

    let assignment = create_standalone_assignment(model_spec.num_layers);
    let mut stage = PipelineStage::from_loaded(loaded, assignment, None, None);

    let (bos_token, vocab_size) = {
        let tokenizer = stage
            .tokenizer()
            .ok_or_else(|| anyhow::anyhow!("Model does not have a tokenizer"))?;
        (tokenizer.bos_token(), tokenizer.vocab_size())
    };

    if args.warmup > 0 && !args.quiet {
        eprintln!("Running {} warmup iteration(s)...", args.warmup);
    }

    for i in 0..args.warmup {
        let warmup_tokens = generate_test_tokens(bos_token, vocab_size, 32);
        run_single_benchmark(&mut stage, &warmup_tokens, 16, args.temperature)?;
        if !args.quiet {
            eprintln!("  Warmup {}/{} complete", i + 1, args.warmup);
        }
    }

    let mut results = Vec::new();

    for &prompt_size in &prompt_sizes {
        if !args.quiet {
            eprintln!("Benchmarking prompt_size={prompt_size}...");
        }

        let test_tokens = generate_test_tokens(bos_token, vocab_size, prompt_size);
        let mut runs = Vec::new();

        for run_idx in 0..args.runs {
            let usage = run_single_benchmark(
                &mut stage,
                &test_tokens,
                args.generation_length,
                args.temperature,
            )?;

            let metrics = compute_single_run_metrics(&usage, run_idx);

            if !args.quiet {
                eprintln!(
                    "  Run {}/{}: prefill={:.1} tok/s, decode={:.1} tok/s, ttft={} ms",
                    run_idx + 1,
                    args.runs,
                    metrics.prefill_tokens_per_second,
                    metrics.decode_tokens_per_second,
                    metrics.time_to_first_token_ms
                );
            }

            runs.push(metrics);
        }

        let aggregated = aggregate_metrics(&runs);

        results.push(SequenceLengthResult {
            prompt_tokens: prompt_size,
            completion_tokens: args.generation_length,
            runs,
            aggregated,
        });
    }

    let total_benchmark_time = benchmark_start.elapsed().as_secs_f64();

    let output = BenchmarkOutput {
        metadata: BenchmarkMetadata {
            model_path: args.model.display().to_string(),
            device: device_type.to_string(),
            dtype: "f16".to_string(),
            generation_length: args.generation_length,
            runs_per_config: args.runs,
            warmup_runs: args.warmup,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or_else(|_| "0".to_string(), |d| d.as_secs().to_string()),
        },
        memory: MemoryMetrics {
            device_total_bytes: device_total,
            baseline_free_bytes: baseline_free,
            post_load_free_bytes: post_load_free,
            model_memory_bytes: model_memory,
        },
        results: results.clone(),
        summary: compute_summary(&results, total_benchmark_time),
    };

    let json_output = serde_json::to_string_pretty(&output)?;
    println!("{json_output}");

    if args.save || args.output.is_some() {
        let output_path = if let Some(path) = &args.output {
            path.clone()
        } else {
            let benchmarks_dir = PathBuf::from(".benchmarks");
            std::fs::create_dir_all(&benchmarks_dir)
                .context("Failed to create .benchmarks directory")?;

            let model_name = args
                .model
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("model");

            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());

            benchmarks_dir.join(format!("{model_name}_{timestamp}.json"))
        };

        std::fs::write(&output_path, &json_output).with_context(|| {
            format!(
                "Failed to write benchmark results to {}",
                output_path.display()
            )
        })?;

        if !args.quiet {
            eprintln!("Results saved to: {}", output_path.display());
        }
    }

    info!("Benchmark complete in {:.2}s", total_benchmark_time);

    Ok(())
}
