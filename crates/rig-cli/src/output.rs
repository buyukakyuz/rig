use serde::Serialize;

#[derive(Serialize)]
pub struct GenerateOutput {
    pub output: GenerateContent,
    pub usage: UsageOutput,
}

#[derive(Serialize)]
pub struct GenerateContent {
    #[serde(rename = "type")]
    pub content_type: &'static str,
    pub content: String,
}

impl GenerateContent {
    #[allow(clippy::missing_const_for_fn)]
    pub fn text(content: String) -> Self {
        Self {
            content_type: "text",
            content,
        }
    }
}

#[derive(Serialize)]
pub struct UsageOutput {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub total_time_ms: u64,
    pub time_to_first_token_ms: u64,
    pub tokens_per_second: f64,
}

impl From<rig_core::UsageStats> for UsageOutput {
    fn from(stats: rig_core::UsageStats) -> Self {
        Self {
            prompt_tokens: stats.prompt_tokens,
            completion_tokens: stats.completion_tokens,
            total_tokens: stats.total_tokens(),
            total_time_ms: stats.total_time_ms,
            time_to_first_token_ms: stats.time_to_first_token_ms,
            tokens_per_second: stats.tokens_per_second(),
        }
    }
}

#[derive(Serialize)]
pub struct StatusOutput {
    pub total_nodes: usize,
    pub healthy_nodes: usize,
    pub total_pipelines: usize,
    pub ready_pipelines: usize,
    pub nodes: Vec<NodeOutput>,
    pub pipelines: Vec<PipelineOutput>,
}

#[derive(Serialize)]
pub struct NodeOutput {
    pub node_id: String,
    pub addresses: Vec<String>,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_bytes: Option<u64>,
}

#[derive(Serialize)]
pub struct PipelineOutput {
    pub pipeline_id: String,
    pub model_id: String,
    pub status: String,
    pub stages: Vec<StageOutput>,
}

#[derive(Serialize)]
pub struct StageOutput {
    pub stage_id: u32,
    pub node_id: String,
    pub layer_range: String,
    pub ready: bool,
}

// Benchmark output types

#[derive(Serialize)]
pub struct BenchmarkOutput {
    pub metadata: BenchmarkMetadata,
    pub memory: MemoryMetrics,
    pub results: Vec<SequenceLengthResult>,
    pub summary: BenchmarkSummary,
}

#[derive(Serialize)]
pub struct BenchmarkMetadata {
    pub model_path: String,
    pub device: String,
    pub dtype: String,
    pub generation_length: usize,
    pub runs_per_config: usize,
    pub warmup_runs: usize,
    pub timestamp: String,
}

#[derive(Serialize)]
#[allow(clippy::struct_field_names)]
pub struct MemoryMetrics {
    pub device_total_bytes: u64,
    pub baseline_free_bytes: u64,
    pub post_load_free_bytes: u64,
    pub model_memory_bytes: u64,
}

#[derive(Clone, Serialize)]
pub struct SequenceLengthResult {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub runs: Vec<SingleRunMetrics>,
    pub aggregated: AggregatedMetrics,
}

#[derive(Clone, Serialize)]
pub struct SingleRunMetrics {
    pub run_index: usize,
    pub prefill_time_ms: u64,
    pub decode_time_ms: u64,
    pub total_time_ms: u64,
    pub time_to_first_token_ms: u64,
    pub prefill_tokens_per_second: f64,
    pub decode_tokens_per_second: f64,
}

#[derive(Clone, Serialize)]
pub struct AggregatedMetrics {
    pub mean_prefill_tok_s: f64,
    pub std_prefill_tok_s: f64,
    pub mean_decode_tok_s: f64,
    pub std_decode_tok_s: f64,
    pub mean_ttft_ms: f64,
    pub std_ttft_ms: f64,
    pub mean_total_time_ms: f64,
}

#[derive(Serialize)]
pub struct BenchmarkSummary {
    pub peak_prefill_tok_s: f64,
    pub peak_decode_tok_s: f64,
    pub min_ttft_ms: f64,
    pub total_benchmark_time_seconds: f64,
}
