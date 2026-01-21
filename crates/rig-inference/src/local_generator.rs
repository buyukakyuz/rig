use std::time::Instant;

use rig_core::{
    Activation, ActivationMetadata, DType, GenerationParams, Partition, PartitionError, RequestId,
    Shape, TensorData, Tokenizer, UsageStats,
};
use tokio::sync::mpsc;
use tracing::{debug, info, trace};

use crate::{Sampler, StopChecker};

#[derive(Debug)]
pub enum GeneratorError {
    Forward(PartitionError),
    NoLogits,
    NoTokens,
    NoTokenizer,
}

impl std::fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Forward(e) => write!(f, "forward pass failed: {e}"),
            Self::NoLogits => write!(f, "forward pass returned no logits"),
            Self::NoTokens => write!(f, "no tokens generated"),
            Self::NoTokenizer => write!(f, "model does not support tokenization"),
        }
    }
}

impl std::error::Error for GeneratorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Forward(e) => Some(e),
            _ => None,
        }
    }
}

impl From<PartitionError> for GeneratorError {
    fn from(e: PartitionError) -> Self {
        Self::Forward(e)
    }
}

pub struct LocalGenerator<'a> {
    partition: &'a mut dyn Partition,
    tokenizer: &'a dyn Tokenizer,
}

impl<'a> LocalGenerator<'a> {
    #[must_use]
    pub fn new(partition: &'a mut dyn Partition, tokenizer: &'a dyn Tokenizer) -> Self {
        Self {
            partition,
            tokenizer,
        }
    }

    #[allow(clippy::too_many_lines, clippy::unused_async)]
    pub async fn generate(
        &mut self,
        initial_activation: Activation,
        params: &GenerationParams,
        token_tx: mpsc::UnboundedSender<String>,
    ) -> Result<UsageStats, GeneratorError> {
        let request_id = initial_activation.metadata.request_id;
        let start = Instant::now();

        let eos_token = self.tokenizer.eos_token();
        let prompt_tokens = initial_activation.metadata.positions.len();
        debug!(prompt_tokens, "Prompt token count");

        debug!("Starting prefill forward pass");
        let output = self.partition.forward(initial_activation)?;
        let logits = extract_logits(&output);

        if logits.is_empty() {
            return Err(GeneratorError::NoLogits);
        }

        let mut sampler = Sampler::new(params, None);
        let stop_checker = StopChecker::with_stop_sequences(
            eos_token,
            params.max_tokens,
            params.stop_sequences.clone(),
        );
        debug!(
            eos_token,
            max_tokens = params.max_tokens,
            stop_sequences = ?params.stop_sequences,
            "StopChecker configured"
        );

        let mut decode_stream = self.tokenizer.create_decode_stream(true).ok();

        let first_token = sampler.sample(&logits);
        let mut generated_tokens = vec![first_token];
        #[allow(clippy::cast_possible_truncation)]
        let time_to_first_token = start.elapsed().as_millis() as u64;
        debug!(token = first_token, "Sampled first token");

        let mut current_decoded_text = String::new();

        if let Some(ref mut stream) = decode_stream {
            if let Ok(Some(new_text)) = stream.step(first_token) {
                current_decoded_text.push_str(&new_text);
                let _ = token_tx.send(new_text);
            }
        }

        while stop_checker
            .should_stop_with_text(&generated_tokens, &current_decoded_text)
            .should_continue()
        {
            let last_token = *generated_tokens.last().ok_or(GeneratorError::NoTokens)?;
            let position = prompt_tokens + generated_tokens.len() - 1;
            trace!(last_token, position, "Decode step");

            let decode_activation = create_decode_activation(request_id, last_token, position);
            let output = self.partition.forward(decode_activation)?;
            let logits = extract_logits(&output);

            if logits.is_empty() {
                return Err(GeneratorError::NoLogits);
            }

            let token = sampler.sample(&logits);
            generated_tokens.push(token);

            if let Some(ref mut stream) = decode_stream {
                if let Ok(Some(new_text)) = stream.step(token) {
                    current_decoded_text.push_str(&new_text);
                    let _ = token_tx.send(new_text);
                }
            }

            if generated_tokens.len() % 10 == 0 {
                debug!(count = generated_tokens.len(), "Generated tokens");
            }
        }

        let stop_reason =
            stop_checker.should_stop_with_text(&generated_tokens, &current_decoded_text);
        debug!(
            tokens_generated = generated_tokens.len(),
            %stop_reason,
            "Streaming generation complete"
        );

        self.partition.release_request_cache(request_id);

        #[allow(clippy::cast_possible_truncation)]
        let total_time = start.elapsed().as_millis() as u64;

        info!(
            completion_tokens = generated_tokens.len(),
            %stop_reason,
            total_time_ms = total_time,
            "Streaming generation complete"
        );

        Ok(UsageStats {
            prompt_tokens,
            completion_tokens: generated_tokens.len(),
            total_time_ms: total_time,
            time_to_first_token_ms: time_to_first_token,
        })
    }
}

fn extract_logits(activation: &Activation) -> Vec<f32> {
    let bytes = activation.data.as_bytes();
    let dtype = activation.dtype();

    match dtype {
        DType::F32 => {
            if bytes.len() < 4 {
                return Vec::new();
            }
            bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                    f32::from_le_bytes(arr)
                })
                .collect()
        }
        DType::F16 => {
            if bytes.len() < 2 {
                return Vec::new();
            }
            bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let arr: [u8; 2] = chunk.try_into().unwrap_or([0; 2]);
                    half::f16::from_le_bytes(arr).to_f32()
                })
                .collect()
        }
        DType::BF16 => {
            if bytes.len() < 2 {
                return Vec::new();
            }
            bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let arr: [u8; 2] = chunk.try_into().unwrap_or([0; 2]);
                    half::bf16::from_le_bytes(arr).to_f32()
                })
                .collect()
        }
        _ => Vec::new(),
    }
}

fn create_decode_activation(request_id: RequestId, token: u32, position: usize) -> Activation {
    let bytes = token.to_le_bytes().to_vec();
    trace!(
        token,
        position,
        bytes_len = bytes.len(),
        "Creating decode activation"
    );
    let data = TensorData::cpu(bytes, DType::I8);

    let shape = Shape::new(vec![1, 1, 1]);

    #[allow(clippy::cast_possible_truncation)]
    let position_u32 = position as u32;

    let metadata = ActivationMetadata::new(request_id, position_u32, vec![position_u32], false);

    Activation::new(data, shape, metadata)
}
