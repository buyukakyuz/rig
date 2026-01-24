use std::time::Instant;

use rig_core::{
    GenerationParams, RequestId, SamplingParams, StopChecker, StopReason, StopReasonProto,
};

#[derive(Debug, Clone)]
pub enum GenerationStatus {
    Continue { token: u32, position: u32 },
    Complete { stop_reason: StopReasonProto },
}

pub struct GenerationSession {
    request_id: RequestId,
    sampling_params: SamplingParams,
    stop_checker: StopChecker,
    generated_tokens: Vec<u32>,
    prompt_tokens: usize,
    current_position: usize,
    start_time: Instant,
    time_to_first_token_ms: Option<u64>,
}

impl GenerationSession {
    pub fn new(
        request_id: RequestId,
        params: &GenerationParams,
        eos_token: u32,
        prompt_tokens: usize,
        seed: u64,
    ) -> Self {
        let sampling_params =
            SamplingParams::new(params.temperature, params.top_p, params.top_k, seed);
        let stop_checker = StopChecker::new(eos_token, params.max_tokens);

        Self {
            request_id,
            sampling_params,
            stop_checker,
            generated_tokens: Vec::new(),
            prompt_tokens,
            current_position: prompt_tokens,
            start_time: Instant::now(),
            time_to_first_token_ms: None,
        }
    }

    pub fn on_token(&mut self, token: u32) -> GenerationStatus {
        self.generated_tokens.push(token);
        self.current_position += 1;

        if self.time_to_first_token_ms.is_none() {
            #[allow(clippy::cast_possible_truncation)]
            let ttft = self.start_time.elapsed().as_millis() as u64;
            self.time_to_first_token_ms = Some(ttft);
        }

        let stop_reason = self.stop_checker.should_stop(&self.generated_tokens);

        if stop_reason.is_stopped() {
            GenerationStatus::Complete {
                stop_reason: Self::convert_stop_reason(stop_reason),
            }
        } else {
            #[allow(clippy::cast_possible_truncation)]
            let position = self.current_position as u32;
            GenerationStatus::Continue { token, position }
        }
    }

    #[must_use]
    pub fn generated_tokens(&self) -> &[u32] {
        &self.generated_tokens
    }

    #[must_use]
    pub fn time_to_first_token_ms(&self) -> u64 {
        self.time_to_first_token_ms.unwrap_or(0)
    }

    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    #[must_use]
    pub const fn sampling_params(&self) -> &SamplingParams {
        &self.sampling_params
    }

    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn eos_token(&self) -> u32 {
        self.stop_checker.eos_token()
    }

    fn convert_stop_reason(reason: StopReason) -> StopReasonProto {
        match reason {
            StopReason::EosToken => StopReasonProto::EosToken,
            StopReason::StopSequence(seq) => StopReasonProto::StopSequence(seq),
            StopReason::MaxTokens | StopReason::NotStopped => StopReasonProto::MaxTokens,
        }
    }
}

impl std::fmt::Debug for GenerationSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationSession")
            .field("request_id", &self.request_id)
            .field("prompt_tokens", &self.prompt_tokens)
            .field("generated_tokens", &self.generated_tokens.len())
            .field("current_position", &self.current_position)
            .field("time_to_first_token_ms", &self.time_to_first_token_ms)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_session_creation() {
        let params = GenerationParams::default();
        let session = GenerationSession::new(RequestId::new(), &params, 2, 10, 42);

        assert_eq!(session.prompt_tokens(), 10);
        assert!(session.generated_tokens().is_empty());
    }

    #[test]
    fn test_on_token_continue() {
        let params = GenerationParams::default().with_max_tokens(10);
        let mut session = GenerationSession::new(RequestId::new(), &params, 2, 5, 42);

        let status = session.on_token(100);
        assert!(matches!(
            status,
            GenerationStatus::Continue {
                token: 100,
                position: 6
            }
        ));
        assert_eq!(session.generated_tokens().len(), 1);
    }

    #[test]
    fn test_on_token_eos() {
        let params = GenerationParams::default().with_max_tokens(10);
        let mut session = GenerationSession::new(RequestId::new(), &params, 2, 5, 42);

        let status = session.on_token(2);
        assert!(matches!(
            status,
            GenerationStatus::Complete {
                stop_reason: StopReasonProto::EosToken
            }
        ));
    }

    #[test]
    fn test_on_token_max_tokens() {
        let params = GenerationParams::default().with_max_tokens(2);
        let mut session = GenerationSession::new(RequestId::new(), &params, 2, 5, 42);

        let status1 = session.on_token(100);
        assert!(matches!(status1, GenerationStatus::Continue { .. }));

        let status2 = session.on_token(101);
        assert!(matches!(
            status2,
            GenerationStatus::Complete {
                stop_reason: StopReasonProto::MaxTokens
            }
        ));
    }
}
