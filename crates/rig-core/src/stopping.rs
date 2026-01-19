use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StopReason {
    NotStopped,
    MaxTokens,
    EosToken,
    StopSequence(String),
}

impl StopReason {
    #[must_use]
    pub const fn should_continue(&self) -> bool {
        matches!(self, Self::NotStopped)
    }

    #[must_use]
    pub const fn is_stopped(&self) -> bool {
        !self.should_continue()
    }
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotStopped => write!(f, "not stopped"),
            Self::MaxTokens => write!(f, "max tokens reached"),
            Self::EosToken => write!(f, "EOS token generated"),
            Self::StopSequence(seq) => write!(f, "stop sequence: {seq}"),
        }
    }
}

pub struct StopChecker {
    eos_token: u32,
    max_tokens: usize,
    stop_sequences: Vec<String>,
}

impl StopChecker {
    #[must_use]
    pub const fn new(eos_token: u32, max_tokens: usize) -> Self {
        Self {
            eos_token,
            max_tokens,
            stop_sequences: Vec::new(),
        }
    }

    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn with_stop_sequences(
        eos_token: u32,
        max_tokens: usize,
        stop_sequences: Vec<String>,
    ) -> Self {
        Self {
            eos_token,
            max_tokens,
            stop_sequences,
        }
    }

    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn should_stop(&self, generated_tokens: &[u32]) -> StopReason {
        if generated_tokens.len() >= self.max_tokens {
            return StopReason::MaxTokens;
        }

        if let Some(&last) = generated_tokens.last()
            && last == self.eos_token
        {
            return StopReason::EosToken;
        }

        StopReason::NotStopped
    }

    #[must_use]
    pub fn should_stop_with_text(
        &self,
        generated_tokens: &[u32],
        decoded_text: &str,
    ) -> StopReason {
        let token_stop = self.should_stop(generated_tokens);
        if token_stop.is_stopped() {
            return token_stop;
        }

        for seq in &self.stop_sequences {
            if decoded_text.ends_with(seq) {
                return StopReason::StopSequence(seq.clone());
            }
        }

        StopReason::NotStopped
    }

    #[must_use]
    pub const fn eos_token(&self) -> u32 {
        self.eos_token
    }

    pub const fn set_eos_token(&mut self, eos_token: u32) {
        self.eos_token = eos_token;
    }

    #[must_use]
    pub const fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    #[must_use]
    pub fn stop_sequences(&self) -> &[String] {
        &self.stop_sequences
    }
}

impl std::fmt::Debug for StopChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StopChecker")
            .field("eos_token", &self.eos_token)
            .field("max_tokens", &self.max_tokens)
            .field("stop_sequences", &self.stop_sequences)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_stopped() {
        let checker = StopChecker::new(2, 100);

        assert_eq!(checker.should_stop(&[]), StopReason::NotStopped);

        assert_eq!(checker.should_stop(&[1, 3, 4, 5]), StopReason::NotStopped);
    }

    #[test]
    fn test_eos_detection() {
        let checker = StopChecker::new(2, 100);

        assert_eq!(checker.should_stop(&[1, 3, 2]), StopReason::EosToken);

        let tokens = vec![2, 3, 4, 5];
        assert_eq!(checker.should_stop(&tokens), StopReason::NotStopped);
    }

    #[test]
    fn test_max_tokens() {
        let checker = StopChecker::new(2, 5);

        assert_eq!(checker.should_stop(&[1, 2, 3, 4]), StopReason::NotStopped);
        assert_eq!(checker.should_stop(&[1, 1, 1, 1, 1]), StopReason::MaxTokens);

        assert_eq!(
            checker.should_stop(&[1, 1, 1, 1, 1, 1]),
            StopReason::MaxTokens
        );
    }

    #[test]
    fn test_max_tokens_takes_precedence() {
        let checker = StopChecker::new(2, 3);

        let tokens = vec![1, 1, 2];
        assert_eq!(checker.should_stop(&tokens), StopReason::MaxTokens);
    }

    #[test]
    fn test_stop_reason_display() {
        assert_eq!(format!("{}", StopReason::NotStopped), "not stopped");
        assert_eq!(format!("{}", StopReason::MaxTokens), "max tokens reached");
        assert_eq!(format!("{}", StopReason::EosToken), "EOS token generated");
    }

    #[test]
    fn test_stop_reason_accessors() {
        assert!(StopReason::NotStopped.should_continue());
        assert!(!StopReason::NotStopped.is_stopped());

        assert!(!StopReason::MaxTokens.should_continue());
        assert!(StopReason::MaxTokens.is_stopped());

        assert!(!StopReason::EosToken.should_continue());
        assert!(StopReason::EosToken.is_stopped());
    }

    #[test]
    fn test_checker_accessors() {
        let checker = StopChecker::new(42, 1000);
        assert_eq!(checker.eos_token(), 42);
        assert_eq!(checker.max_tokens(), 1000);
    }

    #[test]
    fn test_set_eos_token() {
        let mut checker = StopChecker::new(2, 100);
        assert_eq!(checker.eos_token(), 2);

        checker.set_eos_token(128_001);
        assert_eq!(checker.eos_token(), 128_001);

        let tokens = vec![1, 2, 3];
        assert_eq!(checker.should_stop(&tokens), StopReason::NotStopped);

        let tokens_with_new_eos = vec![1, 2, 128_001];
        assert_eq!(
            checker.should_stop(&tokens_with_new_eos),
            StopReason::EosToken
        );
    }
}
