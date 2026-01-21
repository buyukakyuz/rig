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

#[cfg(test)]
mod tests {
    use super::*;

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
}
