use crate::error::TokenizerError;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>, TokenizerError>;
    fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError>;
    fn eos_token(&self) -> u32;
    fn bos_token(&self) -> u32;
    fn vocab_size(&self) -> usize;
    fn apply_chat_template(
        &self,
        messages: &[crate::ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let _ = (messages, add_generation_prompt);
        Err(TokenizerError::EncodeFailed(
            "Chat templates not supported by this tokenizer".into(),
        ))
    }
    fn supports_chat_template(&self) -> bool {
        false
    }
}
