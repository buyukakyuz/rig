use crate::error::TokenizerError;

pub trait TokenDecodeStream: Send {
    fn step(&mut self, token_id: u32) -> Result<Option<String>, TokenizerError>;
    fn flush(&mut self) -> Result<Option<String>, TokenizerError>;
}

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

    fn encode_batch(&self, texts: &[&str], add_bos: bool) -> Result<Vec<Vec<u32>>, TokenizerError> {
        texts
            .iter()
            .map(|text| self.encode(text, add_bos))
            .collect()
    }

    fn decode_batch(&self, token_sequences: &[&[u32]]) -> Result<Vec<String>, TokenizerError> {
        token_sequences
            .iter()
            .map(|tokens| self.decode(tokens))
            .collect()
    }

    fn create_decode_stream(
        &self,
        skip_special_tokens: bool,
    ) -> Result<Box<dyn TokenDecodeStream>, TokenizerError> {
        let _ = skip_special_tokens;
        Err(TokenizerError::DecodeFailed(
            "Incremental decoding not supported by this tokenizer".into(),
        ))
    }
}
