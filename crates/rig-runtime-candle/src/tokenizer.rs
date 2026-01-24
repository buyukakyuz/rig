#![allow(clippy::future_not_send)]

use ouroboros::self_referencing;
use tokenizers::Tokenizer as HfTokenizer;

type HfDecodeStream<'a> = tokenizers::DecodeStream<
    'a,
    tokenizers::models::ModelWrapper,
    tokenizers::normalizers::NormalizerWrapper,
    tokenizers::pre_tokenizers::PreTokenizerWrapper,
    tokenizers::processors::PostProcessorWrapper,
    tokenizers::decoders::DecoderWrapper,
>;

#[self_referencing]
pub struct CandleDecodeStream {
    tokenizer: Box<HfTokenizer>,
    #[borrows(tokenizer)]
    #[covariant]
    stream: HfDecodeStream<'this>,
}

impl rig_core::TokenDecodeStream for CandleDecodeStream {
    fn step(
        &mut self,
        token_id: u32,
    ) -> std::result::Result<Option<String>, rig_core::TokenizerError> {
        self.with_stream_mut(|stream| {
            stream
                .step(token_id)
                .map_err(|e| rig_core::TokenizerError::DecodeFailed(e.to_string()))
        })
    }

    fn flush(&mut self) -> std::result::Result<Option<String>, rig_core::TokenizerError> {
        Ok(None)
    }
}

pub struct CandleTokenizer {
    tokenizer: HfTokenizer,
    chat_template: Option<String>,
    eos_token_str: String,
    bos_token_str: String,
    add_bos_token: bool,
    bos_token_id: u32,
    eos_token_id: u32,
}

impl CandleTokenizer {
    pub fn new(
        tokenizer: HfTokenizer,
        chat_template: Option<String>,
        eos_token_str: String,
        bos_token_str: Option<String>,
        add_bos_token: bool,
        bos_token_id: u32,
        eos_token_id: u32,
    ) -> Self {
        Self {
            tokenizer,
            chat_template,
            eos_token_str,
            bos_token_str: bos_token_str.unwrap_or_default(),
            add_bos_token,
            bos_token_id,
            eos_token_id,
        }
    }
}

impl rig_core::Tokenizer for CandleTokenizer {
    fn encode(
        &self,
        text: &str,
        add_bos: bool,
    ) -> std::result::Result<Vec<u32>, rig_core::TokenizerError> {
        let add_special_tokens = add_bos && self.add_bos_token;
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| rig_core::TokenizerError::EncodeFailed(e.to_string()))?;

        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, tokens: &[u32]) -> std::result::Result<String, rig_core::TokenizerError> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| rig_core::TokenizerError::DecodeFailed(e.to_string()))
    }

    fn eos_token(&self) -> u32 {
        self.eos_token_id
    }

    fn bos_token(&self) -> u32 {
        self.bos_token_id
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    fn apply_chat_template(
        &self,
        messages: &[rig_core::ChatMessage],
        add_generation_prompt: bool,
    ) -> std::result::Result<String, rig_core::TokenizerError> {
        let template_str = self.chat_template.as_ref().ok_or_else(|| {
            rig_core::TokenizerError::EncodeFailed(
                "No chat template available for this model".into(),
            )
        })?;

        let mut env = minijinja::Environment::new();
        env.add_template("chat", template_str).map_err(|e| {
            rig_core::TokenizerError::EncodeFailed(format!("Invalid chat template: {e}"))
        })?;

        let template = env
            .get_template("chat")
            .map_err(|e| rig_core::TokenizerError::EncodeFailed(format!("Template error: {e}")))?;

        let messages_value: Vec<minijinja::Value> = messages
            .iter()
            .map(|m| {
                minijinja::context! {
                    role => m.role.as_str(),
                    content => m.content.as_str()
                }
            })
            .collect();

        let ctx = minijinja::context! {
            messages => messages_value,
            eos_token => self.eos_token_str.as_str(),
            bos_token => self.bos_token_str.as_str(),
            add_generation_prompt => add_generation_prompt,
        };

        template.render(ctx).map_err(|e| {
            rig_core::TokenizerError::EncodeFailed(format!("Chat template render failed: {e}"))
        })
    }

    fn supports_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    fn encode_batch(
        &self,
        texts: &[&str],
        add_bos: bool,
    ) -> std::result::Result<Vec<Vec<u32>>, rig_core::TokenizerError> {
        let add_special_tokens = add_bos && self.add_bos_token;
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), add_special_tokens)
            .map_err(|e| rig_core::TokenizerError::EncodeFailed(e.to_string()))?;

        let results = encodings
            .into_iter()
            .map(|encoding| encoding.get_ids().to_vec())
            .collect();
        Ok(results)
    }

    fn decode_batch(
        &self,
        token_sequences: &[&[u32]],
    ) -> std::result::Result<Vec<String>, rig_core::TokenizerError> {
        self.tokenizer
            .decode_batch(token_sequences, true)
            .map_err(|e| rig_core::TokenizerError::DecodeFailed(e.to_string()))
    }

    #[allow(clippy::borrowed_box)]
    fn create_decode_stream(
        &self,
        skip_special_tokens: bool,
    ) -> std::result::Result<Box<dyn rig_core::TokenDecodeStream>, rig_core::TokenizerError> {
        let tokenizer_clone = Box::new(self.tokenizer.clone());

        Ok(Box::new(
            CandleDecodeStreamBuilder {
                tokenizer: tokenizer_clone,
                stream_builder: |tokenizer: &Box<HfTokenizer>| {
                    tokenizer.decode_stream(skip_special_tokens)
                },
            }
            .build(),
        ))
    }
}
