mod channel;

pub use channel::MessageChannel;

use rig_core::error::CodecError;
use rig_core::traits::Codec;
use serde::Serialize;
use serde::de::DeserializeOwned;

#[derive(Debug, Clone, Copy, Default)]
pub struct BincodeCodec;

impl BincodeCodec {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl<M> Codec<M> for BincodeCodec
where
    M: Serialize + DeserializeOwned,
{
    fn encode(&self, msg: &M) -> Result<Vec<u8>, CodecError> {
        bincode::serialize(msg).map_err(|e| CodecError::EncodeFailed(e.to_string()))
    }
    fn decode(&self, data: &[u8]) -> Result<M, CodecError> {
        bincode::deserialize(data).map_err(|e| CodecError::DecodeFailed(e.to_string()))
    }
    fn overhead_bytes(&self) -> usize {
        8
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct SimpleMessage {
        id: u32,
        name: String,
    }

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct ComplexMessage {
        id: u64,
        items: Vec<String>,
        nested: Option<NestedMessage>,
    }

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct NestedMessage {
        value: i32,
        data: Vec<u8>,
    }

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    enum EnumMessage {
        Variant1,
        Variant2(u32),
        Variant3 { field: String },
    }

    #[test]
    fn encode_decode_simple_message() {
        let codec = BincodeCodec::new();
        let msg = SimpleMessage {
            id: 42,
            name: "test".to_string(),
        };

        let encoded = codec.encode(&msg).expect("encode failed");
        let decoded: SimpleMessage = codec.decode(&encoded).expect("decode failed");

        assert_eq!(msg, decoded);
    }

    #[test]
    fn encode_decode_complex_message() {
        let codec = BincodeCodec::new();
        let msg = ComplexMessage {
            id: 12345,
            items: vec!["one".to_string(), "two".to_string(), "three".to_string()],
            nested: Some(NestedMessage {
                value: -42,
                data: vec![1, 2, 3, 4, 5],
            }),
        };

        let encoded = codec.encode(&msg).expect("encode failed");
        let decoded: ComplexMessage = codec.decode(&encoded).expect("decode failed");

        assert_eq!(msg, decoded);
    }

    #[test]
    fn encode_decode_enum_variants() {
        let codec = BincodeCodec::new();

        let messages = vec![
            EnumMessage::Variant1,
            EnumMessage::Variant2(100),
            EnumMessage::Variant3 {
                field: "hello".to_string(),
            },
        ];

        for msg in messages {
            let encoded = codec.encode(&msg).expect("encode failed");
            let decoded: EnumMessage = codec.decode(&encoded).expect("decode failed");
            assert_eq!(msg, decoded);
        }
    }

    #[test]
    fn encode_decode_empty_collections() {
        let codec = BincodeCodec::new();
        let msg = ComplexMessage {
            id: 0,
            items: vec![],
            nested: None,
        };

        let encoded = codec.encode(&msg).expect("encode failed");
        let decoded: ComplexMessage = codec.decode(&encoded).expect("decode failed");

        assert_eq!(msg, decoded);
    }

    #[test]
    fn decode_invalid_data_fails() {
        let codec = BincodeCodec::new();

        let invalid = vec![0xFF, 0xFF, 0xFF, 0xFF];
        let result: Result<SimpleMessage, _> = codec.decode(&invalid);

        assert!(result.is_err());
        assert!(matches!(result, Err(CodecError::DecodeFailed(_))));
    }

    #[test]
    fn decode_empty_data_fails() {
        let codec = BincodeCodec::new();

        let result: Result<SimpleMessage, _> = codec.decode(&[]);

        assert!(result.is_err());
        assert!(matches!(result, Err(CodecError::DecodeFailed(_))));
    }

    #[test]
    fn overhead_bytes_is_reasonable() {
        let codec = BincodeCodec::new();
        let overhead = <BincodeCodec as Codec<SimpleMessage>>::overhead_bytes(&codec);

        assert!(overhead > 0);
        assert!(overhead < 64);
    }

    #[test]
    fn codec_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BincodeCodec>();
    }
}
