use crate::error::CodecError;

pub trait Codec<M>: Send + Sync {
    fn encode(&self, msg: &M) -> Result<Vec<u8>, CodecError>;
    fn decode(&self, data: &[u8]) -> Result<M, CodecError>;
    fn overhead_bytes(&self) -> usize;
}
