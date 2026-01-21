use std::path::Path;

use crate::error::{PartitionError, RuntimeError};
use crate::traits::cache::KvCache;
use crate::traits::tokenizer::Tokenizer;
use crate::types::{
    Activation, MemoryUsage, ModelId, ModelSpec, PartitionSpec, RequestId, RuntimeCapabilities,
    RuntimeId,
};

pub trait Runtime: Send + Sync {
    fn id(&self) -> RuntimeId;
    fn capabilities(&self) -> RuntimeCapabilities;
    fn discover_model(&self, model_id: ModelId, path: &Path) -> Result<ModelSpec, RuntimeError>;

    fn load_partition(
        &self,
        model: &ModelSpec,
        partition: &PartitionSpec,
    ) -> impl std::future::Future<Output = Result<Box<dyn Partition>, RuntimeError>> + Send;
}

pub trait Partition: Send + Sync {
    fn spec(&self) -> &PartitionSpec;
    fn forward(&self, input: Activation) -> Result<Activation, PartitionError>;
    fn memory_usage(&self) -> MemoryUsage;
    fn kv_cache(&self) -> Option<&dyn KvCache> {
        None
    }
    fn kv_cache_mut(&mut self) -> Option<&mut dyn KvCache> {
        None
    }
    fn tokenizer(&self) -> Option<&dyn Tokenizer> {
        None
    }
    fn release_request_cache(&self, _request_id: RequestId) {}
}
