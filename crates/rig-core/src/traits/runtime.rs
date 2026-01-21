use std::path::Path;

use crate::error::{PartitionError, RuntimeError};
use crate::traits::cache::KvCache;
use crate::types::{
    Activation, LoadedPartition, MemoryUsage, ModelId, ModelSpec, PartitionSpec, RequestId,
    RuntimeCapabilities, RuntimeId,
};

pub trait Runtime: Send + Sync {
    fn id(&self) -> RuntimeId;
    fn capabilities(&self) -> RuntimeCapabilities;
    fn discover_model(&self, model_id: ModelId, path: &Path) -> Result<ModelSpec, RuntimeError>;

    fn load_partition(
        &self,
        model: &ModelSpec,
        partition: &PartitionSpec,
    ) -> impl std::future::Future<Output = Result<LoadedPartition, RuntimeError>> + Send;
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
    fn release_request_cache(&self, _request_id: RequestId) {}
}
