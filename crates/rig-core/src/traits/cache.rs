use crate::error::CacheError;
use crate::types::{CacheSlot, MemoryUsage, RequestId};

pub trait KvCache: Send + Sync {
    fn allocate(&mut self, request_id: RequestId, max_seq_len: usize) -> Result<(), CacheError>;
    fn get(&self, request_id: RequestId) -> Option<&CacheSlot>;
    fn get_mut(&mut self, request_id: RequestId) -> Option<&mut CacheSlot>;
    fn release(&mut self, request_id: RequestId) -> Result<(), CacheError>;
    fn memory_usage(&self) -> MemoryUsage;
    fn active_slots(&self) -> usize;
}
