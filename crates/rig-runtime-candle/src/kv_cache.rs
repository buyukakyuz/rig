use std::collections::HashMap;

use rig_core::KvCache;
use rig_core::error::CacheError;
use rig_core::types::{CacheSlot, MemoryUsage, RequestId};

use crate::cache::PartitionKvCache;

#[derive(Debug)]
pub struct CandleKvCache {
    slots: HashMap<RequestId, CacheSlot>,
    tensor_cache: PartitionKvCache,
    max_slots: usize,
    memory_per_token: usize,
    num_layers: usize,
}

impl CandleKvCache {
    #[must_use]
    pub fn new(num_layers: usize, max_slots: usize, memory_per_token: usize) -> Self {
        Self {
            slots: HashMap::new(),
            tensor_cache: PartitionKvCache::new(num_layers),
            max_slots,
            memory_per_token,
            num_layers,
        }
    }

    #[must_use]
    pub const fn tensor_cache(&self) -> &PartitionKvCache {
        &self.tensor_cache
    }

    pub const fn tensor_cache_mut(&mut self) -> &mut PartitionKvCache {
        &mut self.tensor_cache
    }

    pub fn update_seq_len(&mut self, request_id: &RequestId, new_len: usize) {
        if let Some(slot) = self.slots.get_mut(request_id) {
            slot.seq_len = new_len;
        }
    }

    pub fn clear(&mut self) {
        self.slots.clear();
        self.tensor_cache.clear();
    }
}

impl KvCache for CandleKvCache {
    fn allocate(&mut self, request_id: RequestId, max_seq_len: usize) -> Result<(), CacheError> {
        if self.slots.contains_key(&request_id) {
            return Err(CacheError::AlreadyAllocated(request_id));
        }

        if self.slots.len() >= self.max_slots {
            return Err(CacheError::MaxSlotsReached {
                max: self.max_slots,
            });
        }

        let slot = CacheSlot::new(request_id, max_seq_len);
        self.slots.insert(request_id, slot);

        Ok(())
    }

    fn get(&self, request_id: RequestId) -> Option<&CacheSlot> {
        self.slots.get(&request_id)
    }

    fn get_mut(&mut self, request_id: RequestId) -> Option<&mut CacheSlot> {
        self.slots.get_mut(&request_id)
    }

    fn release(&mut self, request_id: RequestId) -> Result<(), CacheError> {
        if self.slots.remove(&request_id).is_some() {
            self.tensor_cache.clear();
            Ok(())
        } else {
            Err(CacheError::NotAllocated(request_id))
        }
    }

    fn memory_usage(&self) -> MemoryUsage {
        let active_tokens: usize = self.slots.values().map(|s| s.seq_len).sum();
        let cache_bytes = active_tokens * self.memory_per_token * self.num_layers;

        MemoryUsage {
            weights_bytes: 0,
            cache_bytes: cache_bytes as u64,
            scratch_bytes: 0,
        }
    }

    fn active_slots(&self) -> usize {
        self.slots.len()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_release() {
        let mut cache = CandleKvCache::new(32, 4, 16384);

        let req1 = RequestId::new();
        let req2 = RequestId::new();

        cache.allocate(req1, 2048).unwrap();
        cache.allocate(req2, 1024).unwrap();

        assert_eq!(cache.active_slots(), 2);

        assert!(cache.get(req1).is_some());
        assert!(cache.get(req2).is_some());

        cache.release(req1).unwrap();
        assert_eq!(cache.active_slots(), 1);
        assert!(cache.get(req1).is_none());

        cache.release(req2).unwrap();
        assert_eq!(cache.active_slots(), 0);
    }

    #[test]
    fn test_max_slots_limit() {
        let mut cache = CandleKvCache::new(32, 2, 16384);

        cache.allocate(RequestId::new(), 1024).unwrap();
        cache.allocate(RequestId::new(), 1024).unwrap();

        let result = cache.allocate(RequestId::new(), 1024);
        assert!(matches!(
            result,
            Err(CacheError::MaxSlotsReached { max: 2 })
        ));
    }

    #[test]
    fn test_duplicate_allocation_fails() {
        let mut cache = CandleKvCache::new(32, 4, 16384);

        let req = RequestId::new();
        cache.allocate(req, 1024).unwrap();

        let result = cache.allocate(req, 2048);
        assert!(matches!(result, Err(CacheError::AlreadyAllocated(_))));
    }

    #[test]
    fn test_release_unknown_fails() {
        let mut cache = CandleKvCache::new(32, 4, 16384);

        let unknown_req = RequestId::new();
        let result = cache.release(unknown_req);
        assert!(matches!(result, Err(CacheError::NotAllocated(_))));
    }

    #[test]
    fn test_memory_usage_tracking() {
        let mut cache = CandleKvCache::new(32, 4, 16384);

        assert_eq!(cache.memory_usage().cache_bytes, 0);

        let req = RequestId::new();
        cache.allocate(req, 2048).unwrap();

        assert_eq!(cache.memory_usage().cache_bytes, 0);

        cache.update_seq_len(&req, 100);

        let expected = 100 * 16384 * 32;
        assert_eq!(cache.memory_usage().cache_bytes, expected as u64);
    }

    #[test]
    fn test_slot_metadata() {
        let mut cache = CandleKvCache::new(32, 4, 16384);

        let req = RequestId::new();
        cache.allocate(req, 4096).unwrap();

        let slot = cache.get(req).unwrap();
        assert_eq!(slot.seq_len, 0);
        assert_eq!(slot.max_seq_len, 4096);
        assert!(slot.has_capacity(4096));
        assert!(!slot.has_capacity(4097));

        cache.update_seq_len(&req, 1000);
        let slot = cache.get(req).unwrap();
        assert_eq!(slot.seq_len, 1000);
        assert_eq!(slot.remaining_capacity(), 3096);
    }
}
