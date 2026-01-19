use serde::{Deserialize, Serialize};

use crate::types::id::RequestId;
use crate::types::request::GenerationParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I4,
}

impl DType {
    #[must_use]
    pub const fn size_bytes_for_elements(&self, num_elements: usize) -> usize {
        match self {
            Self::F32 => num_elements * 4,
            Self::F16 | Self::BF16 => num_elements * 2,
            Self::I8 => num_elements,
            Self::I4 => num_elements.div_ceil(2),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    #[must_use]
    pub const fn new(dims: Vec<usize>) -> Self {
        Self(dims)
    }

    #[must_use]
    pub fn from_slice(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }

    #[must_use]
    pub fn num_elements(&self) -> usize {
        if self.0.is_empty() {
            0
        } else {
            self.0.iter().product()
        }
    }

    #[must_use]
    pub fn numel(&self) -> usize {
        self.num_elements()
    }

    #[must_use]
    pub fn size_bytes(&self, dtype: DType) -> usize {
        dtype.size_bytes_for_elements(self.num_elements())
    }

    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn dim(&self, index: usize) -> Option<usize> {
        self.0.get(index).copied()
    }

    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.0
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::from_slice(dims)
    }
}

#[derive(Debug, Clone)]
pub enum TensorData {
    Cpu { bytes: Vec<u8>, dtype: DType },
}

impl TensorData {
    #[must_use]
    pub const fn cpu(bytes: Vec<u8>, dtype: DType) -> Self {
        Self::Cpu { bytes, dtype }
    }

    #[must_use]
    pub const fn dtype(&self) -> DType {
        match self {
            Self::Cpu { dtype, .. } => *dtype,
        }
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Cpu { bytes, .. } => bytes,
        }
    }

    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Cpu { bytes, .. } => bytes.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationMetadata {
    pub request_id: RequestId,
    pub sequence_num: u32,
    pub positions: Vec<u32>,
    pub is_prefill: bool,
    pub generation_params: Option<GenerationParams>,
}

impl ActivationMetadata {
    #[must_use]
    pub const fn new(
        request_id: RequestId,
        sequence_num: u32,
        positions: Vec<u32>,
        is_prefill: bool,
    ) -> Self {
        Self {
            request_id,
            sequence_num,
            positions,
            is_prefill,
            generation_params: None,
        }
    }

    #[must_use]
    pub fn with_generation_params(mut self, params: GenerationParams) -> Self {
        self.generation_params = Some(params);
        self
    }
}

#[derive(Debug, Clone)]
pub struct Activation {
    pub data: TensorData,
    pub shape: Shape,
    pub metadata: ActivationMetadata,
}

impl Activation {
    #[must_use]
    pub const fn new(data: TensorData, shape: Shape, metadata: ActivationMetadata) -> Self {
        Self {
            data,
            shape,
            metadata,
        }
    }

    #[must_use]
    pub const fn from_bytes(
        bytes: Vec<u8>,
        dtype: DType,
        shape: Shape,
        metadata: ActivationMetadata,
    ) -> Self {
        Self {
            data: TensorData::cpu(bytes, dtype),
            shape,
            metadata,
        }
    }

    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.data.dtype()
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }

    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.data.size_bytes()
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub weights_bytes: u64,
    pub cache_bytes: u64,
    pub scratch_bytes: u64,
}

impl MemoryUsage {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            weights_bytes: 0,
            cache_bytes: 0,
            scratch_bytes: 0,
        }
    }

    #[must_use]
    pub const fn total(&self) -> u64 {
        self.weights_bytes + self.cache_bytes + self.scratch_bytes
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSlot {
    pub request_id: RequestId,
    pub seq_len: usize,
    pub max_seq_len: usize,
}

impl CacheSlot {
    #[must_use]
    pub const fn new(request_id: RequestId, max_seq_len: usize) -> Self {
        Self {
            request_id,
            seq_len: 0,
            max_seq_len,
        }
    }

    #[must_use]
    pub const fn has_capacity(&self, additional: usize) -> bool {
        self.seq_len + additional <= self.max_seq_len
    }

    #[must_use]
    pub const fn remaining_capacity(&self) -> usize {
        self.max_seq_len - self.seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_size_bytes_for_elements() {
        assert_eq!(DType::F32.size_bytes_for_elements(100), 400);
        assert_eq!(DType::F16.size_bytes_for_elements(100), 200);
        assert_eq!(DType::I8.size_bytes_for_elements(100), 100);
        assert_eq!(DType::I4.size_bytes_for_elements(100), 50);
        assert_eq!(DType::I4.size_bytes_for_elements(101), 51);
    }

    #[test]
    fn shape_num_elements() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.num_elements(), 24);
        assert_eq!(shape.numel(), 24);
        assert_eq!(shape.ndim(), 3);
    }

    #[test]
    fn shape_empty() {
        let shape = Shape::new(vec![]);
        assert_eq!(shape.num_elements(), 0);
        assert_eq!(shape.ndim(), 0);
    }

    #[test]
    fn shape_size_bytes() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.size_bytes(DType::F32), 96);
        assert_eq!(shape.size_bytes(DType::F16), 48);
        assert_eq!(shape.size_bytes(DType::I4), 12);
    }

    #[test]
    fn shape_from_vec() {
        let shape: Shape = vec![1, 2, 3].into();
        assert_eq!(shape.dims(), &[1, 2, 3]);
    }

    #[test]
    fn tensor_data_cpu() {
        let data = TensorData::cpu(vec![0u8; 16], DType::F32);
        assert_eq!(data.dtype(), DType::F32);
        assert_eq!(data.size_bytes(), 16);
        assert_eq!(data.as_bytes().len(), 16);
    }

    #[test]
    fn cache_slot_capacity() {
        let mut slot = CacheSlot::new(RequestId::new(), 2048);
        assert_eq!(slot.remaining_capacity(), 2048);
        assert!(slot.has_capacity(100));

        slot.seq_len = 2000;
        assert_eq!(slot.remaining_capacity(), 48);
        assert!(slot.has_capacity(48));
        assert!(!slot.has_capacity(49));
    }

    #[test]
    fn memory_usage_total() {
        let usage = MemoryUsage {
            weights_bytes: 1000,
            cache_bytes: 500,
            scratch_bytes: 200,
        };
        assert_eq!(usage.total(), 1700);
    }
}
