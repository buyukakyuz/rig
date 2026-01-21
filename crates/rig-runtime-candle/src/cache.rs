use candle_core::{DType, Device, Result, Tensor};

use crate::config::TransformerConfig;

#[derive(Debug, Clone)]
pub struct CausalMaskCache {
    mask: Tensor,
    max_seq_len: usize,
}

impl CausalMaskCache {
    pub fn new(max_seq_len: usize, device: &Device) -> Result<Self> {
        let mask: Vec<u8> = (0..max_seq_len)
            .flat_map(|i| (0..max_seq_len).map(move |j| u8::from(j > i)))
            .collect();

        let mask = Tensor::from_slice(&mask, (max_seq_len, max_seq_len), device)?;

        Ok(Self { mask, max_seq_len })
    }

    pub fn get(&self, seq_len: usize) -> Result<Tensor> {
        if seq_len > self.max_seq_len {
            return Err(candle_core::Error::Msg(format!(
                "Sequence length {seq_len} exceeds max {}",
                self.max_seq_len
            )));
        }
        self.mask.narrow(0, 0, seq_len)?.narrow(1, 0, seq_len)
    }

    #[must_use]
    pub const fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

#[derive(Debug, Clone)]
pub struct RopeCache {
    cos: Tensor,
    sin: Tensor,
}

impl RopeCache {
    pub fn new(config: &TransformerConfig, dtype: DType, device: &Device) -> Result<Self> {
        let head_dim = config.head_dim();
        let max_seq_len = config.max_position_embeddings;
        let theta = config.rope_theta;

        Self::with_params(head_dim, max_seq_len, theta, dtype, device)
    }

    pub fn with_params(
        head_dim: usize,
        max_seq_len: usize,
        theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();

        let inv_freq = Tensor::new(inv_freq, device)?;

        let positions = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;

        let freqs = positions.matmul(&inv_freq.reshape((1, inv_freq.elem_count()))?)?;

        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin })
    }

    pub fn get(&self, seq_len: usize, offset: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        Ok((cos, sin))
    }

    #[must_use]
    pub fn cos(&self) -> &Tensor {
        &self.cos
    }

    #[must_use]
    pub fn sin(&self) -> &Tensor {
        &self.sin
    }

    pub fn max_seq_len(&self) -> Result<usize> {
        self.cos.dim(0)
    }
}

#[derive(Debug, Clone)]
pub struct LayerKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl LayerKvCache {
    #[must_use]
    pub const fn new() -> Self {
        Self { k: None, v: None }
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.k.is_none()
    }

    pub fn seq_len(&self) -> Result<usize> {
        match &self.k {
            Some(k) => Ok(k.dim(2)?),
            None => Ok(0),
        }
    }

    #[must_use]
    pub fn get(&self) -> Option<(&Tensor, &Tensor)> {
        match (&self.k, &self.v) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    pub fn update(&mut self, new_k: Tensor, new_v: Tensor) -> Result<(Tensor, Tensor)> {
        let (k, v) = match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => {
                let k = Tensor::cat(&[prev_k, &new_k], 2)?.contiguous()?;
                let v = Tensor::cat(&[prev_v, &new_v], 2)?.contiguous()?;
                (k, v)
            }
            _ => (new_k.contiguous()?, new_v.contiguous()?),
        };

        self.k = Some(k.clone());
        self.v = Some(v.clone());

        Ok((k, v))
    }

    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }
}

impl Default for LayerKvCache {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct PartitionKvCache {
    layers: Vec<LayerKvCache>,
}

impl PartitionKvCache {
    #[must_use]
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| LayerKvCache::new()).collect(),
        }
    }

    #[must_use]
    pub fn layer(&self, idx: usize) -> Option<&LayerKvCache> {
        self.layers.get(idx)
    }

    #[must_use]
    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut LayerKvCache> {
        self.layers.get_mut(idx)
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    pub fn seq_len(&self) -> Result<usize> {
        for layer in &self.layers {
            if !layer.is_empty() {
                return layer.seq_len();
            }
        }
        Ok(0)
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_cache_creation() {
        let device = Device::Cpu;
        let cache = RopeCache::with_params(128, 4096, 10000.0, DType::F32, &device)
            .unwrap_or_else(|e| panic!("Failed to create cache: {e}"));

        assert_eq!(cache.cos.dims(), &[4096, 64]);
        assert_eq!(cache.sin.dims(), &[4096, 64]);
    }

    #[test]
    fn test_rope_cache_get() {
        let device = Device::Cpu;
        let cache = RopeCache::with_params(128, 4096, 10000.0, DType::F32, &device)
            .unwrap_or_else(|e| panic!("Failed to create cache: {e}"));

        let (cos, sin) = cache
            .get(16, 0)
            .unwrap_or_else(|e| panic!("Failed to get: {e}"));
        assert_eq!(cos.dims(), &[16, 64]);
        assert_eq!(sin.dims(), &[16, 64]);

        let (cos, sin) = cache
            .get(8, 100)
            .unwrap_or_else(|e| panic!("Failed to get: {e}"));
        assert_eq!(cos.dims(), &[8, 64]);
        assert_eq!(sin.dims(), &[8, 64]);
    }

    #[test]
    fn test_layer_kv_cache() {
        let device = Device::Cpu;
        let mut cache = LayerKvCache::new();

        assert!(cache.is_empty());
        assert_eq!(
            cache
                .seq_len()
                .unwrap_or_else(|e| panic!("seq_len failed: {e}")),
            0
        );

        let k1 = Tensor::zeros((1, 4, 8, 32), DType::F32, &device)
            .unwrap_or_else(|e| panic!("Failed to create tensor: {e}"));
        let v1 = Tensor::zeros((1, 4, 8, 32), DType::F32, &device)
            .unwrap_or_else(|e| panic!("Failed to create tensor: {e}"));

        let (k, v) = cache
            .update(k1, v1)
            .unwrap_or_else(|e| panic!("update failed: {e}"));
        assert_eq!(k.dims(), &[1, 4, 8, 32]);
        assert_eq!(v.dims(), &[1, 4, 8, 32]);
        assert!(!cache.is_empty());

        let k2 = Tensor::zeros((1, 4, 4, 32), DType::F32, &device)
            .unwrap_or_else(|e| panic!("Failed to create tensor: {e}"));
        let v2 = Tensor::zeros((1, 4, 4, 32), DType::F32, &device)
            .unwrap_or_else(|e| panic!("Failed to create tensor: {e}"));

        let (k, v) = cache
            .update(k2, v2)
            .unwrap_or_else(|e| panic!("update failed: {e}"));
        assert_eq!(k.dims(), &[1, 4, 12, 32]);
        assert_eq!(v.dims(), &[1, 4, 12, 32]);
    }

    #[test]
    fn test_causal_mask_cache_prefill() {
        let device = Device::Cpu;
        let cache = CausalMaskCache::new(16, &device)
            .unwrap_or_else(|e| panic!("Failed to create mask cache: {e}"));

        let mask = cache
            .get(4)
            .unwrap_or_else(|e| panic!("Failed to get mask: {e}"));
        assert_eq!(mask.dims(), &[4, 4]);

        let expected = vec![0u8, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0];
        let mask_vec: Vec<u8> = mask
            .flatten_all()
            .unwrap_or_else(|e| panic!("flatten failed: {e}"))
            .to_vec1()
            .unwrap_or_else(|e| panic!("to_vec1 failed: {e}"));
        assert_eq!(mask_vec, expected);
    }

    #[test]
    fn test_causal_mask_cache_bounds() {
        let device = Device::Cpu;
        let cache = CausalMaskCache::new(8, &device)
            .unwrap_or_else(|e| panic!("Failed to create mask cache: {e}"));

        assert!(cache.get(8).is_ok());

        assert!(cache.get(9).is_err());
    }
}
