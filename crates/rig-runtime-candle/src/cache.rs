use candle_core::{DType, Device, Result, Tensor};

use crate::config::TransformerConfig;

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
    k_buffer: Option<Tensor>,
    v_buffer: Option<Tensor>,
    current_len: usize,
    max_len: usize,
}

impl LayerKvCache {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            k_buffer: None,
            v_buffer: None,
            current_len: 0,
            max_len: 0,
        }
    }

    pub fn init_buffers(
        &mut self,
        batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<()> {
        if self.k_buffer.is_none() {
            let shape = (batch_size, num_heads, max_seq_len, head_dim);
            self.k_buffer = Some(Tensor::zeros(shape, dtype, device)?);
            self.v_buffer = Some(Tensor::zeros(shape, dtype, device)?);
            self.max_len = max_seq_len;
            self.current_len = 0;
        }
        Ok(())
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    #[must_use]
    pub const fn is_initialized(&self) -> bool {
        self.k_buffer.is_some()
    }

    #[must_use]
    pub const fn seq_len(&self) -> usize {
        self.current_len
    }

    #[must_use]
    pub fn get(&self) -> Option<(Tensor, Tensor)> {
        match (&self.k_buffer, &self.v_buffer) {
            (Some(k_buf), Some(v_buf)) if self.current_len > 0 => {
                let k = k_buf.narrow(2, 0, self.current_len).ok()?;
                let v = v_buf.narrow(2, 0, self.current_len).ok()?;
                Some((k, v))
            }
            _ => None,
        }
    }

    pub fn update(&mut self, new_k: Tensor, new_v: Tensor) -> Result<(Tensor, Tensor)> {
        let new_seq_len = new_k.dim(2)?;

        match (&self.k_buffer, &self.v_buffer) {
            (Some(k_buf), Some(v_buf)) => {
                let new_k = new_k.contiguous()?;
                let new_v = new_v.contiguous()?;

                k_buf.slice_set(&new_k, 2, self.current_len)?;
                v_buf.slice_set(&new_v, 2, self.current_len)?;

                let new_len = self.current_len + new_seq_len;
                self.current_len = new_len;

                let k = k_buf.narrow(2, 0, new_len)?;
                let v = v_buf.narrow(2, 0, new_len)?;
                Ok((k, v))
            }
            _ => Ok((new_k.contiguous()?, new_v.contiguous()?)),
        }
    }

    pub fn clear(&mut self) {
        self.current_len = 0;
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

    pub fn init_buffers(
        &mut self,
        batch_size: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<()> {
        for layer in &mut self.layers {
            layer.init_buffers(
                batch_size,
                num_kv_heads,
                max_seq_len,
                head_dim,
                dtype,
                device,
            )?;
        }
        Ok(())
    }

    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.layers
            .first()
            .is_some_and(LayerKvCache::is_initialized)
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

    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.layers.first().map_or(0, LayerKvCache::seq_len)
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
        assert!(!cache.is_initialized());
        assert_eq!(cache.seq_len(), 0);

        cache
            .init_buffers(1, 4, 100, 32, DType::F32, &device)
            .unwrap_or_else(|e| panic!("init_buffers failed: {e}"));
        assert!(cache.is_initialized());
        assert!(cache.is_empty());

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
        assert_eq!(cache.seq_len(), 8);

        let k2 = Tensor::zeros((1, 4, 4, 32), DType::F32, &device)
            .unwrap_or_else(|e| panic!("Failed to create tensor: {e}"));
        let v2 = Tensor::zeros((1, 4, 4, 32), DType::F32, &device)
            .unwrap_or_else(|e| panic!("Failed to create tensor: {e}"));

        let (k, v) = cache
            .update(k2, v2)
            .unwrap_or_else(|e| panic!("update failed: {e}"));
        assert_eq!(k.dims(), &[1, 4, 12, 32]);
        assert_eq!(v.dims(), &[1, 4, 12, 32]);
        assert_eq!(cache.seq_len(), 12);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.is_initialized());
    }
}
