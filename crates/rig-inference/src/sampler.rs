use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rig_core::GenerationParams;

pub struct Sampler {
    temperature: f32,
    top_p: f32,
    top_k: usize,
    rng: StdRng,
}

impl Sampler {
    #[must_use]
    pub fn new(params: &GenerationParams, seed: Option<u64>) -> Self {
        let rng = seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);

        Self {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            rng,
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        if self.temperature <= 0.0 {
            return Self::argmax(logits);
        }

        let scaled: Vec<f32> = logits.iter().map(|&l| l / self.temperature).collect();
        let probs = Self::softmax(&scaled);
        let (indices, probs) = self.top_k_filter(&probs);
        let (indices, probs) = self.top_p_filter(&indices, &probs);
        let probs = Self::normalize(&probs);
        self.categorical_sample(&indices, &probs)
    }

    fn argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| u32::try_from(i).unwrap_or(0))
    }

    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        if sum > 0.0 {
            exp.iter().map(|&e| e / sum).collect()
        } else {
            #[allow(clippy::cast_precision_loss)]
            let uniform = 1.0 / logits.len() as f32;
            vec![uniform; logits.len()]
        }
    }

    fn top_k_filter(&self, probs: &[f32]) -> (Vec<usize>, Vec<f32>) {
        if self.top_k == 0 || self.top_k >= probs.len() {
            return ((0..probs.len()).collect(), probs.to_vec());
        }

        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        indexed.truncate(self.top_k);

        let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
        let probs: Vec<f32> = indexed.iter().map(|(_, p)| *p).collect();

        (indices, probs)
    }

    fn top_p_filter(&self, indices: &[usize], probs: &[f32]) -> (Vec<usize>, Vec<f32>) {
        if self.top_p >= 1.0 {
            return (indices.to_vec(), probs.to_vec());
        }

        let mut indexed: Vec<(usize, f32)> =
            indices.iter().copied().zip(probs.iter().copied()).collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0;
        let mut keep = Vec::new();
        for &(i, p) in &indexed {
            keep.push((i, p));
            cumsum += p;
            if cumsum >= self.top_p {
                break;
            }
        }

        if keep.is_empty() && !indexed.is_empty() {
            keep.push(indexed[0]);
        }

        let indices: Vec<usize> = keep.iter().map(|(i, _)| *i).collect();
        let probs: Vec<f32> = keep.iter().map(|(_, p)| *p).collect();

        (indices, probs)
    }

    fn normalize(probs: &[f32]) -> Vec<f32> {
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            probs.iter().map(|&p| p / sum).collect()
        } else if probs.is_empty() {
            Vec::new()
        } else {
            #[allow(clippy::cast_precision_loss)]
            let uniform = 1.0 / probs.len() as f32;
            vec![uniform; probs.len()]
        }
    }

    fn categorical_sample(&mut self, indices: &[usize], probs: &[f32]) -> u32 {
        if indices.is_empty() {
            return 0;
        }

        let r: f32 = self.rng.gen_range(0.0..1.0);
        let mut cumsum = 0.0;

        for (&idx, &prob) in indices.iter().zip(probs.iter()) {
            cumsum += prob;
            if r < cumsum {
                return u32::try_from(idx).unwrap_or(0);
            }
        }

        u32::try_from(*indices.last().unwrap_or(&0)).unwrap_or(0)
    }
}

impl std::fmt::Debug for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sampler")
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("top_k", &self.top_k)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn test_params() -> GenerationParams {
        GenerationParams::default()
    }

    #[test]
    fn test_greedy_sampling() {
        let params = test_params().with_temperature(0.0);
        let mut sampler = Sampler::new(&params, Some(42));

        let logits = vec![1.0, 2.0, 5.0, 0.5, 3.0];
        let token = sampler.sample(&logits);

        assert_eq!(token, 2);

        let token2 = sampler.sample(&logits);
        assert_eq!(token2, 2);
    }

    #[test]
    fn test_temperature_sampling_produces_valid_token() {
        let params = test_params().with_temperature(0.7);
        let mut sampler = Sampler::new(&params, Some(42));

        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let token = sampler.sample(&logits);

        assert!(token < 4);
    }

    #[test]
    fn test_top_k_filtering() {
        let params = test_params().with_temperature(1.0).with_top_k(2);
        let mut sampler = Sampler::new(&params, Some(42));

        let logits = vec![0.1, 5.0, 4.0, 0.2];

        for _ in 0..100 {
            let token = sampler.sample(&logits);
            assert!(token == 1 || token == 2, "Token {token} not in top-2");
        }
    }

    #[test]
    fn test_empty_logits() {
        let params = test_params();
        let mut sampler = Sampler::new(&params, Some(42));

        let token = sampler.sample(&[]);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_single_logit() {
        let params = test_params();
        let mut sampler = Sampler::new(&params, Some(42));

        let token = sampler.sample(&[5.0]);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = Sampler::softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax sum is {sum}");

        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_top_p_filtering() {
        let params = test_params().with_temperature(1.0).with_top_p(0.5);
        let mut sampler = Sampler::new(&params, Some(42));

        let logits = vec![0.0, 0.0, 10.0, 0.0];

        let mut counts = [0u32; 4];
        for _ in 0..100 {
            let token = sampler.sample(&logits);
            if token < 4 {
                counts[token as usize] += 1;
            }
        }

        assert!(counts[2] > 90, "Token 2 count: {}", counts[2]);
    }

    #[test]
    fn test_determinism_with_seed() {
        let params = test_params().with_temperature(0.7);

        let mut sampler1 = Sampler::new(&params, Some(12345));
        let mut sampler2 = Sampler::new(&params, Some(12345));

        let logits = vec![1.0, 2.0, 3.0, 2.5, 1.5];

        for _ in 0..10 {
            let t1 = sampler1.sample(&logits);
            let t2 = sampler2.sample(&logits);
            assert_eq!(t1, t2, "Determinism failed with same seed");
        }
    }
}
