#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
}

impl SamplingParams {
    #[must_use]
    pub const fn new(temperature: f32, top_p: f32, top_k: usize, seed: u64) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            seed,
        }
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SampleResult {
    pub token: u32,
}

impl SampleResult {
    #[must_use]
    pub const fn new(token: u32) -> Self {
        Self { token }
    }
}
