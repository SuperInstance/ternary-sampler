#![forbid(unsafe_code)]

//! Sampling strategies for ternary (-1, 0, +1) populations.

use rand::Rng;

/// A ternary value: -1, 0, or +1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ternary {
    Neg,
    Zero,
    Pos,
}

impl Ternary {
    pub fn to_i8(self) -> i8 {
        match self {
            Ternary::Neg => -1,
            Ternary::Zero => 0,
            Ternary::Pos => 1,
        }
    }

    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Ternary::Neg),
            0 => Some(Ternary::Zero),
            1 => Some(Ternary::Pos),
            _ => None,
        }
    }
}

// ── Random sampling ────────────────────────────────────────────────

/// Uniformly sample `n` items from `population` (with replacement).
pub fn random_sample(population: &[Ternary], n: usize) -> Vec<Ternary> {
    if population.is_empty() || n == 0 {
        return Vec::new();
    }
    let mut rng = rand::thread_rng();
    (0..n).map(|_| population[rng.gen_range(0..population.len())]).collect()
}

// ── Stratified sampling ────────────────────────────────────────────

/// Sample while preserving the ratio of Neg/Zero/Pos in the population.
pub fn stratified_sample(population: &[Ternary], n: usize) -> Vec<Ternary> {
    if population.is_empty() || n == 0 {
        return Vec::new();
    }
    let counts = count_ternary(population);
    let total = population.len() as f64;
    let n_f = n as f64;

    let neg_n = ((counts.neg as f64 / total) * n_f).round() as usize;
    let pos_n = ((counts.pos as f64 / total) * n_f).round() as usize;
    let zero_n = n.saturating_sub(neg_n).saturating_sub(pos_n);

    let mut sample = Vec::with_capacity(n);
    sample.extend(std::iter::repeat(Ternary::Neg).take(neg_n));
    sample.extend(std::iter::repeat(Ternary::Zero).take(zero_n));
    sample.extend(std::iter::repeat(Ternary::Pos).take(pos_n));
    sample
}

// ── Weighted sampling ──────────────────────────────────────────────

/// Sample `n` items, each chosen with probability proportional to `weights[i]`.
/// Returns empty vec if lengths differ or total weight is zero.
pub fn weighted_sample(population: &[Ternary], weights: &[f64], n: usize) -> Vec<Ternary> {
    if population.len() != weights.len() || population.is_empty() || n == 0 {
        return Vec::new();
    }
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return Vec::new();
    }
    let cdf: Vec<f64> = weights.iter().scan(0.0, |acc, &w| { *acc += w / total; Some(*acc) }).collect();
    let mut rng = rand::thread_rng();
    (0..n).map(|_| {
        let r: f64 = rng.gen();
        let idx = cdf.partition_point(|&c| c < r);
        population[idx.min(population.len() - 1)]
    }).collect()
}

// ── Reservoir sampling (streaming) ─────────────────────────────────

/// Reservoir sampler that maintains a fixed-size sample from a stream.
pub struct ReservoirSampler {
    reservoir: Vec<Ternary>,
    capacity: usize,
    seen: usize,
}

impl ReservoirSampler {
    pub fn new(capacity: usize) -> Self {
        Self { reservoir: Vec::with_capacity(capacity), capacity, seen: 0 }
    }

    pub fn push(&mut self, item: Ternary) {
        self.seen += 1;
        if self.reservoir.len() < self.capacity {
            self.reservoir.push(item);
        } else {
            let mut rng = rand::thread_rng();
            let j: usize = rng.gen_range(0..self.seen);
            if j < self.capacity {
                self.reservoir[j] = item;
            }
        }
    }

    pub fn into_vec(self) -> Vec<Ternary> {
        self.reservoir
    }

    pub fn len(&self) -> usize {
        self.reservoir.len()
    }

    pub fn is_empty(&self) -> bool {
        self.reservoir.is_empty()
    }
}

// ── Importance sampling ────────────────────────────────────────────

/// Sample rare states more frequently. Items with `importance[i]` weight are
/// sampled proportionally to their importance.
pub fn importance_sample(population: &[Ternary], importance: &[f64], n: usize) -> Vec<Ternary> {
    // Identical mechanism to weighted_sample but semantically distinct
    weighted_sample(population, importance, n)
}

// ── Sample statistics ──────────────────────────────────────────────

/// Statistics of a ternary distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct TernaryStats {
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub count_neg: usize,
    pub count_zero: usize,
    pub count_pos: usize,
    pub n: usize,
}

struct Counts { neg: usize, zero: usize, pos: usize }

fn count_ternary(pop: &[Ternary]) -> Counts {
    let mut c = Counts { neg: 0, zero: 0, pos: 0 };
    for t in pop {
        match t {
            Ternary::Neg => c.neg += 1,
            Ternary::Zero => c.zero += 1,
            Ternary::Pos => c.pos += 1,
        }
    }
    c
}

/// Compute mean, variance, and skewness of a ternary sample.
pub fn sample_statistics(sample: &[Ternary]) -> TernaryStats {
    let counts = count_ternary(sample);
    let n = sample.len() as f64;
    let mean = (counts.pos as f64 - counts.neg as f64) / n;
    // variance of ternary: E[X^2] - E[X]^2, X in {-1,0,1}
    let e_sq = (counts.neg as f64 + counts.pos as f64) / n;
    let variance = e_sq - mean * mean;
    // skewness: E[(X-μ)^3] / σ^3
    let std_dev = variance.sqrt().max(f64::EPSILON);
    let skewness = ((-1.0 - mean).powi(3) * counts.neg as f64
        + (0.0 - mean).powi(3) * counts.zero as f64
        + (1.0 - mean).powi(3) * counts.pos as f64)
        / (n * std_dev.powi(3));
    TernaryStats {
        mean,
        variance,
        skewness,
        count_neg: counts.neg,
        count_zero: counts.zero,
        count_pos: counts.pos,
        n: sample.len(),
    }
}

// ── Sample size estimation ─────────────────────────────────────────

/// Estimate required sample size for a given confidence level and margin of error.
/// Uses simplified formula: n = z^2 * p * (1-p) / e^2 for binary approximation.
pub fn estimate_sample_size(confidence: f64, margin: f64) -> usize {
    // z-score approximation for common confidence levels
    let z = match confidence {
        c if c >= 0.99 => 2.576,
        c if c >= 0.95 => 1.96,
        c if c >= 0.90 => 1.645,
        c if c >= 0.80 => 1.282,
        _ => 1.0,
    };
    let p = 0.5; // worst-case proportion
    let n = (z * z * p * (1.0 - p)) / (margin * margin);
    n.ceil() as usize
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn all_neg(n: usize) -> Vec<Ternary> { vec![Ternary::Neg; n] }
    fn all_pos(n: usize) -> Vec<Ternary> { vec![Ternary::Pos; n] }
    fn mixed() -> Vec<Ternary> {
        use Ternary::*;
        [Neg, Zero, Pos, Neg, Zero, Pos, Neg, Zero, Pos, Neg, Zero, Pos].to_vec()
    }

    #[test]
    fn test_ternary_conversion() {
        assert_eq!(Ternary::Neg.to_i8(), -1);
        assert_eq!(Ternary::Zero.to_i8(), 0);
        assert_eq!(Ternary::Pos.to_i8(), 1);
        assert_eq!(Ternary::from_i8(-1), Some(Ternary::Neg));
        assert_eq!(Ternary::from_i8(0), Some(Ternary::Zero));
        assert_eq!(Ternary::from_i8(1), Some(Ternary::Pos));
        assert_eq!(Ternary::from_i8(2), None);
    }

    #[test]
    fn test_random_sample_length() {
        let pop = mixed();
        let s = random_sample(&pop, 5);
        assert_eq!(s.len(), 5);
        for t in &s {
            assert!(matches!(t, Ternary::Neg | Ternary::Zero | Ternary::Pos));
        }
    }

    #[test]
    fn test_random_sample_empty() {
        assert!(random_sample(&[], 10).is_empty());
        assert!(random_sample(&mixed(), 0).is_empty());
    }

    #[test]
    fn test_stratified_preserves_ratio() {
        let pop = mixed(); // equal ratio
        let s = stratified_sample(&pop, 300);
        let stats = sample_statistics(&s);
        let ratio_neg = stats.count_neg as f64 / s.len() as f64;
        let ratio_pos = stats.count_pos as f64 / s.len() as f64;
        assert!((ratio_neg - 1.0 / 3.0).abs() < 0.05);
        assert!((ratio_pos - 1.0 / 3.0).abs() < 0.05);
    }

    #[test]
    fn test_stratified_single_class() {
        let pop = all_neg(10);
        let s = stratified_sample(&pop, 5);
        assert!(s.iter().all(|t| *t == Ternary::Neg));
    }

    #[test]
    fn test_weighted_sample_biased() {
        let pop = vec![Ternary::Neg, Ternary::Pos];
        let weights = vec![0.99, 0.01];
        let s = weighted_sample(&pop, &weights, 100);
        let neg_count = s.iter().filter(|t| **t == Ternary::Neg).count();
        assert!(neg_count > 80); // heavily biased toward Neg
    }

    #[test]
    fn test_weighted_sample_mismatched_lengths() {
        assert!(weighted_sample(&[Ternary::Zero], &[1.0, 2.0], 5).is_empty());
    }

    #[test]
    fn test_weighted_sample_zero_total() {
        let pop = vec![Ternary::Neg, Ternary::Pos];
        assert!(weighted_sample(&pop, &[0.0, 0.0], 5).is_empty());
    }

    #[test]
    fn test_reservoir_sampler() {
        let mut sampler = ReservoirSampler::new(3);
        let items: Vec<Ternary> = (0..100).map(|i| {
            if i % 3 == 0 { Ternary::Neg } else if i % 3 == 1 { Ternary::Zero } else { Ternary::Pos }
        }).collect();
        for t in &items {
            sampler.push(*t);
        }
        assert_eq!(sampler.len(), 3);
        let v = sampler.into_vec();
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_reservoir_small_input() {
        let mut sampler = ReservoirSampler::new(10);
        sampler.push(Ternary::Pos);
        sampler.push(Ternary::Neg);
        assert_eq!(sampler.len(), 2);
    }

    #[test]
    fn test_importance_sample() {
        let pop = vec![Ternary::Neg, Ternary::Zero, Ternary::Pos];
        let imp = vec![10.0, 1.0, 1.0];
        let s = importance_sample(&pop, &imp, 50);
        assert_eq!(s.len(), 50);
        let neg_count = s.iter().filter(|t| **t == Ternary::Neg).count();
        assert!(neg_count > 25);
    }

    #[test]
    fn test_statistics_all_pos() {
        let stats = sample_statistics(&all_pos(100));
        assert!((stats.mean - 1.0).abs() < 1e-9);
        assert!((stats.variance).abs() < 1e-9);
        assert_eq!(stats.count_pos, 100);
    }

    #[test]
    fn test_statistics_balanced() {
        let pop: Vec<Ternary> = (0..999).map(|i| match i % 3 {
            0 => Ternary::Neg,
            1 => Ternary::Zero,
            _ => Ternary::Pos,
        }).collect();
        let stats = sample_statistics(&pop);
        assert!(stats.mean.abs() < 0.01);
        assert!(stats.skewness.abs() < 0.01);
    }

    #[test]
    fn test_statistics_empty() {
        let stats = sample_statistics(&[]);
        assert!(stats.mean.is_nan() || stats.n == 0);
    }

    #[test]
    fn test_estimate_sample_size() {
        let n = estimate_sample_size(0.95, 0.05);
        assert!(n >= 300 && n <= 500); // ~385
        let n2 = estimate_sample_size(0.99, 0.05);
        assert!(n2 > n);
        let n3 = estimate_sample_size(0.95, 0.01);
        assert!(n3 > n);
    }
}
