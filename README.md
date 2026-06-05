# ternary-sampler

**Sampling and triggering ternary patterns.** Statistical sampling strategies for ternary populations, from uniform random to importance-weighted.

## Why This Exists

In `ternary-tenforward`, conversations run for hundreds of ticks across dozens of agents. You can't analyze every data point — you need to sample. But how you sample matters enormously. Random sampling might miss rare but critical events (the agent that suddenly flips from agreeable to contrarian). Uniform sampling might underrepresent minority positions.

This crate implements five sampling strategies, each suited to different analytical needs:

1. **Random sampling** — quick and dirty, good for large populations where the law of large numbers applies
2. **Stratified sampling** — preserves the Neg/Zero/Pos ratio, essential when you need representative coverage
3. **Weighted sampling** — bias toward certain agents based on external importance weights
4. **Reservoir sampling** — single-pass streaming for unbounded populations
5. **Importance sampling** — oversample rare states to get better estimates of tail behavior

Plus statistics computation and sample size estimation, so you know whether your sample is big enough to trust.

## The Physics Behind It

### Why Sampling Theory Matters for Ternary Populations

A ternary population of N agents has 3^N possible states. For N=8 (a standard ten-forward session), that's 6,561 states. For N=16, it's 43 million. You cannot enumerate all of them. Sampling is the only tractable approach.

The key question: does your sampling strategy preserve the properties you care about?

- **Population mean** (average stance) — preserved by random sampling if the sample is large enough (CLT applies)
- **Population variance** (how divided the agents are) — needs stratified sampling to avoid underestimating
- **Rare events** (spontaneous state flips) — need importance sampling to detect

### Stratified Sampling and Population Balance

In a balanced ternary population (equal numbers of -1, 0, and +1), stratified sampling guarantees your sample has the same ratio. Random sampling might give you 80% +1 by chance, especially in small samples. For the ten-forward anti-monoculture analysis, stratified sampling is essential: you need to know whether the real population has become imbalanced, not whether your sample is.

### Reservoir Sampling for Streaming

`ReservoirSampler` implements the classic Algorithm R (Vitter, 1985). It maintains a fixed-size representative sample from a potentially infinite stream. Every element has equal probability of being in the final reservoir. This is what you use when agents are producing states tick after tick and you want a running sample without storing the entire history.

The algorithm is elegant: for each new element, include it in the reservoir with probability k/n (where k is reservoir capacity, n is elements seen so far). If included, it replaces a random existing element.

### Importance Sampling for Rare Events

In ternary dynamics, the interesting events are the rare ones: spontaneous state flips (the 5% mutation rate in ten-forward), dominance reversals, coalition formations. Importance sampling gives these rare events higher weight in the sample, improving statistical estimates for their frequency and impact.

### Sample Size Estimation

`estimate_sample_size` uses the standard formula `n = z²p(1-p)/e²` with p=0.5 (worst case) to determine how many samples you need for a given confidence level and margin of error. For 95% confidence with ±5% margin, you need ~385 samples. For ±1%, you need ~9,604.

## Key Types and Functions

```rust
/// A ternary value: -1, 0, or +1.
pub enum Ternary { Neg, Zero, Pos }

impl Ternary {
    pub fn to_i8(self) -> i8
    pub fn from_i8(v: i8) -> Option<Self>
}

/// Uniformly sample n items (with replacement).
pub fn random_sample(population: &[Ternary], n: usize) -> Vec<Ternary>

/// Sample while preserving Neg/Zero/Pos ratio.
pub fn stratified_sample(population: &[Ternary], n: usize) -> Vec<Ternary>

/// Sample with custom weights (CDF-based).
pub fn weighted_sample(population: &[Ternary], weights: &[f64], n: usize) -> Vec<Ternary>

/// Streaming reservoir sampler.
pub struct ReservoirSampler { /* ... */ }
impl ReservoirSampler {
    pub fn new(capacity: usize) -> Self
    pub fn push(&mut self, item: Ternary)
    pub fn into_vec(self) -> Vec<Ternary>
    pub fn len(&self) -> usize
    pub fn is_empty(&self) -> bool
}

/// Importance-weighted sampling for rare events.
pub fn importance_sample(population: &[Ternary], importance: &[f64], n: usize) -> Vec<Ternary>

/// Sample statistics: mean, variance, skewness, counts.
pub struct TernaryStats {
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub count_neg: usize,
    pub count_zero: usize,
    pub count_pos: usize,
    pub n: usize,
}

/// Compute statistics from a sample.
pub fn sample_statistics(sample: &[Ternary]) -> TernaryStats

/// Estimate required sample size for given confidence and margin.
pub fn estimate_sample_size(confidence: f64, margin: f64) -> usize
```

## Usage

### Random and Stratified Sampling

```rust
use ternary_sampler::{random_sample, stratified_sample, Ternary};

let population: Vec<Ternary> = (0..1000).map(|i| match i % 3 {
    0 => Ternary::Neg,
    1 => Ternary::Zero,
    _ => Ternary::Pos,
}).collect();

// Random: quick but may be biased
let sample = random_sample(&population, 100);

// Stratified: guaranteed ratio preservation
let balanced = stratified_sample(&population, 100);
// Will have ~33 Neg, ~34 Zero, ~33 Pos
```

### Weighted Sampling

```rust
use ternary_sampler::weighted_sample;

let pop = vec![Ternary::Neg, Ternary::Zero, Ternary::Pos];
let weights = vec![0.8, 0.1, 0.1];  // favor contrarian

let sample = weighted_sample(&pop, &weights, 50);
// ~80% of samples will be Neg
```

### Reservoir Sampling (Streaming)

```rust
use ternary_sampler::{ReservoirSampler, Ternary};

let mut sampler = ReservoirSampler::new(100);

// Process a stream of agent states
for tick in 0..10000 {
    let state = get_agent_state(tick);  // your function
    sampler.push(state);
}

// Get the representative sample
let sample = sampler.into_vec();
assert_eq!(sample.len(), 100);
```

### Statistics

```rust
use ternary_sampler::{stratified_sample, sample_statistics, estimate_sample_size};

let n = estimate_sample_size(0.95, 0.05);  // 385 for 95% CI, ±5%
let sample = stratified_sample(&population, n);
let stats = sample_statistics(&sample);

println!("Mean stance: {:.3}", stats.mean);        // -1 to +1
println!("Variance: {:.3}", stats.variance);        // 0 to 1
println!("Skewness: {:.3}", stats.skewness);        // asymmetry
println!("Distribution: {} Neg, {} Zero, {} Pos",
         stats.count_neg, stats.count_zero, stats.count_pos);
```

### Importance Sampling

```rust
use ternary_sampler::importance_sample;

// Assign high importance to rare contrarian states
let pop = vec![/* 100 agents */];
let importance: Vec<f64> = pop.iter().map(|t| match t {
    Ternary::Neg => 10.0,   // rare → boost
    Ternary::Zero => 1.0,
    Ternary::Pos => 1.0,
}).collect();

let focused = importance_sample(&pop, &importance, 50);
// Oversamples contrarian agents for better estimates
```

## In the Ternary Fleet

This is the **analysis layer** for the DJ metaphor product stack:

- `ternary-tenforward` — produces the agent state stream to be sampled
- **ternary-sampler** — extracts representative subsets for analysis
- `ternary-tempo` — can use sampled data for BPM estimation
- `ternary-needledrop` — perturbation analysis works on sampled populations

## References

- Reservoir sampling: Vitter, J.S. "Random Sampling with a Reservoir" (1985)
- Stratified sampling preserves population ratios — essential for balanced ternary analysis
- Sample size formula: `n = z²p(1-p)/e²` standard in survey statistics
- CLT: for large enough samples, the sample mean is normally distributed regardless of population distribution

## License

MIT
