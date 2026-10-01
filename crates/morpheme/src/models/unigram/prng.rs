//! Tiny deterministic PRNG for Unigram sampling (no external dependency).
//!
//! Seeding uses splitmix64 (so nearby seeds give unrelated streams) and
//! the stream itself is xorshift64*.

use std::cell::Cell;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};

/// A seedable xorshift64* generator.
#[derive(Debug, Clone)]
pub(crate) struct Prng(u64);

/// One splitmix64 step on `state`, returning the output.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

thread_local! {
    static ENTROPY: Cell<u64> = Cell::new({
        let mut h = RandomState::new().build_hasher();
        h.write_u64(0x5851_F42D_4C95_7F2D);
        h.finish()
    });
}

impl Prng {
    /// A generator whose stream is a pure function of `seed` and `salt`.
    pub(crate) fn seeded(seed: u64, salt: u64) -> Self {
        let mut state = seed;
        let a = splitmix64(&mut state);
        state ^= salt;
        let b = splitmix64(&mut state);
        // xorshift64* needs a non-zero state.
        Self((a ^ b.rotate_left(32)) | 1)
    }

    /// A generator seeded from a thread-local entropy pool.
    pub(crate) fn from_entropy() -> Self {
        let seed = ENTROPY.with(|e| {
            let mut state = e.get();
            let out = splitmix64(&mut state);
            e.set(state);
            out
        });
        Self::seeded(seed, 0)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform float in `[0, 1)`.
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Index into `weights` chosen with probability proportional to the
    /// weight (non-finite and negative weights count as zero). Returns
    /// the last index if the weights do not sum to a positive number.
    pub(crate) fn choose_weighted(&mut self, weights: &[f64]) -> usize {
        if weights.is_empty() {
            return 0;
        }
        let usable = |w: &f64| w.is_finite() && *w > 0.0;
        let total: f64 = weights.iter().filter(|w| usable(w)).sum();
        if total <= 0.0 {
            return weights.len() - 1;
        }
        let mut target = self.next_f64() * total;
        for (i, w) in weights.iter().enumerate() {
            if !usable(w) {
                continue;
            }
            if target < *w {
                return i;
            }
            target -= w;
        }
        weights.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(seed: u64, salt: u64) -> Vec<f64> {
        let mut rng = Prng::seeded(seed, salt);
        (0..8).map(|_| rng.next_f64()).collect()
    }

    #[test]
    fn seeded_streams_are_reproducible_and_differ_by_salt() {
        let a = stream(7, 1);
        assert_eq!(a, stream(7, 1));
        assert_ne!(a, stream(7, 2));
        assert_ne!(a, stream(8, 1));
        assert!(a.iter().all(|x| (0.0..1.0).contains(x)));
    }

    #[test]
    fn choose_weighted_respects_zero_weights() {
        let mut rng = Prng::seeded(1, 0);
        for _ in 0..100 {
            assert_eq!(rng.choose_weighted(&[0.0, 1.0, 0.0]), 1);
        }
        assert_eq!(rng.choose_weighted(&[0.0, 0.0]), 1);
        assert_eq!(rng.choose_weighted(&[]), 0);
    }
}
