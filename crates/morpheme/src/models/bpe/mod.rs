//! Byte-Pair Encoding.
//!
//! Semantics follow Hugging Face `tokenizers`: a word is split into
//! chars (with an optional `continuing_subword_prefix` on non-initial
//! chars and an optional `end_of_word_suffix` on the last one), unknown
//! chars become `<0xNN>` byte tokens (`byte_fallback`) or `unk_token`,
//! then merges are applied lowest-rank first.

mod model;
mod serialization;
mod word;

#[cfg(test)]
mod fixture_tests;

pub use model::{Bpe, BpeBuilder, Merges, Vocab};
pub(crate) use serialization::{OrderedVocab, reverse_vocab};
pub(crate) use word::{MergeMap, Word};

/// A pair of token ids.
pub(crate) type Pair = (u32, u32);

/// Tiny thread-local PRNG for BPE dropout (no external dependency).
pub(crate) mod prng {
    use std::cell::Cell;
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    thread_local! {
        static STATE: Cell<u64> = Cell::new({
            let mut h = RandomState::new().build_hasher();
            h.write_u64(0x9E37_79B9_7F4A_7C15);
            h.finish() | 1
        });
    }

    /// Uniform float in `[0, 1)` (xorshift64*).
    pub(crate) fn next_f32() -> f32 {
        STATE.with(|s| {
            let mut x = s.get();
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            s.set(x);
            let r = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
            (r >> 40) as f32 / (1u64 << 24) as f32
        })
    }
}
