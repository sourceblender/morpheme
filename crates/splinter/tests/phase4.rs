//! Performance benchmarks for the v0.1 trainers and models.
//!
//! Run with `cargo bench --bench phase4` (requires the `bench`
//! harness). The "test" binary can also be exercised via
//! `cargo test --release -- --nocapture phase4_bench` for a
//! quick sanity check.

use std::time::Instant;

use splinter::trainer::{BpeTrainer, Trainer, UnigramTrainer};

/// Generate a synthetic corpus by repeating `seed` until we hit
/// `target_words` distinct whitespace-separated tokens. The
/// resulting distribution has heavy repeats — good for trainers that
/// use word frequencies.
fn synthetic_corpus(seed: &[&str], target_words: usize) -> String {
    let mut out = String::new();
    let mut count = 0usize;
    while count < target_words {
        for w in seed {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(w);
            count += 1;
            if count >= target_words {
                break;
            }
        }
    }
    out
}

#[test]
fn phase4_bench_bpe_trainer() {
    // ~200 unique tokens, ~200k word occurrences. Small enough to
    // run in CI; the BPE trainer should finish in well under a
    // second.
    let seed: Vec<&str> = (0..200)
        .map(|i| match i % 4 {
            0 => "the",
            1 => "quick",
            2 => "brown",
            _ => "fox",
        })
        .collect();
    let corpus = synthetic_corpus(&seed, 200_000);

    let t = Instant::now();
    let trainer = BpeTrainer::builder(300).build();
    let _m = trainer.train(corpus.as_str()).expect("train ok");
    let elapsed = t.elapsed();
    println!(
        "BpeTrainer: vocab=300, corpus={} words, elapsed={:?}",
        200_000, elapsed
    );
    // Sanity bound — on any reasonable machine this is sub-second.
    // If this fires in CI we want to know, but not as a hard fail.
    assert!(
        elapsed.as_secs() < 60,
        "BPE trainer took too long: {elapsed:?}"
    );
}

#[test]
fn phase4_bench_unigram_trainer() {
    let seed: Vec<&str> = (0..100)
        .map(|i| match i % 5 {
            0 => "alpha",
            1 => "beta",
            2 => "gamma",
            3 => "delta",
            _ => "epsilon",
        })
        .collect();
    let corpus = synthetic_corpus(&seed, 50_000);

    let t = Instant::now();
    let trainer = UnigramTrainer::builder(60).build();
    let _m = trainer.train(corpus.as_str()).expect("train ok");
    let elapsed = t.elapsed();
    println!(
        "UnigramTrainer: vocab=60, corpus={} words, elapsed={:?}",
        50_000, elapsed
    );
    assert!(
        elapsed.as_secs() < 120,
        "Unigram trainer took too long: {elapsed:?}"
    );
}
