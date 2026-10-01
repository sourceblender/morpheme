//! Shared helpers for the benchmarks: deterministic corpora and fixture
//! loading.

#![allow(dead_code)]

use std::path::PathBuf;

use splinter::Tokenizer;

/// Small deterministic PRNG (xorshift64*), so benchmark inputs are the
/// same on every run without pulling in `rand`.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const WORDS: &str = include_str!("../../../../examples/corpus.txt");

/// English-like sentences built from the words of `examples/corpus.txt`,
/// with a sprinkle of accented, CJK and emoji text.
pub fn sentences(n: usize, seed: u64) -> Vec<String> {
    let words: Vec<&str> = WORDS.split_whitespace().collect();
    let extras = [
        "café", "naïve", "Zürich", "你好", "世界", "😀", "→", "1,000", "3.14",
    ];
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let len = 5 + rng.below(30);
            let mut s = String::new();
            for i in 0..len {
                if i > 0 {
                    s.push(' ');
                }
                if rng.below(20) == 0 {
                    s.push_str(extras[rng.below(extras.len())]);
                } else {
                    s.push_str(words[rng.below(words.len())]);
                }
            }
            s
        })
        .collect()
}

/// About `bytes` of synthetic training text over a vocabulary of
/// `n_words` distinct made-up words (Zipf-ish frequencies).
pub fn training_corpus(bytes: usize, n_words: usize, seed: u64) -> Vec<String> {
    const SYLLABLES: &[&str] = &[
        "ka", "to", "ri", "men", "sa", "lo", "ver", "qui", "den", "sha", "pol", "tru", "ex", "an",
        "ion", "ing", "str", "bra", "gel", "mu", "zo", "pe", "nik", "dor",
    ];
    let mut rng = Rng::new(seed);
    let vocab: Vec<String> = (0..n_words)
        .map(|_| {
            (0..1 + rng.below(5))
                .map(|_| SYLLABLES[rng.below(SYLLABLES.len())])
                .collect()
        })
        .collect();
    let mut lines = Vec::new();
    let mut total = 0;
    while total < bytes {
        let len = 5 + rng.below(20);
        let line: Vec<&str> = (0..len)
            .map(|_| {
                // Square the uniform draw to skew towards frequent words.
                let r = rng.below(n_words);
                vocab[r * r / n_words].as_str()
            })
            .collect();
        let line = line.join(" ");
        total += line.len() + 1;
        lines.push(line);
    }
    lines
}

/// Load a downloaded Hugging Face fixture (see
/// `scripts/fetch-hf-fixtures.sh`), or `None` with a note if missing.
pub fn fixture(name: &str) -> Option<Tokenizer> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/hf")
        .join(format!("{name}.json"));
    if !path.exists() {
        eprintln!(
            "skipping {name}: {} missing (run scripts/fetch-hf-fixtures.sh)",
            path.display()
        );
        return None;
    }
    let mut tok = Tokenizer::from_file(&path).expect("fixture loads");
    // Benchmark the raw pipeline, not the fixture's padding settings.
    tok.set_padding(None);
    tok.set_truncation(None).expect("disabling truncation");
    Some(tok)
}

/// The pretrained tokenizers benchmarked, one per model family.
pub const FIXTURES: &[&str] = &["bert-base-uncased", "gpt2", "llama", "t5-small"];
