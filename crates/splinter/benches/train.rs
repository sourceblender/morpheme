//! Training time for BPE, WordPiece and Unigram on a deterministic
//! ~1.5 MB synthetic corpus.
//!
//! `cargo bench --bench train`

mod common;

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use splinter::models::{Bpe, Unigram, WordPiece};
use splinter::normalizers::{BertNormalizer, Nfkc};
use splinter::pre_tokenizers::{BertPreTokenizer, ByteLevel, Metaspace, PrependScheme};
use splinter::trainers::{BpeTrainer, UnigramTrainer, WordPieceTrainer};
use splinter::{AddedToken, Tokenizer};

const VOCAB: usize = 4_000;

fn bench_train(c: &mut Criterion) {
    let corpus = common::training_corpus(1_500_000, 20_000, 1234);
    let bytes: usize = corpus.iter().map(|l| l.len() + 1).sum();

    let mut group = c.benchmark_group("train");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("bpe_byte_level", |b| {
        b.iter(|| {
            let mut tok = Tokenizer::new(Bpe::default())
                .with_pre_tokenizer(ByteLevel::new(false, true, true));
            let trainer = BpeTrainer::builder()
                .vocab_size(VOCAB)
                .initial_alphabet(ByteLevel::alphabet())
                .show_progress(false)
                .build()
                .unwrap();
            tok.train(trainer, corpus.iter()).unwrap();
            black_box(tok)
        })
    });

    group.bench_function("wordpiece_bert", |b| {
        b.iter(|| {
            let mut tok = Tokenizer::new(WordPiece::default())
                .with_normalizer(BertNormalizer::default())
                .with_pre_tokenizer(BertPreTokenizer);
            let trainer = WordPieceTrainer::builder()
                .vocab_size(VOCAB)
                .special_tokens(vec![AddedToken::new("[UNK]", true)])
                .show_progress(false)
                .build()
                .unwrap();
            tok.train(trainer, corpus.iter()).unwrap();
            black_box(tok)
        })
    });

    group.bench_function("unigram_sentencepiece", |b| {
        b.iter(|| {
            let mut tok = Tokenizer::new(Unigram::default())
                .with_normalizer(Nfkc)
                .with_pre_tokenizer(Metaspace::new('▁', PrependScheme::Always, true));
            let trainer = UnigramTrainer::builder()
                .vocab_size(VOCAB)
                .special_tokens(vec![AddedToken::new("<unk>", true)])
                .unk_token("<unk>")
                .show_progress(false)
                .build()
                .unwrap();
            tok.train(trainer, corpus.iter()).unwrap();
            black_box(tok)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_train);
criterion_main!(benches);
