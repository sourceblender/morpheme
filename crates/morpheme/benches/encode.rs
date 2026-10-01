//! Encode and decode throughput on real pretrained tokenizers.
//!
//! `cargo bench --bench encode`. Needs `scripts/fetch-hf-fixtures.sh`;
//! missing fixtures are skipped.

mod common;

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn bench_encode(c: &mut Criterion) {
    let lines = common::sentences(2_000, 42);
    let bytes: usize = lines.iter().map(|l| l.len()).sum();
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();

    let mut group = c.benchmark_group("encode");
    group.throughput(Throughput::Bytes(bytes as u64));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    for name in common::FIXTURES {
        let Some(tok) = common::fixture(name) else {
            continue;
        };
        group.bench_with_input(BenchmarkId::new("sequential", name), &refs, |b, refs| {
            b.iter(|| {
                for line in refs {
                    black_box(tok.encode(*line, true).unwrap());
                }
            })
        });
        group.bench_with_input(BenchmarkId::new("batch", name), &refs, |b, refs| {
            b.iter(|| black_box(tok.encode_batch(refs.clone(), true).unwrap()))
        });
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let lines = common::sentences(2_000, 7);
    let mut group = c.benchmark_group("decode");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    for name in common::FIXTURES {
        let Some(tok) = common::fixture(name) else {
            continue;
        };
        let ids: Vec<Vec<u32>> = lines
            .iter()
            .map(|l| tok.encode(l.as_str(), true).unwrap().ids().to_vec())
            .collect();
        let n_tokens: usize = ids.iter().map(Vec::len).sum();
        group.throughput(Throughput::Elements(n_tokens as u64));
        group.bench_with_input(BenchmarkId::new("sequential", name), &ids, |b, ids| {
            b.iter(|| {
                for seq in ids {
                    black_box(tok.decode(seq, true).unwrap());
                }
            })
        });
        let slices: Vec<&[u32]> = ids.iter().map(Vec::as_slice).collect();
        group.bench_with_input(BenchmarkId::new("batch", name), &slices, |b, slices| {
            b.iter(|| black_box(tok.decode_batch(slices, true).unwrap()))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
