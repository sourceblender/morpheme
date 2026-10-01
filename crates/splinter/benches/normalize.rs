//! The alignment-tracking `NormalizedString` path: BERT normalization
//! (clean text, CJK padding, NFD + accent stripping, lowercasing) and
//! Unicode NFKC on mixed-script text.
//!
//! `cargo bench --bench normalize`

mod common;

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use splinter::normalizers::{BertNormalizer, Nfkc, NormalizerWrapper};
use splinter::{NormalizedString, Normalizer};

fn bench_normalize(c: &mut Criterion) {
    let text = common::sentences(1_000, 99).join("\n");
    let mut group = c.benchmark_group("normalize");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(5));
    let normalizers: [(&str, NormalizerWrapper); 2] = [
        ("bert", BertNormalizer::default().into()),
        ("nfkc", Nfkc.into()),
    ];
    for (name, normalizer) in &normalizers {
        group.bench_with_input(BenchmarkId::from_parameter(name), &text, |b, text| {
            b.iter(|| {
                let mut n = NormalizedString::from(text.as_str());
                normalizer.normalize(&mut n).unwrap();
                black_box(n)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_normalize);
criterion_main!(benches);
