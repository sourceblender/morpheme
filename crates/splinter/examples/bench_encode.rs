//! Rough encode throughput: `cargo run --release --example bench_encode --
//! <tokenizer.json> <text file>`. Encodes every line one at a time, then
//! as one parallel batch. See docs/benchmarks.md.

use std::time::Instant;

use splinter::Tokenizer;

fn main() -> splinter::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [tokenizer, text] = args.as_slice() else {
        eprintln!("usage: bench_encode <tokenizer.json> <text file>");
        std::process::exit(2);
    };
    let tok = Tokenizer::from_file(tokenizer)?;
    let text = std::fs::read_to_string(text)?;
    let lines: Vec<&str> = text.lines().collect();
    let mb = text.len() as f64 / 1e6;

    let start = Instant::now();
    let mut n_tokens = 0;
    for line in &lines {
        n_tokens += tok.encode(*line, true)?.len();
    }
    let single = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let batch = tok.encode_batch(lines.clone(), true)?;
    let batched = start.elapsed().as_secs_f64();
    assert_eq!(batch.iter().map(|e| e.len()).sum::<usize>(), n_tokens);

    println!(
        "{} lines, {:.1} MB, {} tokens | sequential {:.2}s ({:.1} MB/s) | batch {:.2}s ({:.1} MB/s)",
        lines.len(),
        mb,
        n_tokens,
        single,
        mb / single,
        batched,
        mb / batched
    );
    Ok(())
}
