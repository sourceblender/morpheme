//! Rough end-to-end throughput on any tokenizer and text file:
//!
//! ```sh
//! cargo run --release --example bench_encode -- <tokenizer.json> <text file>
//! ```
//!
//! Encodes every line one at a time, then as one parallel batch, then
//! decodes the batch (sequentially and in parallel). For statistically
//! sound numbers use the Criterion suite (`cargo bench`). See
//! docs/benchmarks.md.

use std::time::Instant;

use splinter::Tokenizer;

fn main() -> splinter::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [tokenizer, text] = args.as_slice() else {
        eprintln!("usage: bench_encode <tokenizer.json> <text file>");
        std::process::exit(2);
    };
    let mut tok = Tokenizer::from_file(tokenizer)?;
    tok.set_padding(None);
    tok.set_truncation(None)?;
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

    let ids: Vec<&[u32]> = batch.iter().map(|e| e.ids()).collect();
    let start = Instant::now();
    for seq in &ids {
        tok.decode(seq, true)?;
    }
    let dec_single = start.elapsed().as_secs_f64();
    let start = Instant::now();
    tok.decode_batch(&ids, true)?;
    let dec_batched = start.elapsed().as_secs_f64();
    let mtok = n_tokens as f64 / 1e6;

    println!(
        "{} lines, {:.1} MB, {} tokens\n\
         encode: sequential {:.2}s ({:.1} MB/s) | batch {:.2}s ({:.1} MB/s)\n\
         decode: sequential {:.2}s ({:.1} M tokens/s) | batch {:.2}s ({:.1} M tokens/s)",
        lines.len(),
        mb,
        n_tokens,
        single,
        mb / single,
        batched,
        mb / batched,
        dec_single,
        mtok / dec_single,
        dec_batched,
        mtok / dec_batched
    );
    Ok(())
}
