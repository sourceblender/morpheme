//! One isolated benchmark operation per process; driven by benchmark_baseline.py.
use std::{hint::black_box, time::Instant};

use morpheme::models::{Bpe, Unigram, WordPiece};
use morpheme::normalizers::{BertNormalizer, Nfkc};
use morpheme::pre_tokenizers::{BertPreTokenizer, ByteLevel, Metaspace, PrependScheme};
use morpheme::trainers::{BpeTrainer, UnigramTrainer, WordPieceTrainer};
use morpheme::{AddedToken, Tokenizer};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [operation, source, corpus, batch_size] = args.as_slice() else {
        return Err("usage: benchmark_probe <operation> <model/file> <corpus> <batch-size>".into());
    };
    let batch_size: usize = batch_size.parse()?;
    if batch_size == 0 {
        return Err("batch size must be positive".into());
    }
    if operation == "load" {
        let start = Instant::now();
        let tokenizer = Tokenizer::from_file(source)?;
        let elapsed = start.elapsed().as_secs_f64();
        let vocab = black_box(tokenizer).vocab_size(true);
        println!("{}", json!({"seconds": elapsed, "vocab": vocab}));
        return Ok(());
    }
    let text = std::fs::read_to_string(corpus)?;
    let lines: Vec<&str> = text.lines().collect();
    if operation == "train" {
        let start = Instant::now();
        let vocab = match source.as_str() {
            "bpe" => {
                let mut tok = Tokenizer::new(Bpe::default())
                    .with_pre_tokenizer(ByteLevel::new(false, true, true));
                let trainer = BpeTrainer::builder()
                    .vocab_size(4000)
                    .initial_alphabet(ByteLevel::alphabet())
                    .show_progress(false)
                    .build()?;
                tok.train(trainer, lines.iter().copied())?;
                black_box(tok).vocab_size(true)
            }
            "wordpiece" => {
                let mut tok = Tokenizer::new(WordPiece::default())
                    .with_normalizer(BertNormalizer::default())
                    .with_pre_tokenizer(BertPreTokenizer);
                let trainer = WordPieceTrainer::builder()
                    .vocab_size(4000)
                    .special_tokens(vec![AddedToken::new("[UNK]", true)])
                    .show_progress(false)
                    .build()?;
                tok.train(trainer, lines.iter().copied())?;
                black_box(tok).vocab_size(true)
            }
            "unigram" => {
                let mut tok = Tokenizer::new(Unigram::default())
                    .with_normalizer(Nfkc)
                    .with_pre_tokenizer(Metaspace::new('▁', PrependScheme::Always, true));
                let trainer = UnigramTrainer::builder()
                    .vocab_size(4000)
                    .special_tokens(vec![AddedToken::new("<unk>", true)])
                    .unk_token("<unk>")
                    .show_progress(false)
                    .build()?;
                tok.train(trainer, lines.iter().copied())?;
                black_box(tok).vocab_size(true)
            }
            _ => return Err("unknown trainer".into()),
        };
        println!(
            "{}",
            json!({"seconds": start.elapsed().as_secs_f64(), "bytes": text.len(), "vocab": vocab})
        );
        return Ok(());
    }
    let mut tok = Tokenizer::from_file(source)?;
    tok.set_padding(None);
    tok.set_truncation(None)?;
    let mut tokens = 0usize;
    let mut seconds = 0.0;
    // Bounded batches include output disposal in timing. Decode preparations
    // happen outside the timer but are included in the process peak RSS.
    for chunk in lines.chunks(batch_size) {
        if operation.starts_with("decode") {
            let encodings = tok.encode_batch(chunk.to_vec(), true)?;
            let ids: Vec<&[u32]> = encodings.iter().map(|e| e.ids()).collect();
            tokens += ids.iter().map(|ids| ids.len()).sum::<usize>();
            let start = Instant::now();
            match operation.as_str() {
                "decode_seq" => {
                    for ids in &ids {
                        black_box(tok.decode(ids, true)?);
                    }
                }
                "decode_batch" => {
                    black_box(tok.decode_batch(&ids, true)?);
                }
                _ => return Err("unknown operation".into()),
            }
            seconds += start.elapsed().as_secs_f64();
        } else {
            let start = Instant::now();
            match operation.as_str() {
                "encode_seq" => {
                    for line in chunk {
                        tokens += black_box(tok.encode(*line, true)?).len();
                    }
                }
                "encode_batch" => {
                    tokens += black_box(tok.encode_batch(chunk.to_vec(), true)?)
                        .iter()
                        .map(|e| e.len())
                        .sum::<usize>();
                }
                _ => return Err("unknown operation".into()),
            }
            seconds += start.elapsed().as_secs_f64();
        }
    }
    println!(
        "{}",
        json!({"seconds": seconds, "bytes": text.len(), "tokens": tokens, "lines": lines.len()})
    );
    Ok(())
}
