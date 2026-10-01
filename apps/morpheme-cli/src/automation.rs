//! Bounded JSONL automation and explicit token-budget counting.

use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Args;
use morpheme::{EncodeInput, Encoding};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};

use super::{Source, load, type_name};

fn positive(value: &str) -> std::result::Result<usize, String> {
    match value.parse::<usize>() {
        Ok(n) if n > 0 && n < usize::MAX => Ok(n),
        _ => Err("expected a positive integer smaller than usize::MAX".into()),
    }
}

#[derive(Args)]
pub struct BatchOptions {
    #[command(flatten)]
    source: Source,
    /// JSONL file; `-` reads standard input.
    #[arg(long, default_value = "-")]
    input: PathBuf,
    /// Maximum records processed together; output retains input order.
    #[arg(long, default_value = "64", value_parser = positive)]
    batch_size: usize,
    /// Maximum bytes in one input record, including its line ending.
    #[arg(long, default_value = "8388608", value_parser = positive)]
    max_record_bytes: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EncodeRecord {
    text: String,
    #[serde(default)]
    pair: Option<String>,
    #[serde(default, deserialize_with = "provided_id")]
    id: Option<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DecodeRecord {
    ids: Vec<u32>,
    #[serde(default, deserialize_with = "provided_id")]
    id: Option<Value>,
}

fn provided_id<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<Value>, D::Error> {
    Value::deserialize(deserializer).map(Some)
}

fn batches<T: DeserializeOwned>(
    options: &BatchOptions,
    mut process: impl FnMut(Vec<(usize, T)>) -> Result<()>,
) -> Result<()> {
    let mut reader: Box<dyn BufRead> = if options.input.as_os_str() == "-" {
        Box::new(BufReader::new(std::io::stdin()))
    } else {
        Box::new(BufReader::new(
            std::fs::File::open(&options.input)
                .with_context(|| format!("open {}", options.input.display()))?,
        ))
    };
    let mut line = 0;
    loop {
        let mut batch = Vec::new();
        while batch.len() < options.batch_size {
            let mut bytes = Vec::new();
            let n = Read::by_ref(&mut reader)
                .take(options.max_record_bytes as u64 + 1)
                .read_until(b'\n', &mut bytes)
                .with_context(|| format!("read record {}", line + 1))?;
            if n == 0 {
                break;
            }
            line += 1;
            if n > options.max_record_bytes {
                bail!(
                    "record {line} exceeds --max-record-bytes {}",
                    options.max_record_bytes
                );
            }
            let record = serde_json::from_slice(&bytes)
                .with_context(|| format!("invalid JSONL record {line}"))?;
            batch.push((line, record));
        }
        if batch.is_empty() {
            return Ok(());
        }
        process(batch)?;
    }
}

fn emit(writer: &mut impl Write, id: Option<&Value>, mut value: Value) -> Result<()> {
    if let Some(id) = id {
        value["id"] = id.clone();
    }
    serde_json::to_writer(&mut *writer, &value)?;
    writeln!(writer)?;
    Ok(())
}

fn encoding_json(encoding: &Encoding) -> Value {
    json!({
        "ids": encoding.ids(), "tokens": encoding.tokens(),
        "offsets": encoding.offsets(), "type_ids": encoding.type_ids(),
        "attention_mask": encoding.attention_mask(),
        "special_tokens_mask": encoding.special_tokens_mask(),
        "word_ids": encoding.word_ids(), "sequence_ids": encoding.sequence_ids(),
        "overflowing": encoding.overflowing().iter().map(encoding_json).collect::<Vec<_>>()
    })
}

pub fn encode_batch(
    options: &BatchOptions,
    specials: bool,
    chars: bool,
    ignore_settings: bool,
) -> Result<()> {
    let mut tokenizer = load(&options.source)?;
    if ignore_settings {
        tokenizer.set_padding(None);
        tokenizer.set_truncation(None)?;
    }
    let mut output = BufWriter::new(std::io::stdout().lock());
    batches::<EncodeRecord>(options, |batch| {
        let inputs: Vec<EncodeInput<'_>> = batch
            .iter()
            .map(|(_, record)| match &record.pair {
                Some(pair) => (record.text.as_str(), pair.as_str()).into(),
                None => record.text.as_str().into(),
            })
            .collect();
        let encode_all = |inputs: Vec<EncodeInput<'_>>| {
            if chars {
                tokenizer.encode_batch_char_offsets(inputs, specials)
            } else {
                tokenizer.encode_batch(inputs, specials)
            }
        };
        let encoded = match encode_all(inputs.clone()) {
            Ok(encoded) => encoded,
            // A parallel batch only reports that *something* failed; redo
            // it one record at a time to name the exact record.
            Err(batch_err) => {
                for ((line, _), input) in batch.iter().zip(inputs) {
                    encode_all(vec![input]).with_context(|| format!("record {line}"))?;
                }
                let first = batch.first().expect("nonempty batch").0;
                return Err(batch_err)
                    .with_context(|| format!("encode batch starting at record {first}"));
            }
        };
        for ((_, record), encoding) in batch.iter().zip(&encoded) {
            emit(
                &mut output,
                record.id.as_ref(),
                json!({"encoding": encoding_json(encoding)}),
            )?;
        }
        output.flush()?;
        Ok(())
    })
}

pub fn decode_batch(options: &BatchOptions, skip_specials: bool) -> Result<()> {
    let tokenizer = load(&options.source)?;
    let mut output = BufWriter::new(std::io::stdout().lock());
    batches::<DecodeRecord>(options, |batch| {
        for (line, record) in &batch {
            if let Some(id) = record
                .ids
                .iter()
                .find(|id| tokenizer.id_to_token(**id).is_none())
            {
                bail!("record {line}: id {id} is not in the vocabulary");
            }
        }
        let ids: Vec<&[u32]> = batch
            .iter()
            .map(|(_, record)| record.ids.as_slice())
            .collect();
        let decoded = tokenizer
            .decode_batch(&ids, skip_specials)
            .context("decode batch failed")?;
        for ((_, record), text) in batch.iter().zip(decoded) {
            emit(&mut output, record.id.as_ref(), json!({"text": text}))?;
        }
        output.flush()?;
        Ok(())
    })
}

pub fn count(
    source: &Source,
    text: String,
    pair: Option<String>,
    specials: bool,
    use_settings: bool,
    as_json: bool,
) -> Result<()> {
    let mut tokenizer = load(source)?;
    if !use_settings {
        tokenizer.set_padding(None);
        tokenizer.set_truncation(None)?;
    }
    let text = if text == "-" {
        let mut text = String::new();
        std::io::stdin().read_to_string(&mut text)?;
        text
    } else {
        text
    };
    let input: EncodeInput<'_> = match &pair {
        Some(pair) => (text.as_str(), pair.as_str()).into(),
        None => text.as_str().into(),
    };
    let count = tokenizer.encode(input, specials)?.len();
    let mut out = std::io::stdout().lock();
    if as_json {
        writeln!(
            out,
            "{}",
            json!({"count": count, "add_special_tokens": specials, "use_tokenizer_settings": use_settings})
        )?;
    } else {
        writeln!(out, "{count}")?;
    }
    Ok(())
}

pub fn inspect(source: &Source) -> Result<()> {
    let tokenizer = load(source)?;
    let specials: Vec<_> = tokenizer
        .added_vocabulary()
        .tokens_with_ids()
        .into_iter()
        .filter(|t| t.token.special)
        .collect();
    writeln!(
        std::io::stdout().lock(),
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1, "model": type_name(Some(tokenizer.model())),
            "vocab_size": tokenizer.vocab_size(true),
            "model_vocab_size": tokenizer.vocab_size(false),
            "added_tokens_count": tokenizer.added_vocabulary().len(),
            "special_tokens": specials,
            "components": {
                "normalizer": type_name(tokenizer.normalizer()),
                "pre_tokenizer": type_name(tokenizer.pre_tokenizer()),
                "post_processor": type_name(tokenizer.post_processor()),
                "decoder": type_name(tokenizer.decoder()),
            },
            "padding": tokenizer.padding(), "truncation": tokenizer.truncation(),
        }))?
    )?;
    Ok(())
}
