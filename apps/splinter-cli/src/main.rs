use anyhow::{Context, Result};
use splinter::{
    BertNormalizer, BertPreTokenizer, Bpe, BpeTrainer, ByteLevel, ByteLevelDecoder, Lowercase,
    Tokenizer, Trainer, UnigramTrainer, WordPieceTrainer,
};

// `include_str!` resolves relative to the source file
// (`apps/splinter-cli/src/main.rs`). Three `..` segments walk up
// to the workspace root, then down to the fixture.
const BUNDLED_FIXTURE: &str = include_str!("../../../crates/splinter/tests/fixtures/tiny.json");

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print_help();
        std::process::exit(2);
    }
    let cmd = args.remove(0);

    match cmd.as_str() {
        "encode" => {
            let path = take_from(&mut args, "--from");
            let normalizer = take_choice(
                &mut args,
                "--normalize",
                &["none", "lowercase", "bert"],
                "none",
            )?;
            let pre = take_choice(
                &mut args,
                "--pre-tokenize",
                &["whitespace", "bert", "byte-level"],
                "whitespace",
            )?;
            let text = args.into_iter().next().context(
                "usage: splinter encode [--from <path>] [--normalize X] [--pre-tokenize Y] <text>",
            )?;
            let mut t = load(&path)?;
            apply_normalizer(&mut t, &normalizer);
            apply_pre_tokenizer(&mut t, &pre);
            let enc = t
                .encode(&text)
                .with_context(|| format!("encode failed for {text:?}"))?;
            println!("tokens:  {:?}", enc.tokens);
            println!("ids:     {:?}", enc.ids);
            println!("offsets: {:?}", enc.offsets);
        }
        "decode" => {
            let path = take_from(&mut args, "--from");
            let pre = take_choice(
                &mut args,
                "--pre-tokenize",
                &["whitespace", "bert", "byte-level"],
                "whitespace",
            )?;
            let ids: Result<Vec<u32>> = args
                .into_iter()
                .next()
                .context(
                    "usage: splinter decode [--from <path>] [--pre-tokenize Y] <id,id,id,...>",
                )?
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<u32>()
                        .with_context(|| format!("not an integer: {s:?}"))
                })
                .collect();
            let ids = ids?;
            let mut t = load(&path)?;
            apply_pre_tokenizer(&mut t, &pre);
            // ByteLevel pairs with ByteLevelDecoder for the inverse.
            if matches!(pre.as_str(), "byte-level") {
                t = t.with_decoder(Box::new(ByteLevelDecoder));
            }
            let text = t.decode(&ids, false).context("decode failed")?;
            println!("{text}");
        }
        "inspect" => {
            let path = take_from(&mut args, "--from");
            let t = load(&path)?;
            println!("vocab size:    {}", t.vocab().len());
            if let Some(bpe) = t.bpe_model() {
                println!("model:         bpe");
                println!("merges:        {}", bpe.num_merges());
                println!("eow suffix:    {:?}", bpe.end_of_word_suffix());
            } else if let Some(wp) = t.wordpiece_model() {
                println!("model:         wordpiece");
                println!("unk token:     {:?}", wp.unk_token);
                println!("prefix:        {:?}", wp.continuing_subword_prefix);
                println!("max chars:     {}", wp.max_input_chars_per_word);
            } else if let Some(ug) = t.unigram_model() {
                println!("model:         unigram");
                println!("unk token:     {:?}", ug.unk_token);
                println!("ws marker:     {:?}", ug.whitespace_marker);
                println!("min score:     {}", ug.min_score);
            }
        }
        "version" | "--version" | "-V" => {
            println!("splinter {}", splinter::VERSION);
        }
        "train" => {
            // Flags can appear in any order. Parse all of them first.
            let input =
                take_flag(&mut args, "--input").context("missing required --input <corpus.txt>")?;
            let vocab_size: usize = take_flag(&mut args, "--vocab-size")
                .context("missing required --vocab-size N")?
                .parse()
                .context("--vocab-size must be a positive integer")?;
            let out =
                take_flag(&mut args, "--out").context("missing required --out <tokenizer.json>")?;
            let algorithm =
                take_flag(&mut args, "--algorithm").unwrap_or_else(|_| "bpe".to_string());
            if !["bpe", "wordpiece", "unigram"].contains(&algorithm.as_str()) {
                anyhow::bail!(
                    "--algorithm must be one of [\"bpe\", \"wordpiece\", \"unigram\"], got {algorithm:?}"
                );
            }
            let min_freq: u64 = take_flag(&mut args, "--min-pair-frequency")
                .ok()
                .map(|s| s.parse().unwrap_or(0))
                .unwrap_or(0);

            match algorithm.as_str() {
                "bpe" => {
                    let trainer = BpeTrainer::builder(vocab_size)
                        .min_pair_frequency(min_freq)
                        .build();
                    tracing::info!(vocab_size, min_freq, "training BPE");
                    let trained = trainer
                        .train(std::path::PathBuf::from(&input))
                        .with_context(|| format!("training failed on {input}"))?;
                    let bpe = Bpe::new(trained.vocab, trained.merges, trained.end_of_word_suffix);
                    let tok = Tokenizer::new(bpe);
                    tok.to_file(&out, true)
                        .with_context(|| format!("failed to write {out}"))?;
                    tracing::info!(path = %out, "wrote tokenizer");
                    println!("wrote {out}");
                    println!(
                        "vocab: {} entries, {} merges",
                        tok.vocab().len(),
                        tok.bpe_model().map(|m| m.num_merges()).unwrap_or(0)
                    );
                }
                "wordpiece" => {
                    let trainer = WordPieceTrainer::builder(vocab_size)
                        .alphabet(vec!["<unk>".to_string()])
                        .min_frequency(min_freq)
                        .build();
                    tracing::info!(vocab_size, min_freq, "training WordPiece");
                    let trained = trainer
                        .train(std::path::PathBuf::from(&input))
                        .with_context(|| format!("training failed on {input}"))?;
                    let wp = splinter::WordPiece::new(
                        trained.vocab,
                        trained.continuing_subword_prefix.clone(),
                        "<unk>".to_string(),
                        100,
                    );
                    let tok = Tokenizer::wordpiece(wp);
                    tok.to_file(&out, true)
                        .with_context(|| format!("failed to write {out}"))?;
                    tracing::info!(path = %out, "wrote tokenizer");
                    println!("wrote {out}");
                    println!("vocab: {} entries", tok.vocab().len());
                }
                "unigram" => {
                    let trainer = UnigramTrainer::builder(vocab_size)
                        .alphabet(vec!["<unk>".to_string()])
                        .min_frequency(min_freq.max(1))
                        .build();
                    tracing::info!(vocab_size, min_freq, "training Unigram");
                    let trained = trainer
                        .train(std::path::PathBuf::from(&input))
                        .with_context(|| format!("training failed on {input}"))?;
                    let ug = splinter::Unigram::new(
                        trained.vocab,
                        trained.log_probs,
                        trained.whitespace_marker.clone(),
                        "<unk>".to_string(),
                        trained.min_score,
                    );
                    let marker_first = trained
                        .whitespace_marker
                        .chars()
                        .next()
                        .unwrap_or('\u{2581}');
                    let tok = Tokenizer::builder(ug)
                        .pre_tokenizer(Box::new(splinter::MetaspacePreTokenizer::new(
                            marker_first,
                            true,
                        )))
                        .build();
                    tok.to_file(&out, true)
                        .with_context(|| format!("failed to write {out}"))?;
                    tracing::info!(path = %out, "wrote tokenizer");
                    println!("wrote {out}");
                    println!("vocab: {} entries", tok.vocab().len());
                }
                _ => unreachable!("take_choice validated the algorithm"),
            }
        }
        "--help" | "-h" | "help" => print_help(),
        other => {
            eprintln!("unknown subcommand: {other}");
            print_help();
            std::process::exit(2);
        }
    }

    Ok(())
}

/// Pull `--flag <value>` off the front of `args`. Returns the value if
/// present and the flag is one of `allowed`. Otherwise returns
/// `default_value` and leaves `args` untouched.
fn take_choice(
    args: &mut Vec<String>,
    flag: &str,
    allowed: &[&str],
    default: &str,
) -> Result<String> {
    if args.first().map(String::as_str) != Some(flag) {
        return Ok(default.to_string());
    }
    if args.len() < 2 {
        anyhow::bail!("{flag} requires a value");
    }
    args.remove(0);
    let v = args.remove(0);
    if !allowed.iter().any(|a| *a == v) {
        anyhow::bail!("{flag} must be one of {allowed:?}, got {v:?}");
    }
    Ok(v)
}

/// Pull `--flag <value>` off the front of `args`. Returns
/// `Err(...)` if the flag is absent. Kept for `take_from`-style
/// uses where order matters; prefer [`take_flag`] for
/// order-independent parsing.
#[allow(dead_code)]
fn take_value(args: &mut Vec<String>, flag: &str) -> Result<String> {
    if args.first().map(String::as_str) != Some(flag) {
        anyhow::bail!("missing required {flag}");
    }
    if args.len() < 2 {
        anyhow::bail!("{flag} requires a value");
    }
    args.remove(0);
    Ok(args.remove(0))
}

/// Search `args` for `--flag <value>` and consume both if found.
/// Returns an error if `--flag` is present without a value, or if
/// the flag is absent.
fn take_flag(args: &mut Vec<String>, flag: &str) -> Result<String> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag {
            if i + 1 >= args.len() {
                anyhow::bail!("{flag} requires a value");
            }
            let v = args.remove(i + 1);
            args.remove(i);
            return Ok(v);
        }
        i += 1;
    }
    anyhow::bail!("missing required {flag}")
}

fn take_from(args: &mut Vec<String>, flag: &str) -> String {
    if args.first().map(String::as_str) == Some(flag) {
        if args.len() < 2 {
            eprintln!("error: {flag} requires a value");
            std::process::exit(2);
        }
        args.remove(0);
        args.remove(0)
    } else {
        "-".into()
    }
}

fn load(arg: &str) -> Result<Tokenizer> {
    if arg == "-" {
        Tokenizer::from_json(BUNDLED_FIXTURE).context("bundled fixture failed to parse")
    } else {
        // Auto-detect: HF tokenizer.json files have `"model": {...}`
        // at the top level with a `"vocab"` object. splinter's own
        // format nests under `"model"` and has `"vocab"` as an
        // array. We sniff the raw bytes to decide.
        let raw = std::fs::read_to_string(arg).with_context(|| format!("failed to read {arg}"))?;
        if is_hf_format(&raw) {
            splinter::tokenizer::hf::from_str(&raw)
                .with_context(|| format!("failed to parse HF format {arg}"))
        } else {
            Tokenizer::from_json(&raw)
                .with_context(|| format!("failed to parse splinter format {arg}"))
        }
    }
}

/// Heuristic: HF files always include `"added_tokens"` at the top
/// level, which splinter's format does not have.
fn is_hf_format(raw: &str) -> bool {
    raw.contains("\"added_tokens\"") || raw.contains("\"pre_tokenizer\"")
}

fn apply_normalizer(t: &mut Tokenizer, choice: &str) {
    let new: Box<dyn splinter::Normalizer> = match choice {
        "none" => Box::new(splinter::IdentityNormalizer),
        "lowercase" => Box::new(Lowercase),
        "bert" => Box::new(BertNormalizer::default()),
        _ => return,
    };
    *t = rebuild_with(t, |b| b.normalizer(new));
}

fn apply_pre_tokenizer(t: &mut Tokenizer, choice: &str) {
    let new: Box<dyn splinter::PreTokenizer> = match choice {
        "whitespace" => Box::new(splinter::Whitespace),
        "bert" => Box::new(BertPreTokenizer),
        "byte-level" => Box::new(ByteLevel::default()),
        _ => return,
    };
    *t = rebuild_with(t, |b| b.pre_tokenizer(new));
}

/// Rebuild a `Tokenizer`, preserving its current model, applying
/// the supplied builder tweak.
///
/// v0.1 limitation: this drops the post-processor. If you've
/// configured one via `with_post_processor` on a loaded tokenizer,
/// re-apply it in code after calling `apply_normalizer` or
/// `apply_pre_tokenizer` here.
fn rebuild_with(
    t: &Tokenizer,
    tweak: impl FnOnce(splinter::TokenizerBuilder) -> splinter::TokenizerBuilder,
) -> Tokenizer {
    if t.has_post_processor() {
        eprintln!(
            "warning: --normalize / --pre-tokenize rebuilds the \
             tokenizer, dropping any post-processor. Re-apply it via \
             `Tokenizer::with_post_processor` in code."
        );
    }
    if let Some(bpe) = t.bpe_model() {
        tweak(Tokenizer::builder(bpe.clone())).build()
    } else if let Some(wp) = t.wordpiece_model() {
        tweak(Tokenizer::builder(wp.clone())).build()
    } else if let Some(ug) = t.unigram_model() {
        tweak(Tokenizer::builder(ug.clone())).build()
    } else {
        unreachable!("Tokenizer always has a model")
    }
}

fn print_help() {
    println!("splinter {}", splinter::VERSION);
    println!();
    println!("USAGE:");
    println!("    splinter <command> [args]");
    println!();
    println!("COMMANDS:");
    println!("    encode [--from <path>] [--normalize X] [--pre-tokenize Y] <text>");
    println!("        --normalize: none | lowercase | bert   (default: none)");
    println!("        --pre-tokenize: whitespace | bert | byte-level   (default: whitespace)");
    println!("    decode [--from <path>] [--pre-tokenize Y] <id,id,...>");
    println!("        Use --pre-tokenize byte-level with ByteLevelDecoder automatically.");
    println!("    inspect [--from <path>]");
    println!("        Print summary statistics for the tokenizer.");
    println!("    train --input <corpus.txt> --vocab-size N --out <tokenizer.json>");
    println!("        [--algorithm bpe|wordpiece|unigram] [--min-pair-frequency M]");
    println!("        Train a tokenizer from a text file and write it to disk.");
    println!("    version             Print the library version");
    println!("    help                Print this message");
}
