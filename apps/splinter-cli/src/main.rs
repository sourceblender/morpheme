//! `splinter` command-line interface.
//!
//! A thin wrapper over the library: every command works on a
//! `tokenizer.json` file (Hugging Face format) — a local path or, with the
//! `hub` feature (on by default), a Hugging Face Hub model id — and uses
//! the full pipeline stored in it; nothing is replaced or dropped.

use std::collections::HashSet;
use std::io::{IsTerminal, Read};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use splinter::decoders::{self, DecoderWrapper};
use splinter::models::{Bpe, ModelWrapper, Unigram, WordLevel, WordPiece};
use splinter::normalizers::{BertNormalizer, Nfkc};
use splinter::pre_tokenizers::{BertPreTokenizer, ByteLevel, Metaspace, PrependScheme, Whitespace};
use splinter::processors::TemplateProcessing;
use splinter::trainers::{
    BpeTrainer, TrainerWrapper, UnigramTrainer, WordLevelTrainer, WordPieceTrainer,
};
use splinter::{AddedToken, Model, Tokenizer};

#[derive(Parser)]
#[command(
    name = "splinter",
    version,
    about = "Fast, Hugging Face-compatible tokenization"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Encode text into tokens and ids.
    Encode {
        #[command(flatten)]
        source: Source,
        /// Second sequence, for pair inputs.
        #[arg(long)]
        pair: Option<String>,
        /// Do not add special tokens (e.g. [CLS]/[SEP]).
        #[arg(long)]
        no_special_tokens: bool,
        /// Report char offsets instead of byte offsets.
        #[arg(long)]
        char_offsets: bool,
        /// Print the encoding as JSON.
        #[arg(long)]
        json: bool,
        /// Text to encode; `-` reads standard input.
        text: String,
    },
    /// Decode ids back into text.
    Decode {
        #[command(flatten)]
        source: Source,
        /// Drop special tokens from the output.
        #[arg(long)]
        skip_special_tokens: bool,
        /// Ids, separated by spaces and/or commas.
        #[arg(required = true, num_args = 1..)]
        ids: Vec<String>,
    },
    /// Describe a tokenizer.json file.
    Inspect {
        #[command(flatten)]
        source: Source,
    },
    /// Train a new tokenizer from text files.
    Train {
        /// Model to train.
        #[arg(short, long, value_enum, default_value_t = ModelKind::Bpe)]
        model: ModelKind,
        /// Component preset. Defaults: bpe → byte-level, wordpiece →
        /// bert, unigram → sentencepiece, wordlevel → whitespace.
        #[arg(short, long, value_enum)]
        preset: Option<Preset>,
        /// Target vocabulary size.
        #[arg(long, default_value_t = 30_000)]
        vocab_size: usize,
        /// Minimum pair/word frequency (BPE, WordPiece, WordLevel).
        #[arg(long, default_value_t = 0)]
        min_frequency: u64,
        /// Special tokens, in id order. Replaces the preset's defaults.
        #[arg(long = "special-token")]
        special_tokens: Vec<String>,
        /// Where to write tokenizer.json.
        #[arg(short, long)]
        out: PathBuf,
        /// Training text files (one or more).
        #[arg(required = true, num_args = 1..)]
        files: Vec<PathBuf>,
        /// Don't show progress bars (they are only shown when stderr is a
        /// terminal).
        #[arg(short, long)]
        quiet: bool,
    },
}

/// Where to load a tokenizer from.
#[derive(Args)]
struct Source {
    /// A tokenizer.json path, or a Hugging Face Hub model id such as
    /// `google-bert/bert-base-uncased` (downloaded and cached).
    #[arg(short, long, visible_alias = "from")]
    tokenizer: String,
    /// Hub revision (branch, tag or commit hash) for a model id.
    #[arg(long)]
    revision: Option<String>,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum ModelKind {
    Bpe,
    Wordpiece,
    Unigram,
    Wordlevel,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Preset {
    /// GPT-2 style: byte-level pre-tokenization, no unknown tokens.
    ByteLevel,
    /// BERT style: lowercase, split on whitespace/punctuation, [CLS]/[SEP].
    Bert,
    /// SentencePiece style: NFKC, `▁` word marker.
    Sentencepiece,
    /// Split on whitespace and punctuation only.
    Whitespace,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Encode {
            source,
            pair,
            no_special_tokens,
            char_offsets,
            json,
            text,
        } => encode(&source, text, pair, !no_special_tokens, char_offsets, json),
        Command::Decode {
            source,
            skip_special_tokens,
            ids,
        } => decode(&source, &ids, skip_special_tokens),
        Command::Inspect { source } => inspect(&source),
        Command::Train {
            model,
            preset,
            vocab_size,
            min_frequency,
            special_tokens,
            out,
            files,
            quiet,
        } => train(
            model,
            preset,
            vocab_size,
            min_frequency,
            special_tokens,
            &out,
            &files,
            !quiet && std::io::stderr().is_terminal(),
        ),
    }
}

fn load(source: &Source) -> Result<Tokenizer> {
    let path = std::path::Path::new(&source.tokenizer);
    if path.exists() {
        if source.revision.is_some() {
            bail!("--revision only applies to Hugging Face Hub model ids, not files");
        }
        return Tokenizer::from_file(path)
            .with_context(|| format!("failed to load {}", path.display()));
    }
    load_from_hub(source)
}

#[cfg(feature = "hub")]
fn load_from_hub(source: &Source) -> Result<Tokenizer> {
    let id = &source.tokenizer;
    let looks_like_id = !id.ends_with(".json") && id.matches('/').count() <= 1;
    if !looks_like_id {
        bail!("failed to load {id}: no such file");
    }
    let mut params = splinter::FromPretrainedParameters::default();
    if let Some(revision) = &source.revision {
        params = params.revision(revision.clone());
    }
    Tokenizer::from_pretrained(id, Some(params))
        .with_context(|| format!("failed to load {id} (not a file; tried the Hugging Face Hub)"))
}

#[cfg(not(feature = "hub"))]
fn load_from_hub(source: &Source) -> Result<Tokenizer> {
    bail!(
        "failed to load {}: no such file (Hub downloads need the `hub` feature)",
        source.tokenizer
    )
}

fn encode(
    source: &Source,
    text: String,
    pair: Option<String>,
    add_special_tokens: bool,
    char_offsets: bool,
    json: bool,
) -> Result<()> {
    let tokenizer = load(source)?;
    let text = if text == "-" {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf
    } else {
        text
    };
    let encoding = match (&pair, char_offsets) {
        (Some(p), false) => tokenizer.encode((text.as_str(), p.as_str()), add_special_tokens),
        (Some(p), true) => {
            tokenizer.encode_char_offsets((text.as_str(), p.as_str()), add_special_tokens)
        }
        (None, false) => tokenizer.encode(text.as_str(), add_special_tokens),
        (None, true) => tokenizer.encode_char_offsets(text.as_str(), add_special_tokens),
    }
    .context("encode failed")?;

    if json {
        let value = serde_json::json!({
            "ids": encoding.ids(),
            "tokens": encoding.tokens(),
            "offsets": encoding.offsets(),
            "type_ids": encoding.type_ids(),
            "attention_mask": encoding.attention_mask(),
            "special_tokens_mask": encoding.special_tokens_mask(),
            "word_ids": encoding.word_ids(),
        });
        println!("{}", serde_json::to_string_pretty(&value)?);
    } else {
        println!("tokens:  {:?}", encoding.tokens());
        println!("ids:     {:?}", encoding.ids());
        println!("offsets: {:?}", encoding.offsets());
        if pair.is_some() {
            println!("types:   {:?}", encoding.type_ids());
        }
    }
    Ok(())
}

fn parse_ids(raw: &[String]) -> Result<Vec<u32>> {
    raw.iter()
        .flat_map(|s| s.split(','))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<u32>()
                .with_context(|| format!("not an id: {s:?}"))
        })
        .collect()
}

fn decode(source: &Source, raw_ids: &[String], skip_special_tokens: bool) -> Result<()> {
    let tokenizer = load(source)?;
    let ids = parse_ids(raw_ids)?;
    let vocab_size = tokenizer.vocab_size(true);
    if let Some(bad) = ids.iter().find(|id| tokenizer.id_to_token(**id).is_none()) {
        bail!("id {bad} is not in the vocabulary (size {vocab_size})");
    }
    println!("{}", tokenizer.decode(&ids, skip_special_tokens)?);
    Ok(())
}

fn type_name<T: serde::Serialize>(component: Option<&T>) -> String {
    component
        .and_then(|c| serde_json::to_value(c).ok())
        .map(|v| describe(&v))
        .unwrap_or_else(|| "none".to_owned())
}

/// `Type` or `Sequence[A, B]` for a serialized component.
fn describe(v: &serde_json::Value) -> String {
    let ty = v["type"].as_str().unwrap_or("?");
    let children = ["normalizers", "pretokenizers", "decoders", "processors"]
        .iter()
        .find_map(|k| v[*k].as_array());
    match children {
        Some(items) if ty == "Sequence" => {
            let inner: Vec<String> = items.iter().map(describe).collect();
            format!("Sequence[{}]", inner.join(", "))
        }
        _ => ty.to_owned(),
    }
}

fn inspect(source: &Source) -> Result<()> {
    let t = load(source)?;
    let model = match t.model() {
        ModelWrapper::Bpe(m) => format!(
            "BPE ({} merges, unk={:?}, byte_fallback={})",
            m.merges().len(),
            m.unk_token(),
            m.byte_fallback()
        ),
        ModelWrapper::WordPiece(m) => format!(
            "WordPiece (unk={:?}, prefix={:?}, max chars={})",
            m.unk_token(),
            m.continuing_subword_prefix(),
            m.max_input_chars_per_word()
        ),
        ModelWrapper::WordLevel(m) => format!("WordLevel (unk={:?})", m.unk_token()),
        ModelWrapper::Unigram(m) => format!(
            "Unigram (unk_id={:?}, byte_fallback={})",
            m.unk_id(),
            m.byte_fallback()
        ),
        // `ModelWrapper` is non-exhaustive: describe future models by type.
        other => describe(&serde_json::to_value(other)?),
    };
    println!("model:          {model}");
    println!("vocab size:     {} (+{} added)", t.vocab_size(false), {
        t.vocab_size(true) - t.vocab_size(false)
    });
    println!("normalizer:     {}", type_name(t.normalizer()));
    println!("pre-tokenizer:  {}", type_name(t.pre_tokenizer()));
    println!("post-processor: {}", type_name(t.post_processor()));
    println!("decoder:        {}", type_name(t.decoder()));
    if let Some(tr) = t.truncation() {
        println!(
            "truncation:     max_length={} stride={}",
            tr.max_length, tr.stride
        );
    }
    if let Some(p) = t.padding() {
        println!("padding:        {:?} with {:?}", p.strategy, p.pad_token);
    }
    let added = t.added_vocabulary().tokens_with_ids();
    if !added.is_empty() {
        let shown: Vec<String> = added
            .iter()
            .take(12)
            .map(|a| format!("{}={}", a.token.content, a.id))
            .collect();
        let more = if added.len() > 12 {
            format!(" … ({} total)", added.len())
        } else {
            String::new()
        };
        println!("added tokens:   {}{more}", shown.join(" "));
    }
    Ok(())
}

fn default_preset(model: ModelKind) -> Preset {
    match model {
        ModelKind::Bpe => Preset::ByteLevel,
        ModelKind::Wordpiece => Preset::Bert,
        ModelKind::Unigram => Preset::Sentencepiece,
        ModelKind::Wordlevel => Preset::Whitespace,
    }
}

fn default_specials(preset: Preset) -> Vec<String> {
    let v: &[&str] = match preset {
        Preset::ByteLevel => &["<|endoftext|>"],
        Preset::Bert => &["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        Preset::Sentencepiece => &["<unk>", "<s>", "</s>", "<pad>"],
        Preset::Whitespace => &["[UNK]", "[PAD]"],
    };
    v.iter().map(|s| s.to_string()).collect()
}

/// The unknown token for a preset, if its model needs one.
fn unk_token(preset: Preset, specials: &[String]) -> Option<String> {
    let wanted = match preset {
        Preset::ByteLevel => return None,
        Preset::Bert | Preset::Whitespace => "[UNK]",
        Preset::Sentencepiece => "<unk>",
    };
    specials.iter().find(|s| *s == wanted).cloned()
}

#[allow(clippy::too_many_arguments)]
fn train(
    model: ModelKind,
    preset: Option<Preset>,
    vocab_size: usize,
    min_frequency: u64,
    special_tokens: Vec<String>,
    out: &PathBuf,
    files: &[PathBuf],
    show_progress: bool,
) -> Result<()> {
    let preset = preset.unwrap_or_else(|| default_preset(model));
    let specials = if special_tokens.is_empty() {
        default_specials(preset)
    } else {
        special_tokens
    };
    let added: Vec<AddedToken> = specials
        .iter()
        .map(|s| AddedToken::new(s.as_str(), true))
        .collect();
    let unk = unk_token(preset, &specials);

    // Trainers keep the model's own options (unk token, …), so start from
    // a model of the right kind.
    let initial_model: ModelWrapper = match model {
        ModelKind::Bpe => {
            let mut b = Bpe::builder();
            if let Some(unk) = &unk {
                b = b.unk_token(unk.clone());
            }
            b.build()?.into()
        }
        ModelKind::Wordpiece => WordPiece::builder()
            .unk_token(unk.clone().unwrap_or_else(|| "[UNK]".to_owned()))
            .build()?
            .into(),
        ModelKind::Wordlevel => WordLevel::builder()
            .unk_token(unk.clone().unwrap_or_else(|| "[UNK]".to_owned()))
            .build()?
            .into(),
        ModelKind::Unigram => Unigram::default().into(),
    };
    let mut tokenizer = Tokenizer::new(initial_model);
    let mut initial_alphabet: HashSet<char> = HashSet::new();
    match preset {
        Preset::ByteLevel => {
            tokenizer.set_pre_tokenizer(Some(ByteLevel::new(false, true, true).into()));
            tokenizer.set_decoder(Some(ByteLevel::default().into()));
            tokenizer.set_post_processor(Some(ByteLevel::new(false, true, true).into()));
            initial_alphabet = ByteLevel::alphabet();
        }
        Preset::Bert => {
            tokenizer.set_normalizer(Some(BertNormalizer::default().into()))?;
            tokenizer.set_pre_tokenizer(Some(BertPreTokenizer.into()));
            tokenizer.set_decoder(Some(DecoderWrapper::from(decoders::WordPiece::new(
                "##".to_owned(),
                true,
            ))));
        }
        Preset::Sentencepiece => {
            tokenizer.set_normalizer(Some(Nfkc.into()))?;
            let marker = Metaspace::new('▁', PrependScheme::Always, true);
            tokenizer.set_pre_tokenizer(Some(marker.clone().into()));
            tokenizer.set_decoder(Some(marker.into()));
        }
        Preset::Whitespace => {
            tokenizer.set_pre_tokenizer(Some(Whitespace.into()));
        }
    }

    let trainer: TrainerWrapper = match model {
        ModelKind::Bpe => BpeTrainer::builder()
            .vocab_size(vocab_size)
            .min_frequency(min_frequency)
            .special_tokens(added.clone())
            .initial_alphabet(initial_alphabet)
            .show_progress(show_progress)
            .build()?
            .into(),
        ModelKind::Wordpiece => WordPieceTrainer::builder()
            .vocab_size(vocab_size)
            .min_frequency(min_frequency)
            .special_tokens(added.clone())
            .initial_alphabet(initial_alphabet)
            .show_progress(show_progress)
            .build()?
            .into(),
        ModelKind::Wordlevel => WordLevelTrainer::builder()
            .vocab_size(vocab_size)
            .min_frequency(min_frequency)
            .special_tokens(added.clone())
            .show_progress(show_progress)
            .build()?
            .into(),
        ModelKind::Unigram => {
            let mut b = UnigramTrainer::builder()
                .vocab_size(vocab_size)
                .special_tokens(added.clone())
                .show_progress(show_progress);
            if let Some(unk) = &unk {
                b = b.unk_token(unk.clone());
            }
            b.build()?.into()
        }
    };

    tokenizer
        .train_from_files(trainer, files)
        .context("training failed")?;
    set_unk(&mut tokenizer, unk.as_deref())?;

    if preset == Preset::Bert {
        let id = |t: &str| {
            tokenizer
                .token_to_id(t)
                .with_context(|| format!("special token {t} missing after training"))
        };
        let (cls, sep) = (id("[CLS]")?, id("[SEP]")?);
        let processor = TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")?
            .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")?
            .special_tokens(vec![("[CLS]", cls), ("[SEP]", sep)])
            .build()?;
        tokenizer.set_post_processor(Some(processor.into()));
    }

    tokenizer
        .save(out, true)
        .with_context(|| format!("failed to write {}", out.display()))?;
    eprintln!(
        "trained {} tokenizer: {} tokens → {}",
        match model {
            ModelKind::Bpe => "BPE",
            ModelKind::Wordpiece => "WordPiece",
            ModelKind::Unigram => "Unigram",
            ModelKind::Wordlevel => "WordLevel",
        },
        tokenizer.vocab_size(true),
        out.display()
    );
    Ok(())
}

/// WordPiece/WordLevel trainers keep the model's default unk token
/// (`[UNK]`); make sure the trained model actually uses the preset's.
fn set_unk(tokenizer: &mut Tokenizer, unk: Option<&str>) -> Result<()> {
    let Some(unk) = unk else { return Ok(()) };
    if tokenizer.model().token_to_id(unk).is_none() {
        bail!("unknown token {unk:?} is not in the trained vocabulary");
    }
    Ok(())
}
