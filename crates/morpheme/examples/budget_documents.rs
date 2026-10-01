//! Prepare JSONL documents for a token budget without silent truncation.
//! See docs/document-workflow.md for local, pinned-Hub and offline use.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

use morpheme::Tokenizer;
use serde::Deserialize;
use serde_json::json;
use sha2::Digest;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    id: serde_json::Value,
    text: String,
}

fn load(source: &str, revision: Option<&str>) -> Result<(Tokenizer, Option<String>)> {
    if Path::new(source).is_file() {
        if revision.is_some() {
            return Err("revision applies only to Hub sources".into());
        }
        let bytes = std::fs::read(source)?;
        let hash: String = sha2::Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        return Ok((
            Tokenizer::from_json(std::str::from_utf8(&bytes)?)?,
            Some(hash),
        ));
    }
    #[cfg(feature = "hub")]
    {
        let revision = revision
            .filter(|r| r.len() == 40 && r.bytes().all(|b| b.is_ascii_hexdigit()))
            .ok_or("Hub sources require a full pinned commit hash")?;
        let params = morpheme::FromPretrainedParameters::default().revision(revision);
        Ok((Tokenizer::from_pretrained(source, Some(params))?, None))
    }
    #[cfg(not(feature = "hub"))]
    Err("Hub sources require building this example with --features hub".into())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(4..=5).contains(&args.len()) {
        return Err("usage: budget_documents <tokenizer.json|Hub-id> <budget> <input.jsonl> <output.jsonl> [commit]".into());
    }
    let source = &args[0];
    let budget = args[1].parse::<usize>()?;
    if budget == 0 {
        return Err("budget must be positive".into());
    }
    let input = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    if output.exists()
        && (std::fs::canonicalize(output)? == std::fs::canonicalize(input)?
            || (Path::new(source).is_file()
                && std::fs::canonicalize(output)? == std::fs::canonicalize(source)?))
    {
        return Err("output must not overwrite the input dataset or tokenizer".into());
    }
    let revision = args.get(4).map(String::as_str);
    let (mut tokenizer, hash) = load(source, revision)?;
    tokenizer.set_padding(None);
    tokenizer.set_truncation(None)?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut prepared = tempfile::NamedTempFile::new_in(parent)?;
    let mut reader = BufReader::new(std::fs::File::open(input)?);
    let mut records = 0;
    loop {
        let mut bytes = Vec::new();
        let size = Read::by_ref(&mut reader)
            .take(8 * 1024 * 1024 + 1)
            .read_until(b'\n', &mut bytes)?;
        if size == 0 {
            break;
        }
        records += 1;
        if size > 8 * 1024 * 1024 {
            return Err(format!("record {records} exceeds 8 MiB").into());
        }
        let document: Document =
            serde_json::from_slice(&bytes).map_err(|e| format!("record {records}: {e}"))?;
        let encoding = tokenizer
            .encode(document.text.as_str(), true)
            .map_err(|e| format!("record {records}: {e}"))?;
        if encoding.len() > budget {
            return Err(format!(
                "record {records} ({:?}): {} tokens exceeds budget {budget}",
                document.id,
                encoding.len()
            )
            .into());
        }
        serde_json::to_writer(
            &mut prepared,
            &json!({
                "id": document.id, "text": document.text, "token_count": encoding.len(),
                "ids": encoding.ids(), "offsets": encoding.offsets(),
                "tokenizer": {"source":source, "revision":revision, "sha256":hash},
            }),
        )?;
        writeln!(prepared)?;
    }
    prepared.as_file().sync_all()?;
    prepared.persist(output).map_err(|e| e.error)?;
    eprintln!("Prepared {records} documents within a {budget}-token budget");
    Ok(())
}
