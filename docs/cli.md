# CLI automation

`morpheme` loads one tokenizer per command. Local files and Hub model ids
are accepted with `-t`; pin a Hub model with `--revision <commit>` and set
`HF_HUB_OFFLINE=1` to require a verified cached copy.

## Count tokens

```sh
morpheme count -t tokenizer.json "Document to budget"
morpheme count -t tokenizer.json --json --no-special-tokens - < document.txt
morpheme count -t tokenizer.json --use-tokenizer-settings "Inference input"
```

By default, `count` disables configured padding and truncation so long
inputs cannot appear to fit a budget merely because they were truncated.
It includes post-processor special tokens unless `--no-special-tokens`
is supplied. `--pair` accepts a second sequence. `--use-tokenizer-settings`
counts the actual padded/truncated main encoding, excluding overflow.
JSON output contains `count`, `add_special_tokens`, and
`use_tokenizer_settings`. Single-text stdin is read as one complete string.

These are tokenizer text counts. For chat requests, apply the model's
chat template first; this CLI does not render chat messages or infer
provider-specific request overhead.

## Stream JSONL batches

```sh
morpheme encode-batch -t tokenizer.json --input documents.jsonl \
  --batch-size 64 --ignore-tokenizer-settings > encoded.jsonl
morpheme decode-batch -t tokenizer.json --input ids.jsonl \
  --skip-special-tokens > decoded.jsonl
```

Encode input is one object per line:

```json
{"id":"doc-1","text":"café"}
{"id":2,"text":"question","pair":"answer"}
```

Each output is `{"id":...,"encoding":{...}}`. `id` is optional and is
echoed unchanged. The encoding includes ids, tokens, offsets, type ids,
masks, word ids, sequence ids, and recursively represented overflow.
Offsets are bytes unless `--char-offsets` is supplied. By default,
encoding respects the tokenizer's padding and truncation settings;
`--ignore-tokenizer-settings` disables both. Specials are included unless
`--no-special-tokens` is supplied. Batch-longest padding is per processed
batch, not across the complete input file.

Decode input is `{"id":...,"ids":[1,2,3]}`; output is
`{"id":...,"text":"..."}`. To decode encode-batch output, project each
record's `encoding.ids` into `ids`; the schemas are intentionally explicit.
Unknown ids are rejected rather than silently dropped.

The default input is stdin (`--input -`). Records retain input order and
are processed in parallel batches of at most 64. Each record is limited
to 8 MiB including its line ending; configure `--max-record-bytes` and
`--batch-size` to suit available memory. Memory grows with the current
batch, its encodings, and tokenizer state, rather than the whole dataset.
CRLF and a final record without a newline are accepted. Empty text is
valid; blank lines, unknown fields, invalid UTF-8/JSON, and oversized
records fail with a nonzero exit status and a record number.

Completed batches may already have been written when a later record
fails. No output is written for a batch whose parsing/encoding fails.
Write to a temporary output file and rename it only on command success
when the complete dataset must be published atomically.

## Inspect configuration

```sh
morpheme inspect -t tokenizer.json --json
```

The summary uses `schema_version: 1` and reports model/component names,
model and total vocabulary sizes, added-token count, special tokens with
their ids/options, and padding/truncation configuration. It does not dump
the complete model vocabulary. Without `--json`, inspection remains a
human-readable description.
