# morpheme (Python)

Python bindings for the [morpheme](https://github.com/sourceblender/morpheme)
tokenizer library: fast, Hugging Face-compatible `tokenizer.json`
loading, encoding and decoding, built with [PyO3](https://pyo3.rs) and
[maturin](https://www.maturin.rs). The API is deliberately small: load a
tokenizer, encode, decode, count tokens, and look up the vocabulary. It
does not reproduce the Hugging Face `tokenizers` Python API, and it does
not train tokenizers (use the Rust library or the `morpheme` CLI).

## Install

```sh
pip install morpheme
```

Prebuilt abi3 wheels (one wheel per platform for CPython 3.9 and later)
cover Linux x86_64 and aarch64 (manylinux), macOS on Apple silicon, and
Windows x64. Other platforms, including Intel macOS, install from the
sdist and need a [Rust toolchain](https://rustup.rs).

## Usage

```python
import morpheme

tok = morpheme.Tokenizer.from_pretrained("google-bert/bert-base-uncased")
# or: morpheme.Tokenizer.from_file("tokenizer.json")
#     morpheme.Tokenizer.from_str(json_text)

enc = tok.encode("Hello, world!")
print(enc.ids)                # [101, 7592, 1010, 2088, 999, 102]
print(enc.tokens)             # ['[CLS]', 'hello', ',', 'world', '!', '[SEP]']
print(enc.offsets)            # character spans: [(0, 0), (0, 5), (5, 6), ...]
print(tok.count("Hello, world!"))  # 6, the same as len(enc.ids)
print(tok.decode(enc.ids))    # hello, world!

batch = tok.encode_batch(["a", "b"])  # releases the GIL while encoding
try:
    morpheme.Tokenizer.from_file("missing.json")
except morpheme.MorphemeError as e:
    print(e)
```

| `Tokenizer` | |
| --- | --- |
| `from_file(path)`, `from_str(json)`, `from_pretrained(repo_id, revision=None)` | Load a `tokenizer.json` (the Hub download uses the cache shared with Python `huggingface_hub`; `HF_TOKEN` and `HF_HUB_OFFLINE` apply) |
| `encode(text, add_special_tokens=True)`, `encode_batch(texts, add_special_tokens=True)` | `Encoding` objects with `ids`, `tokens`, `offsets` (character spans), `type_ids`, `attention_mask`, `special_tokens_mask` |
| `decode(ids, skip_special_tokens=True)`, `decode_batch(sequences, skip_special_tokens=True)` | Text |
| `count(text, add_special_tokens=True)` | Number of ids `encode` returns (the file's padding/truncation settings apply) |
| `token_to_id(token)`, `id_to_token(id)`, `vocab_size(with_added_tokens=True)` | Vocabulary lookups |
| `to_str(pretty=False)`, `save(path, pretty=False)` | Write `tokenizer.json` |

Errors raise `morpheme.MorphemeError`; `morpheme.__version__` is the
package version.

## Building locally

From a checkout, with the Hugging Face fixtures the tests use
(`./scripts/fetch-hf-fixtures.sh` at the repository root):

```sh
python -m venv .venv
.venv/bin/pip install maturin pytest
(cd bindings/python && ../../.venv/bin/maturin develop --locked)
.venv/bin/pytest bindings/python/tests
```

This is what CI runs, from the repository root. `maturin develop`
installs the extension into the active virtualenv, or into a `.venv` in
the current directory or one of its parents.

The Rust crate (`morpheme-python`) is `publish = false` (it is not on
crates.io); release wheels are built by the `Python wheels` workflow with
`maturin build --release` and uploaded to PyPI when a version tag is
pushed.
