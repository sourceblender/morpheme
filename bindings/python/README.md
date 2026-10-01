# morpheme (Python)

Minimal Python bindings for the [morpheme](https://github.com/sourceblender/morpheme)
tokenizer library, built with [PyO3](https://pyo3.rs) and
[maturin](https://www.maturin.rs). The API is deliberately small: load a
tokenizer, encode, decode, count tokens, and look up the vocabulary. It does not
reproduce the Hugging Face `tokenizers` Python API.

## Usage

```python
import morpheme

tok = morpheme.Tokenizer.from_file("tokenizer.json")
# or: morpheme.Tokenizer.from_pretrained("bert-base-uncased")

enc = tok.encode("Hello world!")
print(enc.ids, enc.tokens, enc.offsets)  # offsets are character spans
print(tok.count("Hello world!"))          # == len(enc.ids)
print(tok.decode(enc.ids))                # "hello world!"
batch = tok.encode_batch(["a", "b"])      # releases the GIL
try:
    morpheme.Tokenizer.from_file("missing.json")
except morpheme.MorphemeError as e:
    print(e)
```

## Building locally

```sh
cd bindings/python
uv run --with maturin --with pytest sh -c 'maturin develop && pytest'
```

The Rust crate is `publish = false` (it is not on crates.io); wheels are built
with `maturin build --release`.

Prebuilt wheels cover Linux (x86_64, aarch64), macOS (Apple silicon) and Windows (x64); other platforms, including Intel macOS, install from the sdist and need a Rust toolchain.
