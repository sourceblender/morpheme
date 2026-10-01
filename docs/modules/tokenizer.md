# `tokenizer`

## Purpose

The top-level glue object. Holds a `Normalizer`, `PreTokenizer`,
`Model`, `PostProcessor`, `Decoder`, plus the vocabulary and config.

## Public API

```rust
pub struct Tokenizer { /* private */ }

impl Tokenizer {
    pub fn new(model: Box<dyn Model>) -> Self;
    pub fn with_normalizer(self, n: Box<dyn Normalizer>) -> Self;
    pub fn with_pre_tokenizer(self, p: Box<dyn PreTokenizer>) -> Self;
    pub fn with_post_processor(self, p: Box<dyn PostProcessor>) -> Self;
    pub fn with_decoder(self, d: Box<dyn Decoder>) -> Self;

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding>;
    pub fn encode_batch(&self, texts: &[&str], ...) -> Result<Vec<Encoding>>;
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String>;

    pub fn from_file(path: &Path) -> Result<Self>;
    pub fn to_file(&self, path: &Path, pretty: bool) -> Result<()>;

    pub fn vocab_size(&self) -> usize;
    pub fn token_to_id(&self, token: &str) -> Option<u32>;
    pub fn id_to_token(&self, id: u32) -> Option<String>;
}
```

## Algorithm

Encodes by running the pipeline in order. The high-level structure
matches [`docs/architecture.md`](../architecture.md).

## Performance notes

- The encode pipeline is the only thing that matters here.
- `encode_batch` uses `rayon` for parallelism.

## Test strategy

- Round-trip tests across multiple configs.
- Snapshot JSON for `to_file` against golden files.

## Known limitations

- Single-threaded `encode`; parallelism is only via `encode_batch`.