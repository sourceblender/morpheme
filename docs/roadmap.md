# Roadmap

> Living document. Rough phases, not dates.

## Phase 0 — Scaffold ✅

- [x] Workspace + library + CLI skeleton.
- [x] GitHub Actions CI: fmt, clippy, test, MSRV.
- [x] Open-source docs tree.
- [x] MIT license.
- [x] Remote: `git@github.com:sourceblender/splinter`.

## Phase 1 — MVP BPE

- [x] `splinter::vocab` — `Vocab` with O(1) lookup, ordered ids.
- [x] `splinter::normalizer` — `Normalizer` trait + `IdentityNormalizer` (pass-through).
- [x] `splinter::pre_tokenizer` — `Whitespace` (BertPreTokenizer, ByteLevel land in 1.1).
- [x] `splinter::model::bpe` — pair merge loop with ranked merges (cache, SIMD land in Phase 4).
- [ ] `splinter::decoder` — `WordPieceDecoder`, `ByteLevelDecoder`.
- [x] `splinter::Tokenizer` — `encode`, `from_json`, `to_json`, `from_file`, `to_file`.
- [x] CLI: `splinter encode`, `splinter inspect` (`splinter decode` pending decoder).

### Phase 1.1

- [x] `BertNormalizer`, `NFD`/`NFKC`, `Lowercase`, `StripAccents`, `Replace`.
- [x] `BertPreTokenizer`, `ByteLevel` (alphabet + byte mapping).
- [x] `WordPieceDecoder`, `ByteLevelDecoder`.
- [x] `splinter decode` CLI command.
- [ ] `RegexReplace` normalizer (lands in 1.2 — adds the `regex` dep).

## Phase 2 — WordPiece + Unigram

- [x] `splinter::trainer` — BPE trainer with incremental updates,
  alphabet pre-population, min-pair-frequency, and a deterministic
  alphabetical tiebreak.
- [x] `splinter::model::wordpiece` — greedy longest-match with vocab.
- [x] `splinter::trainer::wordpiece` — additive WordPiece trainer.
- [x] `splinter::model::unigram` — Viterbi best-path over subword lattice.
- [x] `splinter::trainer::unigram` — EM trainer.
- [x] `splinter::pre_tokenizer::metaspace` — whitespace→marker.

## Phase 3 — HF `tokenizers.json` interop

- [x] `splinter::tokenizer::hf` — load HF-format JSON files. Supports
  the common normalizer / pre-tokenizer / decoder / model
  components; rejects unsupported ones with a clear error.
- [x] CLI auto-detects HF vs splinter JSON format.
- [x] HF BPE `byte_fallback` and `dropout` supported (Phase 3.1).
- [ ] HF BPE `continuing_subword_suffix` — rare.
- [x] Post-processors: `RobertaPostProcessor`, `TemplatePostProcessor`
  (Phase 3.2). In-memory API; JSON round-trip and HF loader
  support deferred.
- [x] HF loader for `RobertaProcessing` and `TemplateProcessing`
  (Phase 3.3). JSON round-trip on the splinter side still deferred.
- [x] Performance pass (Phase 4): `rayon`-parallel BPE trainer
  initial pass, subword trie for Unigram Viterbi.

## Phase 3 — Interop

- [ ] Load `tokenizer.json` (HF format) — best-effort, with warnings for unsupported pieces.
- [ ] Save in HF format — round-trip test corpus.
- [ ] Document drift in [`docs/interop.md`](./interop.md).

## Phase 4 — Performance

- [ ] Criterion benchmarks against `tokenizers` for BPE / WordPiece / Unigram.
- [ ] `memchr`-driven ASCII fast paths.
- [ ] Cached encode lookups.
- [ ] Multi-threaded trainer.

## Phase 5 — Ecosystem

- [ ] Python bindings via PyO3.
- [ ] WASM target via `wasm-bindgen`.
- [ ] Publish `crates/splinter` to crates.io.
- [ ] Pre-built binaries for the CLI.

## Out of scope

- Tokenizer-free LLMs.
- Training of language models.
- A model registry.