# Security policy

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| < 0.4   | :x:                |

`morpheme` is pre-1.0 (v0.x). Security fixes go into the latest release
line only: the `morpheme` and `morpheme-cli` crates, the prebuilt CLI
binaries, and the `morpheme` Python package, which are released together
under the same version. The WASM bindings are not released as a package;
fixes for them land on `main`, so rebuild from the latest release tag or
`main`. Older lines get fixes only at the maintainers' discretion.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

Report privately by emailing **security@sourceblender.dev**. Please include:

- A description of the issue and its impact.
- A minimal reproduction (failing test, snippet, or steps).
- Affected version / commit SHA, and which artifact (Rust crate, CLI
  binary, Python wheel, or WASM build).

You should receive an acknowledgement within **3 business days**. We aim to
disclose coordinated fixes within **90 days** of the report.

## Scope

In scope:

- Memory unsafety in the `morpheme` library, the `morpheme-cli` binary,
  or the Python and WASM bindings.
- Panic-causing malformed tokenizer JSON that crosses a trust boundary.
  (Loading, encoding and decoding are fuzzed weekly in CI; see the
  fuzzing section of [`docs/contributing.md`](./docs/contributing.md#fuzzing).)
- Hub client issues: credential leakage (`HF_TOKEN`), cache integrity
  or path handling in the `hub` feature.
- Dependency vulnerabilities that we can address without breaking compatibility.

Out of scope:

- Issues in upstream crates we don't control. We'll forward, not patch.
- Denial of service via pathological inputs (training on adversary-controlled
  corpora). Document assumptions, but no CVE.
