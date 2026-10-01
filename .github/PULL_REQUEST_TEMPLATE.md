---
name: Pull request
about: Open a pull request
---

## Summary

<!-- One paragraph: what does this change, and why? -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation
- [ ] Refactor / chore
- [ ] CI / tooling

## Module(s) touched

<!-- Tick all that apply; reference docs/modules/. -->

- [ ] `normalizer`
- [ ] `pre_tokenizer`
- [ ] `bpe`
- [ ] `wordpiece`
- [ ] `unigram`
- [ ] `post_processor`
- [ ] `decoder`
- [ ] `trainer`
- [ ] `tokenizer` (glue)
- [ ] `cli`
- [ ] other (describe)

## Checklist

- [ ] I have read [`CONTRIBUTING.md`](../CONTRIBUTING.md) and [`docs/contributing.md`](../docs/contributing.md).
- [ ] I added / updated tests for the change.
- [ ] I updated the relevant docs (`docs/modules/*.md`, `CHANGELOG.md`,
      an ADR, etc.).
- [ ] I ran the local gate:
  - [ ] `cargo fmt --all -- --check`
  - [ ] `cargo clippy --workspace --all-targets -- -D warnings`
  - [ ] `cargo test --workspace`

## Related issues

<!-- `Fixes #123`, `Relates to #456`, etc. -->

## Screenshots / output

<!-- If relevant. Remove if not. -->