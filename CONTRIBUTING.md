# Contributing to morpheme

Thanks for your interest in contributing! This document covers the day-to-day workflow. The deeper design discussion lives in [`docs/contributing.md`](./docs/contributing.md).

## Code of conduct

By participating, you agree to abide by the [Code of Conduct](./CODE_OF_CONDUCT.md).

## Getting started

1. Fork the repository.
2. Create a topic branch: `git switch -c feat/my-change`.
3. Make your change. Add tests. Update docs.
4. Fetch the Hugging Face fixtures used by the golden tests (once):
   ```sh
   ./scripts/fetch-hf-fixtures.sh
   ```
5. Run the local gate (`just gate` runs the same three commands):
   ```sh
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   ```
   CI also checks all-features and no-default-features builds, rustdoc,
   the MSRV, wasm32, and the Python and WASM bindings. If your change
   touches those areas, run the matching commands from the
   [full local gate](./docs/contributing.md#local-gate).
6. Open a pull request. Fill out the PR template.

## Commit messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

<body>

<footer>
```

Common scopes: `lib`, `cli`, `models`, `normalizers`, `pre-tokenizers`, `processors`, `decoders`, `trainers`, `hub`, `interop`, `python`, `wasm`, `bench`, `fuzz`, `docs`, `ci`, `release`.

## Reporting bugs

Use the [bug report issue template](./.github/ISSUE_TEMPLATE/bug_report.md). Include a minimal reproduction.

## Suggesting features

Use the [feature request template](./.github/ISSUE_TEMPLATE/feature_request.md). Reference the relevant module in [`docs/modules/`](./docs/modules) if it exists.

## Security issues

See [`SECURITY.md`](./SECURITY.md). Do **not** open a public issue.

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT license](./LICENSE).