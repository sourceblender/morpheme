# Architecture Decision Records

We use ADRs to capture the *why* behind substantial design choices.
Each ADR is a short markdown file, one decision per file.

## Index

| Number | Title | Status |
| ------ | ----- | ------ |
| [0001](./0001-adopt-hugging-face-tokenizers-design.md) | Adopt the Hugging Face `tokenizers` design and file format | accepted |
| [0002](./0002-correct-upstream-edge-case-bugs.md) | Correct upstream edge-case bugs | accepted |
| [0003](./0003-verify-cache-and-save-atomically.md) | Verify cache contents and save tokenizer files atomically | accepted |

## Conventions

- Filename: `NNNN-short-title.md`.
- Status: `proposed` → `accepted` → `superseded by NNNN`.
- We don't delete superseded ADRs. We just update the status.

Start from [`0000-template.md`](./0000-template.md).
