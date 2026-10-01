# Security policy

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |
| < latest | :x:                |

`splinter` is pre-alpha. Security fixes will be backported only at the
discretion of the maintainers, and only for the most recent release line.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

Report privately by emailing **security@sourceblender.dev**. Please include:

- A description of the issue and its impact.
- A minimal reproduction (failing test, snippet, or steps).
- Affected version / commit SHA.

You should receive an acknowledgement within **3 business days**. We aim to
disclose coordinated fixes within **90 days** of the report.

## Scope

In scope:

- Memory unsafety in the `splinter` library or `splinter-cli` binary.
- Panic-causing malformed tokenizer JSON that crosses a trust boundary.
- Dependency vulnerabilities that we can address without breaking compatibility.

Out of scope:

- Issues in upstream crates we don't control. We'll forward, not patch.
- Denial of service via pathological inputs (training on adversary-controlled
  corpora). Document assumptions, but no CVE.