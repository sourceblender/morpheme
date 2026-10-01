# 0003 — Verify cache contents and save tokenizer files atomically

- **Status:** accepted
- **Date:** 2026-10-01

## Context

Concurrent cache publication now uses complete temporary files, but a
successful transfer does not prove content integrity. A corrupted cached
blob can also be reused indefinitely. Direct tokenizer saves can expose
partial JSON to readers or damage an existing file on interruption.

## Decision

Require content-addressed Hub ETags: Git object SHA-1 (including the blob
header) or raw LFS SHA-256. Verify downloads before publication and cache
entries before reuse. Repair corruption online; reject unverifiable or
corrupt cache entries offline. Copied snapshots are verified against the
matching content-addressed blob, while symlink names carry their hash.

Serialize tokenizer files first, write a unique temporary file in the
destination directory, synchronize it, and atomically replace the
destination. Preserve existing file permissions. Replacing a symlink
replaces the link itself, so shared Hub blobs are not modified by saves.

## Consequences

- Cache reads perform hashing, trading some I/O for verified contents.
- Mirrors must provide a Git SHA-1 or LFS SHA-256 ETag; opaque ETags and
  standalone snapshot copies without their blob are not accepted.
- Readers observe a complete old or new tokenizer file. Directory entry
  durability after power loss is not guaranteed by this API.
- Atomic file saving requires `tempfile` even without the Hub feature;
  hashing dependencies remain optional behind `hub`.

## Notes (2026-10-01, issues #29, #36, #37, #38)

- **New files are `0644` on Unix.** `tempfile` creates temporary files
  `0600`, and a plain rename carries that mode into the cache and into a
  freshly saved `tokenizer.json`. Blobs, refs, copied snapshots and new
  saves now get `0644` (what `huggingface_hub` produces under the default
  umask) before publication; existing files keep their permissions, as
  before. `std` has no umask API, so the mode is fixed rather than
  umask-derived.
- **Repo ids may not contain `--`.** The cache directory joins `org` and
  `name` with `--`, so `acme--model` and `acme/model` would share a
  folder (and `refs/`). `huggingface_hub` rejects such ids too.
- **Tokens stay on the endpoint's origin.** The blob request only carries
  the bearer token when the download URL has the same scheme, host and
  port as the endpoint, so an `https` endpoint redirecting to `http://`
  never sees the token over cleartext. An `https` endpoint also builds its
  agents with `https_only`, and URLs in error messages are stripped of
  their query string so presigned CDN URLs do not leak into logs.
- **Resilience.** Requests time out after 10 s waiting for response
  headers or body data (not only on connect). A `429`/`5xx` is retried
  once after a short pause; if the metadata request still fails with a
  transient status, a valid cached snapshot of the requested revision is
  served, otherwise the HTTP error is returned. `401`/`403`/`404` remain
  definitive. When the revision is a commit hash, the Hub's
  `x-repo-commit` must equal it, so a mirror serving another commit is an
  error rather than a misleading `refs/<hash>` entry.
