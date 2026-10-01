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
