# 0002 — Correct upstream edge-case bugs that corrupt tokenizer results

- **Status:** accepted
- **Date:** 2026-10-01

## Context

Morpheme follows Hugging Face `tokenizers` and tests its behavior against
version 0.23.2. Repository review also reproduced correctness hazards
shared with that reference: added-token allocation can collide with sparse
model ids, retraining can retain stale special-token ids, and BPE can emit
byte fallback before an earlier unknown-token run. Reproducing these bugs
breaks decoding or associates tokens with the wrong input positions.

## Decision

Correct these cases rather than preserving corrupt results. Allocate new
ids above every occupied id and error on exhaustion; rebuild added-token
bindings after training; flush pending unknowns before byte fallback.
Also reject impossible special-token truncation budgets, preserve sequence
ownership in overflow, and enforce the public alignment APIs' contracts.

Keep the Hugging Face file format and the pinned real-model golden tests.
Document deliberate behavioral differences in `docs/interop.md`, and add
regression tests that check token ids, decoding, offsets and sequence
ownership rather than changing reference goldens to accommodate fixes.

## Consequences

- Ordinary supported tokenizer files remain interoperable with Hugging Face.
- The affected edge cases deliberately produce different results from
  `tokenizers` 0.23.2, or return an error instead of producing invalid output.
- Loading and saving existing vocabulary ids remains lossless. Retraining
  creates a new vocabulary and can therefore reassign added-token ids.
- Explicit post-processor and padding ids remain user configuration;
  callers must update them if retraining changes their vocabulary ids.

## Alternatives considered

- **Preserve every upstream bug.** Rejected because silent token-id
  collisions and reordered input violate round trips and alignment guarantees.
- **Change the tokenizer file format.** Unnecessary; the corrections fit
  the existing Hugging Face schema.
