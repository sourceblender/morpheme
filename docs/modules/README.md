# Modules

> One deep-dive per library module. Stub files live here and grow as the
> code does.

- [`normalizer.md`](./normalizer.md)
- [`pre-tokenizer.md`](./pre-tokenizer.md)
- [`bpe.md`](./bpe.md)
- [`wordpiece.md`](./wordpiece.md)
- [`unigram.md`](./unigram.md)
- [`post-processor.md`](./post-processor.md)
- [`decoder.md`](./decoder.md)
- [`trainer.md`](./trainer.md)
- [`tokenizer.md`](./tokenizer.md)

## Conventions

Each module doc follows the same skeleton:

1. **Purpose** — what this module owns.
2. **Public API** — the traits and concrete types exposed.
3. **Algorithm** — enough to read the code, no pseudocode.
4. **Performance notes** — where we spend the time.
5. **Test strategy** — what we cover and how.
6. **Known limitations** — what's intentionally not done yet.