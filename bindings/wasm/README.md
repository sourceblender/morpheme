# morpheme-wasm

Browser bindings for [morpheme](../../README.md) through
[`wasm-bindgen`](https://rustwasm.github.io/wasm-bindgen/). The crate
depends on `morpheme` with `default-features = false`, so batch paths run
sequentially (no rayon) and `Tokenizer::save` is compiled out; loading
from JSON, encoding and decoding are exactly the native code.

## API

```ts
class Tokenizer {
  static fromJson(json: string): Tokenizer;            // throws on invalid JSON
  encode(text: string, addSpecialTokens: boolean): Uint32Array;
  tokens(text: string, addSpecialTokens: boolean): string[];
  count(text: string, addSpecialTokens: boolean): number;
  decode(ids: Uint32Array, skipSpecialTokens: boolean): string;
  free(): void;                                        // release the wasm memory
}
```

Errors from the Rust side are thrown as JavaScript `Error`s with the
Rust error message.

## Build

With [`wasm-pack`](https://rustwasm.github.io/wasm-pack/) (`brew install
wasm-pack` or `cargo install wasm-pack`):

```sh
rustup target add wasm32-unknown-unknown
cd bindings/wasm
wasm-pack build --target web --release
```

This writes `pkg/morpheme_wasm.js`, `pkg/morpheme_wasm_bg.wasm` and
TypeScript declarations. For a bundler, use `--target bundler`; for Node,
`--target nodejs`.

Without `wasm-pack`, use `wasm-bindgen-cli` (its version must match the
`wasm-bindgen` crate in `Cargo.lock`):

```sh
cargo build -p morpheme-wasm --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir bindings/wasm/pkg \
  target/wasm32-unknown-unknown/release/morpheme_wasm.wasm
```

## Example

`www/index.html` fetches a `tokenizer.json` from a URL, then shows the
token count and tokens for a textarea as you type. It imports the module
from `../pkg`, so build first, then serve the crate directory (ES modules
and `fetch` need an HTTP origin, not `file://`):

```sh
cd bindings/wasm
python3 -m http.server 8080
# open http://localhost:8080/www/
```

The default URL points at GPT-2 on the Hugging Face Hub, which serves
`tokenizer.json` with CORS headers.

## Checks

The crate also compiles natively so the workspace gate covers it:

```sh
cargo check -p morpheme-wasm --target wasm32-unknown-unknown
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

`tests/smoke.mjs` exercises the generated JavaScript ABI in Node against
the Hugging Face fixtures (`encode` returns a `Uint32Array` with the ids
the Rust encoder produces, `decode` round-trips, `count` equals the id
count, and invalid JSON throws). CI runs it after building the `web`
target with `wasm-bindgen-cli`; locally:

```sh
./scripts/fetch-hf-fixtures.sh
(cd bindings/wasm && wasm-pack build --target web --release)
node bindings/wasm/tests/smoke.mjs        # or: node ... <pkg-dir>
```
