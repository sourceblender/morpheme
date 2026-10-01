# morpheme-wasm

Browser bindings for [morpheme](../../README.md) through
[`wasm-bindgen`](https://wasm-bindgen.github.io/wasm-bindgen/). The crate
depends on `morpheme` with `default-features = false`, so batch paths run
sequentially (no rayon) and `Tokenizer::save` is compiled out; loading
from JSON, encoding and decoding are exactly the native code.

The crate is `publish = false` and the package is not on npm yet: build
it from a checkout as described below.

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

```js
import init, { Tokenizer } from "./pkg/morpheme_wasm.js";

await init();
const tok = Tokenizer.fromJson(await (await fetch("tokenizer.json")).text());
const ids = tok.encode("Hello, world!", true); // Uint32Array
console.log(tok.count("Hello, world!", true), tok.decode(ids, true));
tok.free();
```

## Build

With [`wasm-pack`](https://drager.github.io/wasm-pack/) (`brew install
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
`wasm-bindgen` crate in `Cargo.lock`, currently 0.2.129):

```sh
cargo install --locked wasm-bindgen-cli --version 0.2.129
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

The crate also compiles natively, so the workspace clippy and build
cover it:

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
