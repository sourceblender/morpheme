// Smoke test for the generated JavaScript ABI. Run after building the
// `web` target into `../pkg` (see ../README.md):
//
//   node bindings/wasm/tests/smoke.mjs [pkg-dir]
//
// It needs the Hugging Face fixtures (`scripts/fetch-hf-fixtures.sh`).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkg = resolve(process.argv[2] ?? resolve(here, "../pkg"));
const fixtures = resolve(here, "../../../crates/morpheme/tests/data/hf");

const { default: init, Tokenizer } = await import(
  pathToFileURL(resolve(pkg, "morpheme_wasm.js")).href
);
await init({ module_or_path: readFileSync(resolve(pkg, "morpheme_wasm_bg.wasm")) });

const text = "Hello, world! Tokenizers in the browser.";

// gpt2: no special tokens; ids from `morpheme::Tokenizer::encode` on the
// pinned fixture (the golden tests check those against Hugging Face).
{
  const tok = Tokenizer.fromJson(readFileSync(resolve(fixtures, "gpt2.json"), "utf8"));
  const expectedIds = [15496, 11, 995, 0, 29130, 11341, 287, 262, 6444, 13];
  const expectedTokens = ["Hello", ",", "Ġworld", "!", "ĠToken", "izers", "Ġin", "Ġthe", "Ġbrowser", "."];

  const ids = tok.encode(text, true);
  assert.ok(ids instanceof Uint32Array, `encode returned ${ids?.constructor?.name}`);
  assert.deepEqual(Array.from(ids), expectedIds);
  assert.deepEqual(tok.tokens(text, true), expectedTokens);
  assert.equal(tok.count(text, true), ids.length);
  assert.equal(tok.decode(ids, true), text);
  assert.equal(tok.decode(new Uint32Array([15496, 11]), true), "Hello,");
  tok.free();
}

// bert-base-uncased: special tokens are added and skipped on decode.
{
  const tok = Tokenizer.fromJson(readFileSync(resolve(fixtures, "bert-base-uncased.json"), "utf8"));
  const expectedIds = [101, 7592, 1010, 2088, 999, 19204, 17629, 2015, 1999, 1996, 16602, 1012, 102];

  const ids = tok.encode(text, true);
  assert.deepEqual(Array.from(ids), expectedIds);
  assert.equal(tok.count(text, true), 13);
  assert.equal(tok.count(text, false), 11);
  assert.equal(tok.tokens(text, true)[0], "[CLS]");
  assert.equal(tok.decode(ids, true), "hello, world! tokenizers in the browser.");
  assert.ok(tok.decode(ids, false).startsWith("[CLS] hello"));
  tok.free();
}

// Errors are JavaScript exceptions carrying the Rust message.
assert.throws(
  () => Tokenizer.fromJson("{not json"),
  (e) => e instanceof Error && /json/i.test(e.message)
);

console.log("morpheme-wasm smoke test: ok");
