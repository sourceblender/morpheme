#!/usr/bin/env python3
"""Build seed corpora for the fuzz targets from real tokenizer configs.

    python3 fuzz/make_corpus.py

Reads the example tokenizers (examples/*.json), the fetched Hugging Face
fixtures (crates/morpheme/tests/data/hf/*.json, if present) and the
golden inputs, and writes fuzz/corpus/<target>/. The corpus directory is
gitignored; regenerate it whenever you like.
"""

import base64
import hashlib
import json
import pathlib
import struct

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "fuzz/corpus"
KINDS = ["normalizer", "pre_tokenizer", "model", "post_processor", "decoder"]
# Must match `common::fixtures()` in fuzz_targets/common.rs: the `decode`
# target picks a tokenizer by index into this list.
DECODE_FIXTURES = ["bert-base-uncased", "gpt2", "llama", "t5-small", "qwen2.5"]


def write(target: str, data: bytes) -> None:
    d = OUT / target
    d.mkdir(parents=True, exist_ok=True)
    (d / hashlib.sha1(data).hexdigest()).write_bytes(data)


def charsmaps(value) -> list[bytes]:
    """Every `precompiled_charsmap` blob (decoded) found under `value`,
    including inside `Sequence` normalizers."""
    found = []
    if isinstance(value, dict):
        blob = value.get("precompiled_charsmap")
        if isinstance(blob, str):
            found.append(base64.b64decode(blob))
        for v in value.values():
            found += charsmaps(v)
    elif isinstance(value, list):
        for v in value:
            found += charsmaps(v)
    return found


def shrink_model(model: dict) -> dict:
    """Keep model configs small so the fuzzer can mutate them usefully."""
    m = dict(model)
    vocab = m.get("vocab")
    if isinstance(vocab, dict):
        keep = dict(sorted(vocab.items(), key=lambda kv: kv[1])[:300])
        m["vocab"] = keep
        if "merges" in m:
            merges = []
            for pair in m["merges"]:
                a, b = pair if isinstance(pair, list) else pair.split(" ", 1)
                if a in keep and b in keep and (a + b) in keep:
                    merges.append(pair)
            m["merges"] = merges
    elif isinstance(vocab, list):
        m["vocab"] = vocab[:300]
        if isinstance(m.get("unk_id"), int) and m["unk_id"] >= len(m["vocab"]):
            m["unk_id"] = 0
    return m


def main() -> None:
    sources = sorted((ROOT / "examples").glob("*.json"))
    sources += sorted((ROOT / "crates/morpheme/tests/data/hf").glob("*.json"))
    for path in sources:
        doc = json.loads(path.read_text(encoding="utf-8"))
        small = dict(doc)
        small["model"] = shrink_model(doc["model"])
        small["added_tokens"] = [
            t for t in doc.get("added_tokens", []) if t["id"] < 400
        ][:20]
        write("load_json", json.dumps(small, ensure_ascii=False).encode())
        for i, kind in enumerate(KINDS):
            value = small.get(kind)
            if value is not None:
                write("components_json", bytes([i]) + json.dumps(value, ensure_ascii=False).encode())
        # Real SentencePiece charsmaps (T5, ALBERT, XLM-R) as raw bytes.
        for blob in charsmaps(doc.get("normalizer")):
            write("precompiled", blob)

    for golden in sorted((ROOT / "crates/morpheme/tests/golden").glob("*.json")):
        doc = json.loads(golden.read_text(encoding="utf-8"))
        fixture = DECODE_FIXTURES.index(doc["fixture"]) if doc["fixture"] in DECODE_FIXTURES else None
        for case in doc["cases"]:
            text = case["input"].encode()
            write("encode", text)
            write("normalized_string", text)
            if fixture is not None:
                # Tokenizer index, then the golden ids as little-endian u32s;
                # `arbitrary` reads the Vec from this byte stream.
                ids = case["with_special"]["ids"]
                write("decode", bytes([fixture]) + struct.pack(f"<{len(ids)}I", *ids))
    print(f"wrote seed corpora to {OUT}")


if __name__ == "__main__":
    main()
