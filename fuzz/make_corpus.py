#!/usr/bin/env python3
"""Build seed corpora for the fuzz targets from real tokenizer configs.

    python3 fuzz/make_corpus.py

Reads the example tokenizers (examples/*.json), the fetched Hugging Face
fixtures (crates/morpheme/tests/data/hf/*.json, if present) and the
golden inputs, and writes fuzz/corpus/<target>/. The corpus directory is
gitignored; regenerate it whenever you like.
"""

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "fuzz/corpus"
KINDS = ["normalizer", "pre_tokenizer", "model", "post_processor", "decoder"]


def write(target: str, data: bytes) -> None:
    d = OUT / target
    d.mkdir(parents=True, exist_ok=True)
    (d / hashlib.sha1(data).hexdigest()).write_bytes(data)


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

    for golden in sorted((ROOT / "crates/morpheme/tests/golden").glob("*.json")):
        for case in json.loads(golden.read_text(encoding="utf-8"))["cases"]:
            text = case["input"].encode()
            write("encode", text)
            write("normalized_string", text)
    print(f"wrote seed corpora to {OUT}")


if __name__ == "__main__":
    main()
