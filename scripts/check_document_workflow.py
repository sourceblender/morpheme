#!/usr/bin/env python3
"""Exercise the executable Rust document consumer, including atomic failures."""

import hashlib
import argparse
import json
import os
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", action="store_true", help="also exercise a fresh pinned Hub load and offline replay")
    args = parser.parse_args()
    subprocess.run(["cargo", "build", "--release", "--locked", "-p", "morpheme-cli"], cwd=ROOT, check=True)
    subprocess.run(["cargo", "build", "--release", "--locked", "-p", "morpheme", "--features", "hub",
                    "--example", "budget_documents"], cwd=ROOT, check=True)
    suffix = ".exe" if os.name == "nt" else ""
    cli = ROOT / "target/release" / ("morpheme" + suffix)
    consumer = ROOT / "target/release/examples" / ("budget_documents" + suffix)
    with tempfile.TemporaryDirectory(prefix="morpheme-documents-") as temporary:
        work = pathlib.Path(temporary)
        corpus = work / "corpus.txt"
        corpus.write_text("hello café 😀\nalpha beta gamma\n", encoding="utf-8")
        tokenizer = work / "tokenizer.json"
        subprocess.run([cli, "train", "--quiet", "--model", "bpe", "--vocab-size", "300",
                        "--out", tokenizer, corpus], check=True)
        # File settings must not hide an oversized document.
        config = json.loads(tokenizer.read_text(encoding="utf-8"))
        config["truncation"] = {"direction":"Right", "max_length":1, "strategy":"LongestFirst", "stride":0}
        tokenizer.write_text(json.dumps(config), encoding="utf-8")
        inputs = work / "documents.jsonl"
        texts = ["hello café 😀", "", "alpha beta gamma"]
        inputs.write_text("\n".join(json.dumps({"id":i, "text":text}, ensure_ascii=False)
                                    for i, text in enumerate(texts)), encoding="utf-8")
        output = work / "prepared.jsonl"
        subprocess.run([consumer, tokenizer, "100", inputs, output], check=True)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        if len(rows) != 3 or [row["text"] for row in rows] != texts:
            raise RuntimeError("document order/text changed")
        expected_hash = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
        for row in rows:
            if row["token_count"] != len(row["ids"]) or row["tokenizer"]["sha256"] != expected_hash:
                raise RuntimeError("incorrect count or tokenizer fingerprint")
        before = output.read_bytes()
        for contents, expected in [
            (json.dumps({"id":"long", "text":"alpha " * 100}), "exceeds budget"),
            (json.dumps({"id":1, "text":"ok"}) + "\nnot-json", "record 2"),
        ]:
            inputs.write_text(contents, encoding="utf-8")
            result = subprocess.run([consumer, tokenizer, "5", inputs, output], capture_output=True, encoding="utf-8")
            if result.returncode == 0 or expected not in result.stderr or output.read_bytes() != before:
                raise RuntimeError("failed preparation changed the published dataset or hid an error")
        # One process, many distinct records; output is streamed, not retained.
        with inputs.open("w", encoding="utf-8") as stream:
            for i in range(10_000):
                stream.write(json.dumps({"id":i, "text":f"document {i} café 😀"}, ensure_ascii=False) + "\n")
        subprocess.run([consumer, tokenizer, "100", inputs, output], check=True)
        with output.open(encoding="utf-8") as stream:
            count = sum(1 for _ in stream)
        if count != 10_000:
            raise RuntimeError("stress run lost records")
        if args.hub:
            inputs.write_text(json.dumps({"id":"hub", "text":"Hello café 😀"}) + "\n", encoding="utf-8")
            revision = "86b5e0934494bd15c9632b12f734a8a67f723594"
            env = dict(os.environ, HF_HUB_CACHE=str(work / "cache"), HF_HUB_OFFLINE="0")
            command = [consumer, "google-bert/bert-base-uncased", "64", inputs, output, revision]
            subprocess.run(command, env=env, check=True)
            online = output.read_bytes()
            env["HF_HUB_OFFLINE"] = "1"
            subprocess.run(command, env=env, check=True)
            if output.read_bytes() != online:
                raise RuntimeError("offline pinned replay differs")
        # The released 0.1.1 library API also supports this consumer; docs
        # provide a standalone manifest for using it without this checkout.
    print("Document workflow: Unicode, budgets, fingerprints, atomic failures, and 10,000 records passed")


if __name__ == "__main__":
    main()
