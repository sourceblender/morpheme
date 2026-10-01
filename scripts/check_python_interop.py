#!/usr/bin/env python3
"""Check that Hugging Face `tokenizers` loads splinter-trained files and
encodes exactly like splinter does.

Trains one tokenizer per model type with the splinter CLI, then compares
ids, tokens, char offsets and decoded text between the CLI and Python.

    uv run --with tokenizers==0.23.2 scripts/check_python_interop.py
"""

import json
import pathlib
import subprocess
import sys
import tempfile

from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEXTS = [
    "The café's crème brûlée costs $14.50!",
    "Hello, world! 😀 Zürich and 你好",
    "  odd   spacing\ttabs\nand newlines ",
    "[CLS] special tokens [SEP] <|endoftext|> <unk>",
    "",
]


def main() -> int:
    subprocess.run(
        ["cargo", "build", "--quiet", "--release", "-p", "splinter-cli"], cwd=ROOT, check=True
    )
    cli = ROOT / "target/release/splinter"
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for model in ["bpe", "wordpiece", "unigram", "wordlevel"]:
            path = pathlib.Path(tmp) / f"{model}.json"
            subprocess.run(
                [cli, "train", "--model", model, "--vocab-size", "300", "--out", path,
                 ROOT / "examples/corpus.txt"],
                check=True, capture_output=True,
            )
            hf = Tokenizer.from_file(str(path))
            for text in TEXTS:
                out = subprocess.run(
                    [cli, "encode", "-t", path, "--char-offsets", "--json", "--", text],
                    check=True, capture_output=True, text=True,
                ).stdout
                ours = json.loads(out)
                e = hf.encode(text)
                want = {
                    "ids": e.ids,
                    "tokens": e.tokens,
                    "offsets": [list(o) for o in e.offsets],
                    "type_ids": e.type_ids,
                }
                bad = [k for k in want if ours[k] != want[k]]
                ids = ",".join(map(str, e.ids)) or None
                if ids:
                    decoded = subprocess.run(
                        [cli, "decode", "-t", path, ids],
                        check=True, capture_output=True, text=True,
                    ).stdout[:-1]
                    if decoded != hf.decode(e.ids, skip_special_tokens=False):
                        bad.append("decode")
                status = "ok  " if not bad else "FAIL"
                print(f"{status} {model:<9} {text[:30]!r} {' '.join(bad)}")
                failures += bool(bad)
    print(f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
