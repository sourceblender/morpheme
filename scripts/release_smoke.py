#!/usr/bin/env python3
"""Verify a freshly downloaded release archive or an installed CLI."""

import argparse
import hashlib
import json
import pathlib
import subprocess
import tarfile
import tempfile
import zipfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def smoke(cli, version, work):
    def run(*args):
        return subprocess.check_output([str(cli), *map(str, args)], text=True, encoding="utf-8").strip()

    require(run("--version") == f"morpheme {version}", "CLI version mismatch")
    corpus = work / "corpus.txt"
    corpus.write_text("alpha beta gamma\nalpha beta\n", encoding="utf-8")
    for model in ["bpe", "wordpiece", "wordlevel", "unigram"]:
        tokenizer = work / f"{model}.json"
        run("train", "--quiet", "--model", model, "--vocab-size", "100",
            "--out", tokenizer, corpus)
        encoded = json.loads(run("encode", "-t", tokenizer, "--json", "alpha beta"))
        require(encoded["ids"] and len(encoded["ids"]) == len(encoded["tokens"]),
                f"{model}: inconsistent encoding")
        text = run("decode", "-t", tokenizer, "--skip-special-tokens",
                   ",".join(map(str, encoded["ids"])))
        require(text == "alpha beta", f"{model}: unexpected decoded text {text!r}")
        require(run("inspect", "-t", tokenizer), f"{model}: inspect returned no output")
        inspection = json.loads(run("inspect", "-t", tokenizer, "--json"))
        require(inspection["schema_version"] == 1, f"{model}: inspect schema mismatch")
        count = json.loads(run("count", "-t", tokenizer, "--json", "alpha beta"))
        require(count["count"] == len(encoded["ids"]), f"{model}: token count mismatch")
        records = [{"id": None, "text": "alpha beta"}, {"id": "café", "text": "beta alpha"}]
        inputs = work / f"{model}-inputs.jsonl"
        inputs.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records), encoding="utf-8")
        batch = [json.loads(line) for line in run("encode-batch", "-t", tokenizer,
                    "--input", inputs, "--batch-size", "1").splitlines()]
        require(len(batch) == len(records), f"{model}: lost batch records")
        require([row["id"] for row in batch] == [row["id"] for row in records],
                f"{model}: batch order or IDs changed")
        require(batch[0]["encoding"]["ids"] == encoded["ids"], f"{model}: batch encoding differs")
        inputs.write_text("\n".join(json.dumps({"id": row["id"], "ids": row["encoding"]["ids"]},
                            ensure_ascii=False) for row in batch), encoding="utf-8")
        decoded = [json.loads(line) for line in run("decode-batch", "-t", tokenizer,
                    "--input", inputs, "--skip-special-tokens", "--batch-size", "1").splitlines()]
        require(decoded == records, f"{model}: batch roundtrip differs")
    print(f"Release smoke passed: {cli} ({version})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--target")
    source.add_argument("--cli", type=pathlib.Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="morpheme-release-") as temporary:
        work = pathlib.Path(temporary)
        if args.cli:
            cli = args.cli.resolve()
        else:
            windows = "windows" in args.target
            archive = f"morpheme-cli-{args.target}.{'zip' if windows else 'tar.xz'}"
            subprocess.run(["gh", "release", "download", f"v{args.version}",
                            "--repo", "sourceblender/morpheme", "--dir", str(work),
                            "--pattern", archive, "--pattern", archive + ".sha256"], check=True)
            expected = (work / (archive + ".sha256")).read_text().split()[0]
            require(hashlib.sha256((work / archive).read_bytes()).hexdigest() == expected,
                    "archive checksum mismatch")
            # Extract only the executable, never arbitrary archive paths.
            executable = "morpheme.exe" if windows else "morpheme"
            cli = work / executable
            if windows:
                with zipfile.ZipFile(work / archive) as package:
                    matches = [n for n in package.namelist()
                               if pathlib.PurePosixPath(n).name == executable]
                    require(len(matches) == 1, f"expected one executable, found {matches!r}")
                    cli.write_bytes(package.read(matches[0]))
            else:
                with tarfile.open(work / archive) as package:
                    matches = [m for m in package.getmembers()
                               if m.isfile() and pathlib.PurePosixPath(m.name).name == executable]
                    require(len(matches) == 1, f"expected one executable, found {matches!r}")
                    with package.extractfile(matches[0]) as binary:
                        cli.write_bytes(binary.read())
                cli.chmod(0o755)
        smoke(cli, args.version, work)


if __name__ == "__main__":
    main()
