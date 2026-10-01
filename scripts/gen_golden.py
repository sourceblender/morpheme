#!/usr/bin/env python3
"""Generate golden encode/decode outputs from Hugging Face `tokenizers`.

The goldens are the ground truth for `crates/splinter/tests/hf_golden.rs`:
splinter must produce byte-identical ids, tokens, offsets, type ids,
masks, and decoded strings for every fixture in `scripts/hf-fixtures.txt`.

Usage (needs the fixtures from scripts/fetch-hf-fixtures.sh):

    uv run --with tokenizers==0.23.2 scripts/gen_golden.py

Re-run whenever the sentence list or fixture set changes, and commit the
resulting `crates/splinter/tests/golden/*.json`.
"""

import json
import pathlib

import tokenizers
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "crates/splinter/tests/data/hf"
OUT = ROOT / "crates/splinter/tests/golden"

SENTENCES = [
    "",
    " ",
    "Hello, World!",
    "hello world",
    "  leading and trailing spaces  ",
    "multiple   internal    spaces",
    "tabs\tand\nnewlines\r\nmixed",
    "The quick brown fox jumps over the lazy dog.",
    "I'm sure they'll say it's fine, don't you think? We've seen it.",
    "Café naïve résumé façade Ångström",
    "ÀÉÎÕÜ àéîõü",
    "é combining accent",
    "你好，世界！这是一个测试。",
    "日本語のテキストとカタカナ",
    "한국어 텍스트입니다",
    "Россия и Українa",
    "مرحبا بالعالم",
    "Emoji: 😀👍🏽🇺🇸 and ❤️",
    "Numbers 12345 and 3.14159 and 1,000,000",
    "Punctuation!!! ??? ... --- (parens) [brackets] {braces}",
    "zero​width and nbsp",
    "control\x00char\x07bell",
    "ALL CAPS SENTENCE HERE",
    "camelCaseIdentifier snake_case_identifier kebab-case",
    "https://example.com/path?query=1&x=y",
    "def foo(x):\n    return x ** 2  # comment",
    "a" * 120,
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "unaffable",
    "<s>special</s> [CLS] tokens [SEP] <|endoftext|> <unk> <pad> [MASK] <mask>",
    "Ġ Ċ ▁ ## </w>",
    "†Р Ġbyte-lookalikes",
]

PAIRS = [
    ("Hello, World!", "How are you?"),
    ("The quick brown fox", "jumps over the lazy dog."),
    ("Café", "你好"),
]


def enc_to_dict(e):
    return {
        "ids": e.ids,
        "tokens": e.tokens,
        "offsets": [list(o) for o in e.offsets],
        "type_ids": e.type_ids,
        "special_tokens_mask": e.special_tokens_mask,
        "attention_mask": e.attention_mask,
        "word_ids": e.word_ids,
    }


def gen(name: str, path: pathlib.Path):
    tok = Tokenizer.from_file(str(path))
    cases = []
    for s in SENTENCES:
        case = {"input": s}
        for add_special in (True, False):
            e = tok.encode(s, add_special_tokens=add_special)
            key = "with_special" if add_special else "without_special"
            case[key] = enc_to_dict(e)
        ids = case["with_special"]["ids"]
        case["decode_keep_special"] = tok.decode(ids, skip_special_tokens=False)
        case["decode_skip_special"] = tok.decode(ids, skip_special_tokens=True)
        cases.append(case)

    pairs = []
    for a, b in PAIRS:
        e = tok.encode(a, b, add_special_tokens=True)
        pairs.append({"a": a, "b": b, "encoding": enc_to_dict(e)})

    batch_inputs = ["short", "a somewhat longer sentence for padding", "Café 你好 😀"]
    batch = [enc_to_dict(e) for e in tok.encode_batch(batch_inputs)]

    ser = json.loads(tok.to_str())
    model = ser["model"]
    model_fp = {k: v for k, v in model.items() if k not in ("vocab", "merges")}
    vocab = model.get("vocab")
    model_fp["vocab_len"] = len(vocab) if vocab is not None else None
    model_fp["vocab_head"] = (list(vocab.items())[:5] if isinstance(vocab, dict) else (vocab or [])[:5])
    model_fp["vocab_tail"] = (list(vocab.items())[-5:] if isinstance(vocab, dict) else (vocab or [])[-5:])
    if "merges" in model:
        model_fp["merges_len"] = len(model["merges"])
        model_fp["merges_head"] = model["merges"][:5]
        model_fp["merges_tail"] = model["merges"][-5:]
    hf_serialized = {k: v for k, v in ser.items() if k != "model"}
    hf_serialized["model_fingerprint"] = model_fp

    golden = {
        "fixture": name,
        "tokenizers_version": tokenizers.__version__,
        "vocab_size_with_added": tok.get_vocab_size(with_added_tokens=True),
        "vocab_size_without_added": tok.get_vocab_size(with_added_tokens=False),
        "cases": cases,
        "pairs": pairs,
        "batch": {"inputs": batch_inputs, "encodings": batch},
        "hf_serialized": hf_serialized,
    }
    (OUT / f"{name}.json").write_text(
        json.dumps(golden, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {name}: {len(cases)} cases, {len(pairs)} pairs")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fixtures = [
        line.split()[0]
        for line in (ROOT / "scripts/hf-fixtures.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for name in fixtures:
        path = DATA / f"{name}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}; run scripts/fetch-hf-fixtures.sh first")
        gen(name, path)


if __name__ == "__main__":
    main()
