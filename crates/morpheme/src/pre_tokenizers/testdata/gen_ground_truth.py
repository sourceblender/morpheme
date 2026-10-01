# Regenerate ground_truth.json (needs scripts/fetch-hf-fixtures.sh first):
#   uv run --with tokenizers==0.23.2 crates/morpheme/src/pre_tokenizers/testdata/gen_ground_truth.py
import json, glob, os
from tokenizers import Tokenizer, pre_tokenizers as P, decoders as D
from tokenizers import Regex
INPUTS = [
    "", " ", "Hello, World!", "  leading and trailing spaces  ", "multiple   internal    spaces",
    "tabs\tand\nnewlines\r\nmixed", "I'm sure they'll say it's fine, don't you think? We've seen it.",
    "Café naïve résumé façade Ångström", "é combining", "你好，世界！这是一个测试。",
    "日本語のテキストとカタカナ ー", "한국어 텍스트입니다", "Россия и Українa", "مرحبا بالعالم",
    "Emoji: 😀👍🏽🇺🇸 and ❤️", "Numbers 12345 and 3.14159 and 1,000,000", "Punctuation!!! ??? ... --- (parens)",
    "zero​width and nbsp", "def foo(x):\n    return x ** 2  # comment", "▁already▁marked ▁",
    "ABC123def456", "どこで生れ。Yes", "Apples are りんご 林檎",
]
QWEN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
CONFIGS = [
    {"type": "BertPreTokenizer"},
    {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
    {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
    {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
    {"type": "Whitespace"},
    {"type": "WhitespaceSplit"},
    {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True},
    {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": True},
    {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never", "split": False},
    {"type": "Punctuation", "behavior": "Isolated"},
    {"type": "Punctuation", "behavior": "Contiguous"},
    {"type": "Digits", "individual_digits": True},
    {"type": "Digits", "individual_digits": False},
    {"type": "CharDelimiterSplit", "delimiter": " "},
    {"type": "UnicodeScripts"},
    {"type": "FixedLength", "length": 3},
    {"type": "Split", "pattern": {"Regex": QWEN}, "behavior": "Isolated", "invert": False},
    {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithNext", "invert": False},
    {"type": "Split", "pattern": {"Regex": "\\w+"}, "behavior": "Removed", "invert": True},
    {"type": "Sequence", "pretokenizers": [{"type": "WhitespaceSplit"}, {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True}]},
    {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": QWEN}, "behavior": "Isolated", "invert": False}, {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]},
]
def build(cfg):
    # Build via a dummy tokenizer JSON so we exercise HF's own deserializer.
    t = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "normalizer": None,
         "pre_tokenizer": cfg, "post_processor": None, "decoder": None,
         "model": {"type": "WordLevel", "vocab": {"a": 0}, "unk_token": "a"}}
    return Tokenizer.from_str(json.dumps(t)).pre_tokenizer
pretok = []
for cfg in CONFIGS:
    pt = build(cfg)
    pretok.append({"config": cfg, "cases": [{"input": s, "splits": [[a, list(o)] for a, o in pt.pre_tokenize_str(s)]} for s in INPUTS]})

DEC_INPUTS = [
    ["Hello", "Ġmy", "Ġfriend", ",", "Ġhow", "Ġis", "Ġ", "ĠĠ", "Ã©", "ðŁĺ", "Ģ", "<|endoftext|>"],
    ["▁Hey", "▁friend", "!", "▁", "▁▁how", "<0xE5>", "<0x8F>", "<0xAB>", "<0x61>", "<0xE5>", "x"],
    ["hello", "##world", "##s", "!", "do", "not", "'", "s", "I", "'m", "##", "#"],
    ["My</w>", "na", "me</w>", "is</w>", "</w>"],
    ["<pad>", "h", "h", "e", "<pad>", "l", "l", "|", "w", "|", "|"],
    [],
    ["  spaced  ", "x"],
]
DEC_CONFIGS = [
    {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
    {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True},
    {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never", "split": True},
    {"type": "WordPiece", "prefix": "##", "cleanup": True},
    {"type": "WordPiece", "prefix": "##", "cleanup": False},
    {"type": "BPEDecoder", "suffix": "</w>"},
    {"type": "CTC", "pad_token": "<pad>", "word_delimiter_token": "|", "cleanup": True},
    {"type": "ByteFallback"},
    {"type": "Fuse"},
    {"type": "Strip", "content": " ", "start": 1, "stop": 1},
    {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
    {"type": "Replace", "pattern": {"Regex": "[ae]+"}, "content": "_"},
    {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}, {"type": "Strip", "content": " ", "start": 1, "stop": 0}]},
]
def build_dec(cfg):
    t = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "normalizer": None,
         "pre_tokenizer": None, "post_processor": None, "decoder": cfg,
         "model": {"type": "WordLevel", "vocab": {"a": 0}, "unk_token": "a"}}
    return Tokenizer.from_str(json.dumps(t)).decoder
dec = []
for cfg in DEC_CONFIGS:
    d = build_dec(cfg)
    cases = []
    for toks in DEC_INPUTS:
        try:
            cases.append({"tokens": toks, "output": d.decode(toks)})
        except Exception as e:
            cases.append({"tokens": toks, "error": str(e)})
    dec.append({"config": cfg, "cases": cases})

fixtures = {}
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../tests/data/hf/*.json"))):
    name = os.path.basename(f)[:-5]
    raw = json.load(open(f))
    rt = json.loads(Tokenizer.from_file(f).to_str())
    fixtures[name] = {"pre_tokenizer": [raw["pre_tokenizer"], rt["pre_tokenizer"]], "decoder": [raw["decoder"], rt["decoder"]]}
out = {"pre_tokenizers": pretok, "decoders": dec, "fixtures": fixtures}
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth.json"), "w").write(json.dumps(out, ensure_ascii=False, indent=0))
print("ok", len(pretok), len(dec), len(fixtures))
