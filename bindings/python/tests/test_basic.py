from pathlib import Path

import pytest

import morpheme

FIXTURES = Path(__file__).resolve().parents[3] / "crates" / "morpheme" / "tests" / "data" / "hf"
TEXTS = ["Hello world!", "The quick brown fox jumps over the lazy dog.", "", "naïve café"]


def _fixture(name):
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"fixture {path} missing; run scripts/fetch-hf-fixtures.sh")
    return path


@pytest.fixture(params=["bert-base-uncased.json", "gpt2.json"])
def tokenizer(request):
    return morpheme.Tokenizer.from_file(str(_fixture(request.param)))


def test_encode_fields(tokenizer):
    enc = tokenizer.encode("Hello world!")
    n = len(enc.ids)
    assert n == len(enc) > 0
    assert len(enc.tokens) == len(enc.offsets) == len(enc.type_ids) == n
    assert len(enc.attention_mask) == len(enc.special_tokens_mask) == n
    assert all(isinstance(o, tuple) and len(o) == 2 for o in enc.offsets)


def test_decode_round_trip(tokenizer):
    text = "the quick brown fox"
    enc = tokenizer.encode(text)
    assert tokenizer.decode(enc.ids) == text
    assert len(tokenizer.decode(enc.ids, skip_special_tokens=False)) >= len(text)


def test_count_matches_ids(tokenizer):
    for text in TEXTS:
        for add in (True, False):
            assert tokenizer.count(text, add_special_tokens=add) == len(
                tokenizer.encode(text, add_special_tokens=add).ids
            )


def test_batch_matches_single(tokenizer):
    batch = tokenizer.encode_batch(TEXTS)
    assert [e.ids for e in batch] == [tokenizer.encode(t).ids for t in TEXTS]
    assert [e.tokens for e in batch] == [tokenizer.encode(t).tokens for t in TEXTS]
    ids = [e.ids for e in batch]
    assert tokenizer.decode_batch(ids) == [tokenizer.decode(i) for i in ids]
    assert tokenizer.encode_batch([]) == []


def test_vocab_lookup(tokenizer):
    enc = tokenizer.encode("hello", add_special_tokens=False)
    tok = enc.tokens[0]
    tid = tokenizer.token_to_id(tok)
    assert tid == enc.ids[0]
    assert tokenizer.id_to_token(tid) == tok
    assert tokenizer.id_to_token(2**31) is None
    assert tokenizer.token_to_id("\u0000definitely-not-a-token\u0000") is None
    assert tokenizer.vocab_size() >= tokenizer.vocab_size(with_added_tokens=False) > 0


def test_to_str_and_save_round_trip(tokenizer, tmp_path):
    again = morpheme.Tokenizer.from_str(tokenizer.to_str())
    assert again.encode("round trip").ids == tokenizer.encode("round trip").ids
    out = tmp_path / "tokenizer.json"
    tokenizer.save(str(out), pretty=True)
    reloaded = morpheme.Tokenizer.from_file(str(out))
    assert reloaded.encode("x y z").ids == tokenizer.encode("x y z").ids


def test_errors():
    assert issubclass(morpheme.MorphemeError, Exception)
    with pytest.raises(morpheme.MorphemeError):
        morpheme.Tokenizer.from_file("/nonexistent/tokenizer.json")
    with pytest.raises(morpheme.MorphemeError):
        morpheme.Tokenizer.from_str("{not json")
