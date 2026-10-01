//! Added-token, post-processor and padding ids stay consistent with the
//! model's vocabulary: `set_model` rebinds them (#84) and loading rejects
//! an added token that shadows a different model token (#85).

use morpheme::models::WordLevel;
use morpheme::pre_tokenizers::WhitespaceSplit;
use morpheme::processors::{BertProcessing, TemplateProcessing};
use morpheme::{AddedToken, Error, PaddingParams, PaddingStrategy, Tokenizer};

fn wordlevel(vocab: &[(&str, u32)]) -> WordLevel {
    WordLevel::builder()
        .vocab(vocab.iter().map(|(t, id)| (t.to_string(), *id)).collect())
        .unk_token("[UNK]")
        .build()
        .unwrap()
}

fn template() -> TemplateProcessing {
    TemplateProcessing::builder()
        .try_single("[CLS] $A [SEP]")
        .unwrap()
        .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")
        .unwrap()
        .special_tokens(vec![("[CLS]", 2), ("[SEP]", 3)])
        .build()
        .unwrap()
}

/// `{"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "a": 4}` with a
/// `[CLS]`/`[SEP]` template and `[PAD]` padding to 5.
fn configured() -> Tokenizer {
    let mut tok = Tokenizer::new(wordlevel(&[
        ("[UNK]", 0),
        ("[PAD]", 1),
        ("[CLS]", 2),
        ("[SEP]", 3),
        ("a", 4),
    ]))
    .with_pre_tokenizer(WhitespaceSplit)
    .with_post_processor(template());
    tok.set_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(5),
        pad_id: 1,
        pad_token: "[PAD]".into(),
        ..Default::default()
    }));
    tok
}

// ----- #84: set_model -----------------------------------------------

#[test]
fn set_model_moves_added_tokens_past_the_new_vocabulary() {
    // The exact repro from #84.
    let mut t = Tokenizer::new(wordlevel(&[("[UNK]", 0), ("a", 1)]));
    t.add_special_tokens(&[AddedToken::new("<special>", true)])
        .unwrap();
    assert_eq!(t.token_to_id("<special>"), Some(2));
    t.set_model(wordlevel(&[("[UNK]", 0), ("a", 1), ("new", 2)]))
        .unwrap();

    let e = t.encode("new", false).unwrap();
    assert_eq!(e.ids(), [2]);
    assert_eq!(t.decode(e.ids(), false).unwrap(), "new");
    assert_eq!(t.decode(e.ids(), true).unwrap(), "new");

    assert_eq!(t.token_to_id("<special>"), Some(3));
    assert_eq!(t.id_to_token(3).as_deref(), Some("<special>"));
    let e = t.encode("<special>", false).unwrap();
    assert_eq!(e.ids(), [3]);
    assert_eq!(t.decode(e.ids(), false).unwrap(), "<special>");
    assert_eq!(t.decode(e.ids(), true).unwrap(), "");
    // The added token keeps its options.
    let added = t.added_vocabulary().tokens_with_ids();
    assert_eq!(added.len(), 1);
    assert_eq!(added[0].token, AddedToken::new("<special>", true));
    // And the result is a consistent, loadable tokenizer.
    let reloaded = Tokenizer::from_json(&t.to_json(false).unwrap()).unwrap();
    assert_eq!(reloaded.token_to_id("<special>"), Some(3));
}

#[test]
fn set_model_reuses_the_new_model_id_for_known_added_tokens() {
    let mut t = Tokenizer::new(wordlevel(&[("[UNK]", 0), ("a", 1)]));
    t.add_special_tokens(&[AddedToken::new("<s>", true), AddedToken::new("</s>", true)])
        .unwrap();
    assert_eq!(
        (t.token_to_id("<s>"), t.token_to_id("</s>")),
        (Some(2), Some(3))
    );
    t.set_model(wordlevel(&[("[UNK]", 0), ("</s>", 1), ("a", 2), ("b", 3)]))
        .unwrap();
    assert_eq!(t.token_to_id("</s>"), Some(1));
    assert_eq!(t.token_to_id("<s>"), Some(4));
    assert_eq!(t.id_to_token(2).as_deref(), Some("a"));
    assert_eq!(t.id_to_token(3).as_deref(), Some("b"));
    assert_eq!(t.decode(&[1, 2, 3, 4], true).unwrap(), "a b");
    assert!(t.added_vocabulary().is_special_token("</s>"));
}

#[test]
fn set_model_rebinds_template_and_padding_ids() {
    let mut tok = configured();
    assert_eq!(
        tok.encode("a", true).unwrap().ids(),
        [2, 4, 3, 1, 1],
        "before"
    );
    tok.set_model(wordlevel(&[
        ("[UNK]", 0),
        ("a", 1),
        ("[SEP]", 2),
        ("[CLS]", 3),
        ("[PAD]", 4),
        ("b", 5),
    ]))
    .unwrap();
    let e = tok.encode("a", true).unwrap();
    assert_eq!(e.ids(), [3, 1, 2, 4, 4]);
    assert_eq!(e.tokens(), ["[CLS]", "a", "[SEP]", "[PAD]", "[PAD]"]);
    assert_eq!(tok.padding().unwrap().pad_id, 4);
    let pair = tok.encode(("a", "b"), true).unwrap();
    assert_eq!(pair.ids(), [3, 1, 2, 5, 2]);
}

#[test]
fn set_model_rebinds_template_ids_to_added_tokens() {
    // `[CLS]`/`[SEP]`/`[PAD]` are added special tokens the new model lacks,
    // so they move past the new vocabulary and the template follows them.
    let mut tok = configured();
    tok.add_special_tokens(&[
        AddedToken::new("[PAD]", true),
        AddedToken::new("[CLS]", true),
        AddedToken::new("[SEP]", true),
    ])
    .unwrap();
    tok.set_model(wordlevel(&[("[UNK]", 0), ("a", 1), ("b", 2), ("c", 3)]))
        .unwrap();
    assert_eq!(tok.token_to_id("[PAD]"), Some(4));
    assert_eq!(tok.token_to_id("[CLS]"), Some(5));
    assert_eq!(tok.token_to_id("[SEP]"), Some(6));
    let e = tok.encode("a", true).unwrap();
    assert_eq!(e.ids(), [5, 1, 6, 4, 4]);
    assert_eq!(tok.decode(e.ids(), true).unwrap(), "a");
    assert_eq!(tok.decode(&[2, 3], true).unwrap(), "b c");
}

#[test]
fn set_model_rebinds_bert_processing_ids() {
    let mut tok = Tokenizer::new(wordlevel(&[
        ("[UNK]", 0),
        ("[CLS]", 1),
        ("[SEP]", 2),
        ("a", 3),
    ]))
    .with_post_processor(BertProcessing::new(("[SEP]", 2), ("[CLS]", 1)));
    tok.set_model(wordlevel(&[
        ("[UNK]", 0),
        ("a", 1),
        ("[SEP]", 7),
        ("[CLS]", 8),
    ]))
    .unwrap();
    assert_eq!(tok.encode("a", true).unwrap().ids(), [8, 1, 7]);
}

#[test]
fn set_model_with_a_missing_template_token_leaves_the_tokenizer_unchanged() {
    let mut tok = configured();
    tok.add_tokens(&[AddedToken::new("extra", false)]).unwrap();
    let before = tok.to_json(false).unwrap();
    let err = tok
        .set_model(wordlevel(&[
            ("[UNK]", 0),
            ("[PAD]", 1),
            ("[SEP]", 2),
            ("a", 3),
        ]))
        .unwrap_err();
    assert!(matches!(err, Error::Config(_)), "{err:?}");
    assert!(err.to_string().contains("[CLS]"), "{err}");
    assert_eq!(tok.to_json(false).unwrap(), before);
    assert_eq!(tok.encode("a", true).unwrap().ids(), [2, 4, 3, 1, 1]);
}

#[test]
fn set_model_with_a_missing_pad_token_leaves_the_tokenizer_unchanged() {
    let mut tok = configured();
    let before = tok.to_json(false).unwrap();
    let err = tok
        .set_model(wordlevel(&[
            ("[UNK]", 0),
            ("[CLS]", 1),
            ("[SEP]", 2),
            ("a", 3),
        ]))
        .unwrap_err();
    assert!(err.to_string().contains("[PAD]"), "{err}");
    assert_eq!(tok.to_json(false).unwrap(), before);
}

// ----- #85: loading --------------------------------------------------

/// A WordLevel `tokenizer.json` over `{"[UNK]": 0, "a": 1}` with the given
/// `added_tokens` entries.
fn json_with_added(added: serde_json::Value) -> String {
    let tok = Tokenizer::new(wordlevel(&[("[UNK]", 0), ("a", 1)]));
    let mut value: serde_json::Value = serde_json::from_str(&tok.to_json(false).unwrap()).unwrap();
    value["added_tokens"] = added;
    value.to_string()
}

fn added(id: u32, content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "content": content,
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": true,
    })
}

#[test]
fn loading_an_added_token_that_shadows_a_model_token_fails() {
    // The exact repro from #85: special `x` at id 1, which the model uses
    // for `a`.
    let json = json_with_added(serde_json::json!([added(1, "x")]));
    let err = Tokenizer::from_json(&json).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("\"x\"") && msg.contains("\"a\"") && msg.contains("id 1"),
        "{msg}"
    );
}

#[test]
fn loading_an_added_token_that_shares_an_id_with_a_duplicate_model_token_fails() {
    // A BPE vocabulary may map two tokens to one id; `id_to_token(1)`
    // reports `a`, but `b` encodes to 1 too, so `a` cannot claim it.
    let tok = Tokenizer::new(
        morpheme::models::Bpe::builder()
            .vocab_and_merges(
                [("[UNK]", 0), ("a", 1), ("b", 1)]
                    .into_iter()
                    .map(|(t, id)| (t.to_string(), id))
                    .collect(),
                vec![],
            )
            .build()
            .unwrap(),
    );
    assert_eq!(tok.id_to_token(1).as_deref(), Some("a"));
    let mut value: serde_json::Value = serde_json::from_str(&tok.to_json(false).unwrap()).unwrap();
    value["added_tokens"] = serde_json::json!([added(1, "a")]);
    let err = Tokenizer::from_json(&value.to_string()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("\"a\"") && msg.contains("\"b\"") && msg.contains("id 1"),
        "{msg}"
    );
}

#[test]
fn loading_an_added_token_at_its_own_model_id_still_works() {
    // BERT-style: `[UNK]` registered as a special added token at its model
    // id; `y` above the vocabulary.
    let json = json_with_added(serde_json::json!([added(0, "[UNK]"), added(2, "y")]));
    let tok = Tokenizer::from_json(&json).unwrap();
    assert_eq!(tok.token_to_id("[UNK]"), Some(0));
    assert_eq!(tok.token_to_id("y"), Some(2));
    assert_eq!(tok.encode("a", false).unwrap().ids(), [1]);
    assert_eq!(tok.decode(&[0, 1, 2], true).unwrap(), "a");
}

#[test]
fn runtime_add_tokens_never_shadows_a_model_token() {
    // A vocabulary with a hole: new ids go after the highest model id, and
    // tokens the model knows keep the model's id, so the saved file loads.
    let mut tok = Tokenizer::new(wordlevel(&[("[UNK]", 0), ("a", 5)]));
    tok.add_special_tokens(&[AddedToken::new("x", true)])
        .unwrap();
    tok.add_tokens(&[AddedToken::new("a", false), AddedToken::new("y", false)])
        .unwrap();
    assert_eq!(tok.token_to_id("x"), Some(6));
    assert_eq!(tok.token_to_id("a"), Some(5));
    assert_eq!(tok.token_to_id("y"), Some(7));
    let reloaded = Tokenizer::from_json(&tok.to_json(false).unwrap()).unwrap();
    assert_eq!(
        reloaded.to_json(false).unwrap(),
        tok.to_json(false).unwrap()
    );
}
