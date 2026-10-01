//! Tiny hand-crafted BPE fixture.
//!
//! Vocabulary is the 26 lowercase letters plus the end-of-word suffix
//! `</w>`, plus a handful of whole-word merges. This is *not* a real
//! tokenizer — it only exists to exercise the encode / decode path
//! end-to-end and to anchor round-trip property tests.
//!
//! Vocab id 0 is reserved for `<unk>`.

use splinter::{Bpe, Tokenizer, Vocab};

/// End-of-word marker. GPT-2 style.
pub const EOW: &str = "</w>";

/// Build the test tokenizer.
pub fn tokenizer() -> Tokenizer {
    let mut tokens: Vec<String> = Vec::new();
    tokens.push("<unk>".into());
    for c in b'a'..=b'z' {
        tokens.push((c as char).to_string());
    }
    for c in b'a'..=b'z' {
        let mut s = String::new();
        s.push(c as char);
        s.push_str(EOW);
        tokens.push(s);
    }
    tokens.push("aa</w>".into());
    tokens.push("bb</w>".into());
    tokens.push("he</w>".into());
    tokens.push("hel</w>".into());
    tokens.push("hell</w>".into());
    tokens.push("hello</w>".into());

    let vocab = Vocab::from_tokens(tokens).expect("vocab is unique");

    let merges = vec![
        ("a</w>".to_string(), "a</w>".to_string()),
        ("b</w>".to_string(), "b</w>".to_string()),
        ("h</w>".to_string(), "e</w>".to_string()),
        ("he</w>".to_string(), "l</w>".to_string()),
        ("hel</w>".to_string(), "l</w>".to_string()),
        ("hell</w>".to_string(), "o</w>".to_string()),
    ];

    Tokenizer::new(Bpe::new(vocab, merges, EOW))
}
