//! Arbitrary JSON for each pipeline component: deserializing must never
//! panic, and a component that loads must run on real input without
//! panicking.
//!
//! Input layout: first byte selects the component kind, the rest is the
//! component's JSON.

#![no_main]

use libfuzzer_sys::fuzz_target;
use splinter::{
    Decoder, DecoderWrapper, Encoding, Model, ModelWrapper, NormalizedString, Normalizer,
    NormalizerWrapper, PostProcessor, PostProcessorWrapper, PreTokenizedString, PreTokenizer,
    PreTokenizerWrapper, Token,
};

const TEXTS: [&str; 5] = [
    "Hello, world!",
    "  ünïcödé\tnaïve 你好 😀 ",
    "",
    "e\u{301}\u{200b}x\u{0}",
    "▁Ġ ## </w> <0x41>",
];

fn sample_encoding(len: usize) -> Encoding {
    let tokens = (0..len)
        .map(|i| Token::new(i as u32, format!("t{i}"), (i, i + 1)))
        .collect();
    Encoding::from_tokens(tokens, 0)
}

fuzz_target!(|data: &[u8]| {
    let Some((&kind, json)) = data.split_first() else {
        return;
    };
    match kind % 5 {
        0 => {
            let Ok(n) = serde_json::from_slice::<NormalizerWrapper>(json) else {
                return;
            };
            for text in TEXTS {
                let mut s = NormalizedString::from(text);
                if n.normalize(&mut s).is_ok() {
                    for (b, c) in s.get().char_indices() {
                        let r = s.convert_offsets(splinter::OffsetRange::Normalized(
                            b..b + c.len_utf8(),
                        ));
                        if let Some(r) = r {
                            assert!(
                                text.get(r).is_some(),
                                "normalizer produced misaligned offsets"
                            );
                        }
                    }
                }
            }
            let _ = serde_json::to_string(&n).expect("loaded normalizer must serialize");
        }
        1 => {
            let Ok(p) = serde_json::from_slice::<PreTokenizerWrapper>(json) else {
                return;
            };
            for text in TEXTS {
                let mut pts = PreTokenizedString::from(text);
                if p.pre_tokenize(&mut pts).is_ok() {
                    for (_, (start, end), _) in pts.get_splits(splinter::OffsetType::Byte) {
                        assert!(
                            text.get(start..end).is_some(),
                            "pre-tokenizer split {start}..{end} invalid"
                        );
                    }
                }
            }
            let _ = serde_json::to_string(&p).expect("loaded pre-tokenizer must serialize");
        }
        2 => {
            let Ok(m) = serde_json::from_slice::<ModelWrapper>(json) else {
                return;
            };
            for text in TEXTS {
                for word in text.split_whitespace().chain([text]) {
                    if let Ok(tokens) = m.tokenize(word) {
                        for t in tokens {
                            assert!(
                                t.offsets.0 <= t.offsets.1 && t.offsets.1 <= word.len(),
                                "model offsets out of bounds"
                            );
                            let _ = m.id_to_token(t.id);
                        }
                    }
                }
            }
            let _ = m.get_vocab_size();
            let json = serde_json::to_string(&m).expect("loaded model must serialize");
            serde_json::from_str::<ModelWrapper>(&json).expect("model's own JSON must reload");
        }
        3 => {
            let Ok(pp) = serde_json::from_slice::<PostProcessorWrapper>(json) else {
                return;
            };
            let _ = pp.added_tokens(false);
            let _ = pp.added_tokens(true);
            for add in [true, false] {
                let _ = pp.process(sample_encoding(3), None, add);
                let _ = pp.process(sample_encoding(2), Some(sample_encoding(4)), add);
                let _ = pp.process(Encoding::default(), Some(Encoding::default()), add);
            }
            let _ = serde_json::to_string(&pp).expect("loaded post-processor must serialize");
        }
        _ => {
            let Ok(d) = serde_json::from_slice::<DecoderWrapper>(json) else {
                return;
            };
            for text in TEXTS {
                let tokens: Vec<String> = text.split_inclusive(' ').map(String::from).collect();
                let _ = d.decode(tokens);
                let chars: Vec<String> = text.chars().map(String::from).collect();
                let _ = d.decode(chars);
            }
            let _ = d.decode(vec![]);
            let _ = serde_json::to_string(&d).expect("loaded decoder must serialize");
        }
    }
});
