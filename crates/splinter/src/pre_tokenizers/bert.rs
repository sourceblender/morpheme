//! BERT pre-tokenizer: split on whitespace, isolate punctuation.

use unicode_categories::UnicodeCategories;

use crate::error::Result;
use crate::normalized_string::SplitDelimiterBehavior;
use crate::pre_tokenized_string::PreTokenizedString;
use crate::traits::PreTokenizer;

/// BERT's definition of punctuation: ASCII punctuation (which includes
/// symbols like `$`, `^`, `` ` ``) plus every Unicode `P*` category.
pub(crate) fn is_bert_punc(c: char) -> bool {
    c.is_ascii_punctuation() || c.is_punctuation()
}

/// Splits on whitespace (removed) and isolates every punctuation char.
///
/// `"Hey friend!  How?!"` → `["Hey", "friend", "!", "How", "?", "!"]`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BertPreTokenizer;

super::impl_unit_serde!(BertPreTokenizer);

impl PreTokenizer for BertPreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(char::is_whitespace, SplitDelimiterBehavior::Removed))?;
        pretokenized.split(|_, s| s.split(is_bert_punc, SplitDelimiterBehavior::Isolated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NormalizedString;
    use crate::pre_tokenizers::tests::splits;

    #[test]
    fn basic() {
        assert_eq!(
            splits(&BertPreTokenizer, "Hey friend!     How are you?!?"),
            vec![
                ("Hey", (0, 3)),
                ("friend", (4, 10)),
                ("!", (10, 11)),
                ("How", (16, 19)),
                ("are", (20, 23)),
                ("you", (24, 27)),
                ("?", (27, 28)),
                ("!", (28, 29)),
                ("?", (29, 30)),
            ]
        );
    }

    #[test]
    fn chinese_chars_padded_by_normalizer() {
        // Simulate BertNormalizer's handle_chinese_chars.
        let mut n = NormalizedString::from("野口里佳 Noguchi Rika");
        let dest: Vec<(char, isize)> = n
            .get()
            .chars()
            .flat_map(|c| {
                if (c as usize) > 0x4E00 {
                    vec![(' ', 0), (c, 1), (' ', 1)]
                } else {
                    vec![(c, 0)]
                }
            })
            .collect();
        n.transform(dest, 0);
        let mut pts = PreTokenizedString::from(n);
        BertPreTokenizer.pre_tokenize(&mut pts).unwrap();
        let got: Vec<_> = pts
            .get_splits(crate::OffsetType::Byte)
            .into_iter()
            .map(|(s, o, _)| (s.to_owned(), o))
            .collect();
        let want: Vec<(String, (usize, usize))> = vec![
            ("野".into(), (0, 3)),
            ("口".into(), (3, 6)),
            ("里".into(), (6, 9)),
            ("佳".into(), (9, 12)),
            ("Noguchi".into(), (13, 20)),
            ("Rika".into(), (21, 25)),
        ];
        assert_eq!(got, want);
    }
}
