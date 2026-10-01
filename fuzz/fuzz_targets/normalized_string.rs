//! Arbitrary sequences of `NormalizedString` operations: alignments must
//! always map normalized text back into the original string.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use morpheme::{NormalizedString, OffsetRange, SplitDelimiterBehavior};

#[derive(Arbitrary, Debug)]
enum Behavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

impl From<&Behavior> for SplitDelimiterBehavior {
    fn from(b: &Behavior) -> Self {
        match b {
            Behavior::Removed => SplitDelimiterBehavior::Removed,
            Behavior::Isolated => SplitDelimiterBehavior::Isolated,
            Behavior::MergedWithPrevious => SplitDelimiterBehavior::MergedWithPrevious,
            Behavior::MergedWithNext => SplitDelimiterBehavior::MergedWithNext,
            Behavior::Contiguous => SplitDelimiterBehavior::Contiguous,
        }
    }
}

#[derive(Arbitrary, Debug)]
enum Op {
    Nfd,
    Nfkd,
    Nfc,
    Nfkc,
    Lowercase,
    Uppercase,
    ReplaceChar(char, String),
    ReplaceStr(String, String),
    Prepend(String),
    Append(String),
    Strip,
    Lstrip,
    Rstrip,
    FilterOut(char),
    Clear,
    Slice {
        normalized: bool,
        start: u16,
        end: u16,
    },
    Split {
        on: char,
        behavior: Behavior,
        pick: u8,
    },
}

#[derive(Arbitrary, Debug)]
struct Input {
    text: String,
    ops: Vec<Op>,
}

/// Work budget for one input (bytes of normalized text).
const MAX_LEN: usize = 4096;

fn check(n: &NormalizedString) {
    // `convert_offsets` is a linear scan, so checking every char is
    // quadratic: skip the check for results far over the budget.
    if n.len() > MAX_LEN * 4 {
        return;
    }
    let original = n.original();
    let normalized = n.get();
    let (start, end) = n.offsets_original();
    assert!(start <= end, "offsets_original reversed");
    for (b, c) in normalized.char_indices() {
        let range = b..b + c.len_utf8();
        let mapped = n
            .convert_offsets(OffsetRange::Normalized(range.clone()))
            .unwrap_or_else(|| panic!("no mapping for normalized {range:?} in {normalized:?}"));
        assert!(
            original.get(mapped.clone()).is_some(),
            "normalized {range:?} ({c:?}) maps to {mapped:?}, not a valid slice of {original:?}"
        );
    }
    // Mapping original ranges into normalized text must stay in bounds too.
    for (b, c) in original.char_indices() {
        if let Some(r) = n.convert_offsets(OffsetRange::Original(b..b + c.len_utf8())) {
            assert!(
                r.start <= r.end && r.end <= normalized.len(),
                "original→normalized {r:?} out of bounds"
            );
        }
    }
}

fuzz_target!(|input: Input| {
    let mut n = NormalizedString::from(input.text.as_str());
    check(&n);
    for op in input.ops.iter().take(32) {
        // Chained replacements grow the string exponentially; each op is
        // linear, so stop once the text is big enough to be interesting.
        if n.len() > MAX_LEN {
            break;
        }
        match op {
            Op::Nfd => {
                n.nfd();
            }
            Op::Nfkd => {
                n.nfkd();
            }
            Op::Nfc => {
                n.nfc();
            }
            Op::Nfkc => {
                n.nfkc();
            }
            Op::Lowercase => {
                n.lowercase();
            }
            Op::Uppercase => {
                n.uppercase();
            }
            Op::ReplaceChar(c, with) => n.replace(*c, with).expect("char replace cannot fail"),
            Op::ReplaceStr(pat, with) => n
                .replace(pat.as_str(), with)
                .expect("str replace cannot fail"),
            Op::Prepend(s) => {
                n.prepend(s);
            }
            Op::Append(s) => {
                n.append(s);
            }
            Op::Strip => {
                n.strip();
            }
            Op::Lstrip => {
                n.lstrip();
            }
            Op::Rstrip => {
                n.rstrip();
            }
            Op::FilterOut(c) => {
                let c = *c;
                n.filter(move |x| x != c);
            }
            Op::Clear => {
                n.clear();
            }
            Op::Slice {
                normalized,
                start,
                end,
            } => {
                let (start, end) = (*start as usize, *end as usize);
                let sliced = if *normalized {
                    n.slice(OffsetRange::Normalized(start..end))
                } else {
                    n.slice(OffsetRange::Original(start..end))
                };
                if let Some(s) = sliced {
                    n = s;
                }
            }
            Op::Split { on, behavior, pick } => {
                let pieces = n
                    .split(*on, behavior.into())
                    .expect("char split cannot fail");
                for p in &pieces {
                    check(p);
                }
                if !pieces.is_empty() {
                    n = pieces[*pick as usize % pieces.len()].clone();
                }
            }
        }
        check(&n);
    }
});
