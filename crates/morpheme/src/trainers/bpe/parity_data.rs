// Generated from Python tokenizers 0.23.2 (see the comment above the tests).
// SUFFIX vocab ids are not stable in HF itself (suffixed symbols get ids in
// hash-map order), so that case compares the vocab as a set.
const PARITY_BASIC_VOCAB: &[(&str, u32)] = &[("[UNK]", 0), ("a", 1), ("d", 2), ("e", 3), ("h", 4), ("l", 5), ("o", 6), ("r", 7), ("s", 8), ("t", 9), ("w", 10), ("aa", 11), ("lo", 12), ("low", 13), ("el", 14), ("hel", 15), ("lowe", 16), ("hello", 17), ("ld", 18), ("or", 19), ("st", 20), ("wor", 21), ("aaa", 22), ("aaaa", 23), ("lower", 24), ("lowest", 25), ("world", 26)];
const PARITY_BASIC_MERGES: &[(&str, &str)] = &[("a", "a"), ("l", "o"), ("lo", "w"), ("e", "l"), ("h", "el"), ("low", "e"), ("hel", "lo"), ("l", "d"), ("o", "r"), ("s", "t"), ("w", "or"), ("aa", "a"), ("aa", "aa"), ("lowe", "r"), ("lowe", "st"), ("wor", "ld")];
const PARITY_UNICODE_VOCAB: &[(&str, u32)] = &[("a", 0), ("b", 1), ("c", 2), ("e", 3), ("f", 4), ("l", 5), ("n", 6), ("r", 7), ("s", 8), ("v", 9), ("é", 10), ("ï", 11), ("ü", 12), ("世", 13), ("你", 14), ("好", 15), ("界", 16), ("😀", 17), ("af", 18), ("be", 19), ("caf", 20), ("übe", 21), ("你好", 22), ("café", 23), ("über", 24)];
const PARITY_UNICODE_MERGES: &[(&str, &str)] = &[("a", "f"), ("b", "e"), ("c", "af"), ("ü", "be"), ("你", "好"), ("caf", "é"), ("übe", "r")];
const PARITY_SUFFIX_VOCAB: &[(&str, u32)] = &[("e", 0), ("h", 1), ("i", 2), ("n", 3), ("r", 4), ("s", 5), ("t", 6), ("e</w>", 7), ("s</w>", 8), ("n</w>", 9), ("th", 10), ("es", 11), ("er", 12), ("en</w>", 13), ("is</w>", 14)];
const PARITY_SUFFIX_MERGES: &[(&str, &str)] = &[("t", "h"), ("e", "s"), ("e", "r"), ("e", "n</w>"), ("i", "s</w>")];
const PARITY_LIMIT_VOCAB: &[(&str, u32)] = &[("a", 0), ("b", 1), ("c", 2), ("z", 3), ("ab", 4), ("abab", 5), ("cc", 6), ("abababab", 7), ("ccc", 8)];
const PARITY_LIMIT_MERGES: &[(&str, &str)] = &[("a", "b"), ("ab", "ab"), ("c", "c"), ("abab", "abab"), ("cc", "c")];
