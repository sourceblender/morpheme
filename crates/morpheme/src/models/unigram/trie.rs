//! A byte trie used for common-prefix search over the vocabulary.

use rustc_hash::FxHashMap;

/// A byte-level trie mapping keys to vocabulary ids.
///
/// Edges live in a single hash map keyed by `(node, byte)`, which keeps
/// the structure compact for large vocabularies.
#[derive(Debug, Clone, Default)]
pub(crate) struct Trie {
    /// `(parent node, byte) -> child node`. Node `0` is the root.
    edges: FxHashMap<(u32, u8), u32>,
    /// For each node, the id stored at that node (if a key ends there).
    values: Vec<Option<u32>>,
}

impl Trie {
    pub(crate) fn new() -> Self {
        Self {
            edges: FxHashMap::default(),
            values: vec![None],
        }
    }

    /// Insert `key` with `value`. Re-inserting a key overwrites its value.
    pub(crate) fn insert(&mut self, key: &[u8], value: u32) {
        let mut node = 0u32;
        for &b in key {
            let next = self.values.len() as u32;
            node = *self.edges.entry((node, b)).or_insert_with(|| next);
            if node == next {
                self.values.push(None);
            }
        }
        self.values[node as usize] = Some(value);
    }

    /// Call `f(len, value)` for every key that is a prefix of `bytes`,
    /// shortest first.
    #[inline]
    pub(crate) fn common_prefix_search(&self, bytes: &[u8], mut f: impl FnMut(usize, u32)) {
        let mut node = 0u32;
        for (i, &b) in bytes.iter().enumerate() {
            match self.edges.get(&(node, b)) {
                Some(&next) => node = next,
                None => return,
            }
            if let Some(v) = self.values[node as usize] {
                f(i + 1, v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_search_finds_all_prefixes() {
        let mut t = Trie::new();
        t.insert(b"a", 0);
        t.insert(b"ab", 1);
        t.insert(b"abc", 2);
        t.insert(b"b", 3);
        t.insert(b"ab", 4);
        let mut found = vec![];
        t.common_prefix_search(b"abd", |len, v| found.push((len, v)));
        assert_eq!(found, vec![(1, 0), (2, 4)]);
    }
}
