//! A word as a doubly linked list of symbols, with the merge loop used
//! at inference ([`Word::merge_all`]) and the incremental merge used by
//! the trainer ([`Word::merge`]).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rustc_hash::FxHashMap;

use super::Pair;

/// `pair -> (rank, merged id)`.
pub(crate) type MergeMap = FxHashMap<Pair, (u32, u32)>;

/// A pending merge at inference: lowest rank first, then leftmost.
#[derive(Debug, PartialEq, Eq)]
struct PendingMerge {
    pos: usize,
    rank: u32,
    new_id: u32,
}

impl Ord for PendingMerge {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap: invert so the smallest rank (then
        // the smallest position) pops first.
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.pos.cmp(&self.pos))
    }
}

impl PartialOrd for PendingMerge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Symbol {
    id: u32,
    prev: isize,
    next: isize,
    /// Byte length at inference; char count during training. `0`
    /// marks a symbol merged into its left neighbour.
    len: usize,
}

/// A word being merged.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Word {
    symbols: Vec<Symbol>,
}

impl Word {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            symbols: Vec::with_capacity(capacity),
        }
    }

    /// Append a symbol.
    pub(crate) fn add(&mut self, id: u32, len: usize) {
        let n = self.symbols.len() as isize;
        let prev = if let Some(last) = self.symbols.last_mut() {
            last.next = n;
            n - 1
        } else {
            -1
        };
        self.symbols.push(Symbol {
            id,
            prev,
            next: -1,
            len,
        });
    }

    /// Trainer merge: replace every `(a, b)` with `replacement` and
    /// report pair count changes (`-1` for a vanished pair occurrence,
    /// `+1` for a new one). New pairs whose combined length would reach
    /// `max_length` are not reported.
    pub(crate) fn merge(
        &mut self,
        a: u32,
        b: u32,
        replacement: u32,
        max_length: usize,
    ) -> Vec<(Pair, i32)> {
        let mut changes = Vec::new();
        let mut i = 0;
        while i < self.symbols.len() {
            if self.symbols[i].id == a && i + 1 < self.symbols.len() && self.symbols[i + 1].id == b
            {
                let first = self.symbols[i];
                let second = self.symbols[i + 1];
                let merged = Symbol {
                    id: replacement,
                    prev: first.prev,
                    next: second.next,
                    len: first.len + second.len,
                };
                if i > 0 {
                    let left = self.symbols[i - 1];
                    changes.push(((left.id, first.id), -1));
                    if left.len + merged.len < max_length {
                        changes.push(((left.id, replacement), 1));
                    }
                }
                self.symbols.splice(i..i + 2, std::iter::once(merged));
                if i + 1 < self.symbols.len() {
                    let right = self.symbols[i + 1];
                    changes.push(((second.id, right.id), -1));
                    if right.len + merged.len < max_length {
                        changes.push(((replacement, right.id), 1));
                    }
                }
            }
            i += 1;
        }
        changes
    }

    /// Inference merge: apply merges by rank until none applies. With
    /// `dropout`, each candidate merge is skipped with that probability.
    pub(crate) fn merge_all(&mut self, merges: &MergeMap, dropout: Option<f32>) {
        let mut queue: BinaryHeap<PendingMerge> = self
            .symbols
            .windows(2)
            .enumerate()
            .filter_map(|(pos, w)| {
                merges
                    .get(&(w[0].id, w[1].id))
                    .map(|&(rank, new_id)| PendingMerge { pos, rank, new_id })
            })
            .collect();
        let mut skipped: Vec<PendingMerge> = Vec::new();

        while let Some(top) = queue.pop() {
            if dropout.is_some_and(|p| super::prng::next_f32() < p) {
                skipped.push(top);
                continue;
            }
            queue.extend(skipped.drain(..));

            let cur = self.symbols[top.pos];
            if cur.len == 0 || cur.next == -1 {
                continue;
            }
            let next_pos = cur.next as usize;
            let right = self.symbols[next_pos];
            // The queued merge may be stale.
            match merges.get(&(cur.id, right.id)) {
                Some(&(_, id)) if id == top.new_id => {}
                _ => continue,
            }

            let sym = &mut self.symbols[top.pos];
            sym.id = top.new_id;
            sym.len += right.len;
            sym.next = right.next;
            self.symbols[next_pos].len = 0;
            if right.next > -1 && (right.next as usize) < self.symbols.len() {
                self.symbols[right.next as usize].prev = top.pos as isize;
            }

            let cur = self.symbols[top.pos];
            if cur.prev >= 0 {
                let prev = self.symbols[cur.prev as usize];
                if let Some(&(rank, new_id)) = merges.get(&(prev.id, cur.id)) {
                    queue.push(PendingMerge {
                        pos: cur.prev as usize,
                        rank,
                        new_id,
                    });
                }
            }
            if cur.next >= 0 && (cur.next as usize) < self.symbols.len() {
                let next = self.symbols[cur.next as usize];
                if let Some(&(rank, new_id)) = merges.get(&(cur.id, next.id)) {
                    queue.push(PendingMerge {
                        pos: top.pos,
                        rank,
                        new_id,
                    });
                }
            }
        }
        self.symbols.retain(|s| s.len != 0);
    }

    pub(crate) fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.symbols.iter().map(|s| s.id)
    }

    pub(crate) fn id_vec(&self) -> Vec<u32> {
        self.ids().collect()
    }

    /// Byte offsets of each symbol within the word.
    pub(crate) fn offsets(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let mut pos = 0;
        self.symbols.iter().map(move |s| {
            let start = pos;
            pos += s.len;
            (start, pos)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trainer_merge_reports_changes() {
        // "hello" with ids h=0 e=1 l=2 o=3; merge (l, l) -> 4.
        let mut w = Word::default();
        for id in [0, 1, 2, 2, 3] {
            w.add(id, 1);
        }
        let changes = w.merge(2, 2, 4, usize::MAX);
        assert_eq!(w.id_vec(), vec![0, 1, 4, 3]);
        assert_eq!(
            changes,
            vec![((1, 2), -1), ((1, 4), 1), ((2, 3), -1), ((4, 3), 1)]
        );
    }

    #[test]
    fn trainer_merge_overlapping_pairs() {
        let mut w = Word::default();
        for _ in 0..4 {
            w.add(0, 1);
        }
        w.merge(0, 0, 1, usize::MAX);
        assert_eq!(w.id_vec(), vec![1, 1]);
    }

    #[test]
    fn merge_all_by_rank() {
        // a b c, merges: (b,c)->bc rank 0, (a,b)->ab rank 1.
        let mut merges = MergeMap::default();
        merges.insert((1, 2), (0, 4));
        merges.insert((0, 1), (1, 3));
        let mut w = Word::default();
        for id in [0, 1, 2] {
            w.add(id, 1);
        }
        w.merge_all(&merges, None);
        assert_eq!(w.id_vec(), vec![0, 4]);
        assert_eq!(w.offsets().collect::<Vec<_>>(), vec![(0, 1), (1, 3)]);
    }

    #[test]
    fn dropout_one_disables_merges() {
        let mut merges = MergeMap::default();
        merges.insert((0, 1), (0, 2));
        let mut w = Word::default();
        w.add(0, 1);
        w.add(1, 1);
        w.merge_all(&merges, Some(1.0));
        assert_eq!(w.id_vec(), vec![0, 1]);
    }
}
