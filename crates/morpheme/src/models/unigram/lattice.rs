//! The segmentation lattice used by Viterbi decoding, n-best search and
//! the EM trainer.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// One candidate piece in the lattice.
#[derive(Debug, Clone)]
pub(crate) struct Node {
    /// Vocabulary id (or the BOS/EOS sentinel ids).
    pub id: usize,
    /// Byte position where the piece starts.
    pub pos: usize,
    /// Byte length of the piece.
    pub length: usize,
    /// Score (log probability) of the piece.
    pub score: f64,
    prev: Option<usize>,
    backtrace_score: f64,
}

/// A lattice over a sentence: every vocabulary piece that matches at
/// every position, plus BOS/EOS sentinels.
#[derive(Debug, Clone)]
pub(crate) struct Lattice<'a> {
    sentence: &'a str,
    nodes: Vec<Node>,
    /// For each byte position, the nodes starting there.
    pub begin_nodes: Vec<Vec<usize>>,
    /// For each byte position, the nodes ending there.
    pub end_nodes: Vec<Vec<usize>>,
}

/// `log(exp(x) + exp(y))`, or `y` when `init` is set.
pub(crate) fn log_sum_exp(x: f64, y: f64, init: bool) -> f64 {
    if init {
        return y;
    }
    let (lo, hi) = if x > y { (y, x) } else { (x, y) };
    if hi > lo + 50.0 {
        hi
    } else {
        hi + ((lo - hi).exp() + 1.0).ln()
    }
}

const BOS_NODE: usize = 0;
const EOS_NODE: usize = 1;

impl<'a> Lattice<'a> {
    /// An empty lattice over `sentence` with the given sentinel ids.
    pub(crate) fn new(sentence: &'a str, bos_id: usize, eos_id: usize) -> Self {
        let len = sentence.len();
        let mut begin_nodes = vec![Vec::new(); len + 1];
        let mut end_nodes = vec![Vec::new(); len + 1];
        let sentinel = |id, pos| Node {
            id,
            pos,
            length: 0,
            score: 0.0,
            prev: None,
            backtrace_score: 0.0,
        };
        let nodes = vec![sentinel(bos_id, 0), sentinel(eos_id, len)];
        begin_nodes[len].push(EOS_NODE);
        end_nodes[0].push(BOS_NODE);
        Self {
            sentence,
            nodes,
            begin_nodes,
            end_nodes,
        }
    }

    /// Add a piece of `length` bytes starting at `pos`.
    pub(crate) fn insert(&mut self, pos: usize, length: usize, score: f64, id: usize) {
        let node_id = self.nodes.len();
        self.nodes.push(Node {
            id,
            pos,
            length,
            score,
            prev: None,
            backtrace_score: 0.0,
        });
        self.begin_nodes[pos].push(node_id);
        self.end_nodes[pos + length].push(node_id);
    }

    /// Byte length of the sentence.
    pub(crate) fn len(&self) -> usize {
        self.sentence.len()
    }

    /// The sentence.
    pub(crate) fn sentence(&self) -> &'a str {
        self.sentence
    }

    /// The suffix of the sentence starting at char `n`.
    #[cfg(test)]
    pub(crate) fn surface(&self, n: usize) -> &str {
        match self.sentence.char_indices().nth(n) {
            Some((pos, _)) => &self.sentence[pos..],
            None => "",
        }
    }

    /// Node by lattice index.
    pub(crate) fn node(&self, node_id: usize) -> &Node {
        &self.nodes[node_id]
    }

    /// The BOS sentinel.
    #[cfg(test)]
    pub(crate) fn bos_node(&self) -> &Node {
        &self.nodes[BOS_NODE]
    }

    /// The EOS sentinel.
    #[cfg(test)]
    pub(crate) fn eos_node(&self) -> &Node {
        &self.nodes[EOS_NODE]
    }

    /// The text covered by `node`.
    pub(crate) fn piece(&self, node: &Node) -> &'a str {
        &self.sentence[node.pos..node.pos + node.length]
    }

    /// Best-scoring segmentation (node ids, BOS/EOS excluded). Empty if
    /// some position cannot be reached.
    pub(crate) fn viterbi(&mut self) -> Vec<usize> {
        let len = self.len();
        let mut pos = 0;
        while pos <= len {
            if self.begin_nodes[pos].is_empty() {
                return vec![];
            }
            for bi in 0..self.begin_nodes[pos].len() {
                let r = self.begin_nodes[pos][bi];
                let r_score = self.nodes[r].score;
                let mut best: Option<(usize, f64)> = None;
                for &l in &self.end_nodes[pos] {
                    let score = self.nodes[l].backtrace_score + r_score;
                    if best.is_none_or(|(_, b)| score > b) {
                        best = Some((l, score));
                    }
                }
                match best {
                    Some((l, score)) => {
                        self.nodes[r].prev = Some(l);
                        self.nodes[r].backtrace_score = score;
                    }
                    None => {
                        self.nodes[r].prev = None;
                        return vec![];
                    }
                }
            }
            match self.sentence[pos..].chars().next() {
                Some(c) => pos += c.len_utf8(),
                None => break,
            }
        }

        let mut results = vec![];
        let mut node = match self.nodes[EOS_NODE].prev {
            Some(n) => n,
            None => return vec![],
        };
        while let Some(prev) = self.nodes[node].prev {
            results.push(node);
            node = prev;
        }
        results.reverse();
        results
    }

    /// The pieces of the best segmentation.
    #[cfg(test)]
    pub(crate) fn tokens(&mut self) -> Vec<String> {
        self.viterbi()
            .into_iter()
            .map(|n| self.piece(&self.nodes[n]).to_owned())
            .collect()
    }

    /// The `n` best segmentations, best first (A* search over the
    /// backtrace scores computed by Viterbi).
    pub(crate) fn nbest(&mut self, n: usize) -> Vec<Vec<usize>> {
        match n {
            0 => return vec![],
            1 => return vec![self.viterbi()],
            _ => {}
        }

        struct Hypothesis {
            node: usize,
            next: Option<usize>,
            gx: f64,
        }
        struct Entry {
            fx: f64,
            seq: u64,
            hyp: usize,
        }
        impl PartialEq for Entry {
            fn eq(&self, other: &Self) -> bool {
                self.cmp(other) == Ordering::Equal
            }
        }
        impl Eq for Entry {}
        impl PartialOrd for Entry {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Entry {
            fn cmp(&self, other: &Self) -> Ordering {
                // Max-heap on fx; among equals, the earliest pushed first.
                self.fx
                    .total_cmp(&other.fx)
                    .then_with(|| other.seq.cmp(&self.seq))
            }
        }

        const MAX_AGENDA: usize = 100_000;
        const MIN_AGENDA: usize = 512;

        self.viterbi();
        let mut hyps: Vec<Hypothesis> = Vec::new();
        let mut agenda: BinaryHeap<Entry> = BinaryHeap::new();
        let mut seq = 0u64;
        let eos_score = self.nodes[EOS_NODE].score;
        hyps.push(Hypothesis {
            node: EOS_NODE,
            next: None,
            gx: eos_score,
        });
        agenda.push(Entry {
            fx: eos_score,
            seq,
            hyp: 0,
        });

        let bos_id = self.nodes[BOS_NODE].id;
        let mut results: Vec<Vec<usize>> = Vec::new();
        while let Some(top) = agenda.pop() {
            let top_node = hyps[top.hyp].node;
            if self.nodes[top_node].id == bos_id {
                let mut path = Vec::new();
                let mut cur = hyps[top.hyp].next;
                while let Some(h) = cur {
                    if hyps[h].next.is_none() {
                        break; // EOS
                    }
                    path.push(hyps[h].node);
                    cur = hyps[h].next;
                }
                results.push(path);
                if results.len() == n {
                    return results;
                }
                continue;
            }
            let top_gx = hyps[top.hyp].gx;
            let pos = self.nodes[top_node].pos;
            for &l in &self.end_nodes[pos] {
                let fx = self.nodes[l].backtrace_score + top_gx;
                let gx = self.nodes[l].score + top_gx;
                hyps.push(Hypothesis {
                    node: l,
                    next: Some(top.hyp),
                    gx,
                });
                seq += 1;
                agenda.push(Entry {
                    fx,
                    seq,
                    hyp: hyps.len() - 1,
                });
            }
            if agenda.len() > MAX_AGENDA {
                let keep = MIN_AGENDA.min(n * 10);
                let mut shrunk = BinaryHeap::with_capacity(keep);
                for _ in 0..keep {
                    match agenda.pop() {
                        Some(e) => shrunk.push(e),
                        None => break,
                    }
                }
                agenda = shrunk;
            }
        }
        results
    }

    /// The pieces of the `n` best segmentations.
    #[cfg(test)]
    pub(crate) fn nbest_tokens(&mut self, n: usize) -> Vec<Vec<String>> {
        self.nbest(n)
            .into_iter()
            .map(|path| {
                path.into_iter()
                    .map(|i| self.piece(&self.nodes[i]).to_owned())
                    .collect()
            })
            .collect()
    }

    /// One segmentation sampled from the lattice with probability
    /// proportional to `exp(theta * score)` (forward filtering, backward
    /// sampling, as SentencePiece's `Lattice::Sample`). `uniform` must
    /// return values in `[0, 1)`. Empty if some position cannot be
    /// reached.
    pub(crate) fn sample(&self, theta: f64, uniform: &mut dyn FnMut() -> f64) -> Vec<usize> {
        let len = self.len();
        let n = self.nodes.len();
        let mut alpha = vec![f64::NEG_INFINITY; n];
        alpha[BOS_NODE] = 0.0;
        let boundaries = self
            .sentence
            .char_indices()
            .map(|(pos, _)| pos)
            .chain(std::iter::once(len));
        for pos in boundaries {
            if self.begin_nodes[pos].is_empty() {
                return vec![];
            }
            for &r in &self.begin_nodes[pos] {
                for (k, &l) in self.end_nodes[pos].iter().enumerate() {
                    alpha[r] =
                        log_sum_exp(alpha[r], theta * self.nodes[l].score + alpha[l], k == 0);
                }
            }
        }
        if !alpha[EOS_NODE].is_finite() {
            return vec![];
        }

        let mut results = Vec::new();
        let mut node = EOS_NODE;
        let mut probs: Vec<f64> = Vec::new();
        while node != BOS_NODE {
            let pos = self.nodes[node].pos;
            let candidates = &self.end_nodes[pos];
            if candidates.is_empty() {
                return vec![];
            }
            let mut z = 0.0;
            for (k, &l) in candidates.iter().enumerate() {
                z = log_sum_exp(z, alpha[l] + theta * self.nodes[l].score, k == 0);
            }
            probs.clear();
            probs.extend(
                candidates
                    .iter()
                    .map(|&l| (alpha[l] + theta * self.nodes[l].score - z).exp()),
            );
            let mut target = uniform() * probs.iter().sum::<f64>();
            let mut chosen = candidates.len() - 1;
            for (k, p) in probs.iter().enumerate() {
                if target < *p {
                    chosen = k;
                    break;
                }
                target -= p;
            }
            node = candidates[chosen];
            if node != BOS_NODE {
                results.push(node);
            }
        }
        results.reverse();
        results
    }

    /// Forward-backward: add `freq * P(node)` to `expected[node.id]` for
    /// every piece node, and return `freq * log Z`.
    ///
    /// `expected` must be indexed by vocabulary id and large enough for
    /// every piece id in the lattice.
    pub(crate) fn populate_marginal(&self, freq: f64, expected: &mut [f64]) -> f64 {
        let len = self.len();
        let n = self.nodes.len();
        let mut alpha = vec![0.0; n];
        let mut beta = vec![0.0; n];
        for pos in 0..=len {
            for &r in &self.begin_nodes[pos] {
                for (k, &l) in self.end_nodes[pos].iter().enumerate() {
                    alpha[r] = log_sum_exp(alpha[r], self.nodes[l].score + alpha[l], k == 0);
                }
            }
        }
        for pos in (0..=len).rev() {
            for &l in &self.end_nodes[pos] {
                for (k, &r) in self.begin_nodes[pos].iter().enumerate() {
                    beta[l] = log_sum_exp(beta[l], self.nodes[r].score + beta[r], k == 0);
                }
            }
        }
        let z = alpha[EOS_NODE];
        for pos in 0..len {
            for &i in &self.begin_nodes[pos] {
                let node = &self.nodes[i];
                let total = alpha[i] + node.score + beta[i] - z;
                if let Some(slot) = expected.get_mut(node.id) {
                    *slot += freq * total.exp();
                }
            }
        }
        freq * z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-3, "{a} != {b}");
    }

    #[test]
    fn set_sentence() {
        let l = Lattice::new("", 1, 2);
        assert_eq!(l.len(), 0);
        let l = Lattice::new("test", 1, 2);
        assert_eq!(l.surface(1), "est");
        assert_eq!(l.surface(3), "t");
        assert_eq!(l.bos_node().id, 1);
        assert_eq!(l.eos_node().id, 2);
        let l = Lattice::new("テストab", 1, 2);
        assert_eq!(l.len(), 11);
        assert_eq!(l.surface(1), "ストab");
        assert_eq!(l.surface(4), "b");
    }

    #[test]
    fn insert_nodes() {
        let mut l = Lattice::new("ABあい", 1, 2);
        l.insert(0, 1, 0.0, 3);
        l.insert(1, 1, 0.0, 4);
        l.insert(2, 3, 0.0, 5);
        l.insert(5, 3, 0.0, 6);
        l.insert(0, 2, 0.0, 7);
        l.insert(1, 4, 0.0, 8);
        l.insert(2, 6, 0.0, 9);
        let pieces: Vec<&str> = (2..9).map(|i| l.piece(l.node(i))).collect();
        assert_eq!(pieces, ["A", "B", "あ", "い", "AB", "Bあ", "あい"]);
        assert_eq!(l.begin_nodes[0].len(), 2);
        assert_eq!(l.begin_nodes[5].len(), 1);
        assert_eq!(l.begin_nodes[8].len(), 1);
        assert_eq!(l.end_nodes[2].len(), 2);
        assert_eq!(l.end_nodes[8].len(), 2);
        assert_eq!(l.node(l.end_nodes[8][1]).id, 9);
    }

    #[test]
    fn viterbi_incomplete_and_complete() {
        let mut l = Lattice::new("ABC", 1, 2);
        assert!(l.viterbi().is_empty());
        l.insert(0, 1, 0.0, 3);
        assert!(l.viterbi().is_empty());
        l.insert(1, 1, 0.0, 4);
        l.insert(2, 1, 0.0, 5);
        assert_eq!(l.viterbi().len(), 3);
    }

    #[test]
    fn viterbi_prefers_high_scores() {
        let mut l = Lattice::new("ABC", 1, 2);
        l.insert(0, 1, 0.0, 3);
        l.insert(1, 1, 0.0, 4);
        l.insert(2, 1, 0.0, 5);
        assert_eq!(l.tokens(), ["A", "B", "C"]);
        l.insert(0, 2, 2.0, 6);
        assert_eq!(l.tokens(), ["AB", "C"]);
        l.insert(1, 2, 5.0, 7);
        assert_eq!(l.tokens(), ["A", "BC"]);
        l.insert(0, 3, 10.0, 8);
        assert_eq!(l.tokens(), ["ABC"]);
    }

    #[test]
    fn nbest_orders_by_score() {
        let mut l = Lattice::new("ABC", 1, 2);
        l.insert(0, 1, 0.0, 3);
        l.insert(1, 1, 0.0, 4);
        l.insert(2, 1, 0.0, 5);
        l.insert(0, 2, 2.0, 6);
        l.insert(1, 2, 5.0, 7);
        l.insert(0, 3, 10.0, 8);
        assert_eq!(
            l.nbest_tokens(10),
            vec![
                vec!["ABC"],
                vec!["A", "BC"],
                vec!["AB", "C"],
                vec!["A", "B", "C"]
            ]
        );
        assert!(l.nbest_tokens(0).is_empty());
        assert_eq!(l.nbest_tokens(1), vec![vec!["ABC"]]);
    }

    #[test]
    fn log_sum_exp_matches_direct() {
        let mut x = 0.0;
        for (i, y) in [1.0, 2.0, 3.0].iter().enumerate() {
            x = log_sum_exp(x, *y, i == 0);
        }
        approx(x, (1f64.exp() + 2f64.exp() + 3f64.exp()).ln());
    }

    #[test]
    fn sample_matches_lattice_probabilities() {
        let mut l = Lattice::new("ABC", 1, 2);
        l.insert(0, 1, 1.0, 3);
        l.insert(1, 1, 1.2, 4);
        l.insert(2, 1, 2.5, 5);
        l.insert(0, 2, 3.0, 6);
        l.insert(1, 2, 4.0, 7);
        l.insert(0, 3, 2.0, 8);
        // A simple LCG; only the distribution matters here.
        let mut state = 12345u64;
        let mut uniform = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut counts = std::collections::HashMap::new();
        let n = 20_000;
        for _ in 0..n {
            let path = l.sample(1.0, &mut uniform);
            let ids: Vec<usize> = path.iter().map(|&i| l.node(i).id).collect();
            *counts.entry(ids).or_insert(0usize) += 1;
        }
        let p1 = (1.0f64 + 1.2 + 2.5).exp();
        let p2 = (3.0f64 + 2.5).exp();
        let p3 = (1.0f64 + 4.0).exp();
        let p4 = 2.0f64.exp();
        let z = p1 + p2 + p3 + p4;
        let freq = |ids: &[usize]| counts.get(ids).copied().unwrap_or(0) as f64 / n as f64;
        assert!((freq(&[3, 4, 5]) - p1 / z).abs() < 0.02);
        assert!((freq(&[6, 5]) - p2 / z).abs() < 0.02);
        assert!((freq(&[3, 7]) - p3 / z).abs() < 0.02);
        assert!((freq(&[8]) - p4 / z).abs() < 0.02);
    }

    #[test]
    fn sample_incomplete_lattice_is_empty() {
        let l = Lattice::new("ABC", 1, 2);
        let mut uniform = || 0.5;
        assert!(l.sample(1.0, &mut uniform).is_empty());
    }

    #[test]
    fn marginals() {
        let mut l = Lattice::new("ABC", 1, 2);
        l.insert(0, 1, 1.0, 3);
        l.insert(1, 1, 1.2, 4);
        l.insert(2, 1, 2.5, 5);
        l.insert(0, 2, 3.0, 6);
        l.insert(1, 2, 4.0, 7);
        l.insert(0, 3, 2.0, 8);
        let mut probs = vec![0.0; 9];
        let p1 = (1.0f64 + 1.2 + 2.5).exp();
        let p2 = (3.0f64 + 2.5).exp();
        let p3 = (1.0f64 + 4.0).exp();
        let p4 = 2.0f64.exp();
        let z = p1 + p2 + p3 + p4;
        let log_z = l.populate_marginal(1.0, &mut probs);
        approx(log_z, z.ln());
        approx(probs[0], 0.0);
        approx(probs[3], (p1 + p3) / z);
        approx(probs[4], p1 / z);
        approx(probs[5], (p1 + p2) / z);
        approx(probs[6], p2 / z);
        approx(probs[7], p3 / z);
        approx(probs[8], p4 / z);
    }
}
