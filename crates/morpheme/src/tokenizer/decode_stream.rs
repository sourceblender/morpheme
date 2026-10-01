//! Incremental decoding for token-by-token generation.

use super::Tokenizer;
use crate::error::{Error, Result};

/// Decodes ids one at a time, yielding text as soon as it is complete.
///
/// Decoding each id on its own breaks text that spans several tokens: a
/// UTF-8 character split across byte-fallback tokens, or decoders that
/// strip a leading space only at the start of a sequence. `DecodeStream`
/// keeps just enough context to emit exactly the new text each step, so
/// concatenating every returned chunk equals [`Tokenizer::decode`] of all
/// ids. Semantics match Hugging Face `tokenizers`' `DecodeStream`.
///
/// ```no_run
/// # use morpheme::Tokenizer;
/// let tokenizer = Tokenizer::from_file("tokenizer.json")?;
/// let mut stream = tokenizer.decode_stream(true);
/// for id in tokenizer.encode("Hello, streaming world! 😀", false)?.ids() {
///     if let Some(chunk) = stream.step(*id)? {
///         print!("{chunk}");
///     }
/// }
/// # Ok::<(), morpheme::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct DecodeStream<'tok> {
    tokenizer: &'tok Tokenizer,
    skip_special_tokens: bool,
    /// Ids still needed to decode the next chunk: the ones behind
    /// `prefix` plus any not yet emitted.
    ids: Vec<u32>,
    /// Text of `ids[..prefix_index]`, already emitted; trimmed off the
    /// next decode.
    prefix: String,
    prefix_index: usize,
    /// `prefill` was called and its prefix is not computed yet.
    pending_prefill: bool,
}

impl<'tok> DecodeStream<'tok> {
    pub(crate) fn new(tokenizer: &'tok Tokenizer, skip_special_tokens: bool) -> Self {
        Self {
            tokenizer,
            skip_special_tokens,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            pending_prefill: false,
        }
    }

    /// Start from ids whose text was already shown (e.g. the prompt), so
    /// the next chunk is decoded in their context but not re-emitted.
    /// The prefill may end inside a multi-token character; that character
    /// is emitted by the step that completes it.
    ///
    /// Like HF, incomplete characters are recognised by the trailing
    /// U+FFFD their bytes decode to. A real U+FFFD at the end of the
    /// prefill is told apart when it comes from several byte tokens
    /// (removing one of them would *add* replacement characters), but a
    /// single id whose text is just U+FFFD is indistinguishable from an
    /// incomplete byte and is emitted again with the next chunk.
    #[must_use]
    pub fn prefill(mut self, ids: &[u32]) -> Self {
        self.ids.extend_from_slice(ids);
        self.pending_prefill = !self.ids.is_empty();
        self
    }

    /// Add the next id. Returns the newly completed text, or `None` if the
    /// id does not complete any text yet (e.g. half of a UTF-8 character).
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        self.step_many(&[id])
    }

    /// Add several ids at once; returns all text they complete.
    pub fn step_many(&mut self, ids: &[u32]) -> Result<Option<String>> {
        let (tokenizer, skip) = (self.tokenizer, self.skip_special_tokens);
        let decode = |ids: &[u32]| tokenizer.decode(ids, skip);

        // First step after `prefill`: the prefilled text was already shown.
        // If the prefill ends inside a character (byte fallback), only the
        // ids up to the last complete character count as shown; the rest
        // stay as context so the character is emitted once completed. A
        // character is at most 4 bytes, so at most 3 trailing byte tokens
        // can be incomplete; anything longer is real U+FFFD text.
        //
        // Walking back must not split a *complete* character that happens
        // to be U+FFFD (bytes EF BF BD): removing one of its bytes turns
        // the remaining ones into more replacement characters, whereas
        // removing bytes of an incomplete character never adds any (a
        // byte-fallback run is replaced as a whole, a byte-level one per
        // maximal invalid sequence, so the texts themselves are not
        // compared). A single id whose text is U+FFFD stays ambiguous.
        if self.pending_prefill {
            self.pending_prefill = false;
            let len = self.ids.len();
            let full = decode(&self.ids)?;
            let trailing_fffd = |s: &str| s.chars().rev().take_while(|&c| c == '\u{FFFD}').count();
            let (mut k, mut prefix) = (len, full.clone());
            let mut fffd = trailing_fffd(&full);
            while fffd > 0 && k > 0 && k + 3 > len {
                let shorter = decode(&self.ids[..k - 1])?;
                let shorter_fffd = trailing_fffd(&shorter);
                if shorter_fffd > fffd {
                    // Removing this id split a complete character.
                    break;
                }
                (k, prefix, fffd) = (k - 1, shorter, shorter_fffd);
            }
            if fffd > 0 {
                // Not an incomplete character: real U+FFFD text.
                (k, prefix) = (len, full);
            }
            self.prefix = prefix;
            self.prefix_index = k;
        }

        self.ids.extend_from_slice(ids);
        let text = decode(&self.ids)?;
        if text.len() <= self.prefix.len() || text.ends_with('\u{FFFD}') {
            return Ok(None);
        }
        let Some(new_text) = text.strip_prefix(self.prefix.as_str()) else {
            return Err(Error::Decoder(format!(
                "stream decode: text {text:?} for id {} does not start with the \
                 previously decoded {:?}",
                self.ids.last().copied().unwrap_or_default(),
                self.prefix
            )));
        };
        let new_text = new_text.to_owned();
        let new_prefix_index = self.ids.len() - self.prefix_index;
        self.ids.drain(..self.prefix_index);
        self.prefix = decode(&self.ids)?;
        self.prefix_index = new_prefix_index;
        Ok(Some(new_text))
    }
}
