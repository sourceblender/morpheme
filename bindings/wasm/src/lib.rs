//! Browser bindings for [`morpheme`] through `wasm-bindgen`.
//!
//! A `Tokenizer` is loaded from the text of a Hugging Face `tokenizer.json`
//! (`Tokenizer.fromJson`) and exposes `count`, `encode`, `tokens` and
//! `decode`. Errors become JavaScript exceptions carrying the Rust error
//! message. Build with `wasm-pack build --target web` (see the README).

use wasm_bindgen::prelude::*;

/// A loaded tokenizer.
#[wasm_bindgen]
pub struct Tokenizer {
    inner: morpheme::Tokenizer,
}

fn js_error(e: morpheme::Error) -> JsValue {
    JsError::new(&e.to_string()).into()
}

#[wasm_bindgen]
impl Tokenizer {
    /// Load from the contents of a `tokenizer.json` file.
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(json: &str) -> Result<Tokenizer, JsValue> {
        let inner = morpheme::Tokenizer::from_json(json).map_err(js_error)?;
        Ok(Tokenizer { inner })
    }

    /// Token ids for `text`, as a `Uint32Array`.
    #[wasm_bindgen]
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, JsValue> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(js_error)?;
        Ok(encoding.ids().to_vec())
    }

    /// Token strings for `text`.
    #[wasm_bindgen]
    pub fn tokens(&self, text: &str, add_special_tokens: bool) -> Result<Vec<String>, JsValue> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(js_error)?;
        Ok(encoding.tokens().to_vec())
    }

    /// Number of tokens in `text`.
    #[wasm_bindgen]
    pub fn count(&self, text: &str, add_special_tokens: bool) -> Result<u32, JsValue> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(js_error)?;
        u32::try_from(encoding.len()).map_err(|_| JsError::new("token count exceeds u32").into())
    }

    /// Text for `ids`.
    #[wasm_bindgen]
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, JsValue> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(js_error)
    }
}
