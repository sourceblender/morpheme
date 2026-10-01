//! Minimal Python bindings for `morpheme` (PyO3 + maturin).
//!
//! The surface is deliberately bounded: load a tokenizer, encode, decode,
//! count, and look up the vocabulary. It does not reproduce the Hugging Face
//! `tokenizers` Python API.

use std::path::PathBuf;

use morpheme::tokenizer::FromPretrainedParameters;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(
    morpheme,
    MorphemeError,
    PyException,
    "Error raised by morpheme."
);

fn to_py(err: morpheme::Error) -> PyErr {
    MorphemeError::new_err(err.to_string())
}

/// The result of encoding one input.
#[pyclass(name = "Encoding", module = "morpheme", frozen)]
pub struct PyEncoding {
    inner: morpheme::Encoding,
}

#[pymethods]
impl PyEncoding {
    /// Token ids.
    #[getter]
    fn ids(&self) -> Vec<u32> {
        self.inner.ids().to_vec()
    }

    /// Token strings.
    #[getter]
    fn tokens(&self) -> Vec<String> {
        self.inner.tokens().to_vec()
    }

    /// Character offsets `(start, end)` of each token in the input, like the
    /// Hugging Face Python package: `text[start:end]` is the token's span.
    #[getter]
    fn offsets(&self) -> Vec<(usize, usize)> {
        self.inner.offsets().to_vec()
    }

    /// Segment (type) ids.
    #[getter]
    fn type_ids(&self) -> Vec<u32> {
        self.inner.type_ids().to_vec()
    }

    /// Attention mask.
    #[getter]
    fn attention_mask(&self) -> Vec<u32> {
        self.inner.attention_mask().to_vec()
    }

    /// Special-tokens mask (1 for special tokens).
    #[getter]
    fn special_tokens_mask(&self) -> Vec<u32> {
        self.inner.special_tokens_mask().to_vec()
    }

    fn __len__(&self) -> usize {
        self.inner.ids().len()
    }

    fn __repr__(&self) -> String {
        format!("Encoding(num_tokens={})", self.inner.ids().len())
    }
}

/// A `tokenizer.json`-compatible tokenizer.
#[pyclass(name = "Tokenizer", module = "morpheme", frozen)]
pub struct PyTokenizer {
    inner: morpheme::Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    #[staticmethod]
    fn from_file(path: PathBuf) -> PyResult<Self> {
        let inner = morpheme::Tokenizer::from_file(path).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Parse a tokenizer from a `tokenizer.json` string.
    #[staticmethod]
    fn from_str(json: &str) -> PyResult<Self> {
        let inner = morpheme::Tokenizer::from_json(json).map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Download (and cache) `tokenizer.json` from the Hugging Face Hub.
    #[staticmethod]
    #[pyo3(signature = (repo_id, revision=None))]
    fn from_pretrained(py: Python<'_>, repo_id: &str, revision: Option<&str>) -> PyResult<Self> {
        let params = revision.map(|r| FromPretrainedParameters::default().revision(r));
        let inner = py
            .detach(|| morpheme::Tokenizer::from_pretrained(repo_id, params))
            .map_err(to_py)?;
        Ok(Self { inner })
    }

    /// Encode one text (offsets are in characters).
    #[pyo3(signature = (text, add_special_tokens=true))]
    fn encode(&self, text: &str, add_special_tokens: bool) -> PyResult<PyEncoding> {
        let inner = self
            .inner
            .encode_char_offsets(text, add_special_tokens)
            .map_err(to_py)?;
        Ok(PyEncoding { inner })
    }

    /// Encode a list of texts (offsets are in characters). Releases the GIL
    /// while encoding.
    #[pyo3(signature = (texts, add_special_tokens=true))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> PyResult<Vec<PyEncoding>> {
        let encodings = py
            .detach(|| {
                let inputs: Vec<&str> = texts.iter().map(String::as_str).collect();
                self.inner
                    .encode_batch_char_offsets(inputs, add_special_tokens)
            })
            .map_err(to_py)?;
        Ok(encodings
            .into_iter()
            .map(|inner| PyEncoding { inner })
            .collect())
    }

    /// Decode a list of ids back to text.
    #[pyo3(signature = (ids, skip_special_tokens=true))]
    fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        self.inner.decode(&ids, skip_special_tokens).map_err(to_py)
    }

    /// Decode several id lists. Releases the GIL while decoding.
    #[pyo3(signature = (sequences, skip_special_tokens=true))]
    fn decode_batch(
        &self,
        py: Python<'_>,
        sequences: Vec<Vec<u32>>,
        skip_special_tokens: bool,
    ) -> PyResult<Vec<String>> {
        py.detach(|| {
            let refs: Vec<&[u32]> = sequences.iter().map(Vec::as_slice).collect();
            self.inner.decode_batch(&refs, skip_special_tokens)
        })
        .map_err(to_py)
    }

    /// Number of tokens `text` encodes to.
    #[pyo3(signature = (text, add_special_tokens=true))]
    fn count(&self, text: &str, add_special_tokens: bool) -> PyResult<usize> {
        Ok(self
            .inner
            .encode(text, add_special_tokens)
            .map_err(to_py)?
            .ids()
            .len())
    }

    /// Id of `token`, or `None` if it is not in the vocabulary.
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Token string for `id`, or `None` if out of range.
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Vocabulary size, optionally including added tokens.
    #[pyo3(signature = (with_added_tokens=true))]
    fn vocab_size(&self, with_added_tokens: bool) -> usize {
        self.inner.vocab_size(with_added_tokens)
    }

    /// Serialize to a `tokenizer.json` string.
    #[pyo3(signature = (pretty=false))]
    fn to_str(&self, pretty: bool) -> PyResult<String> {
        self.inner.to_json(pretty).map_err(to_py)
    }

    /// Write `tokenizer.json` to `path`.
    #[pyo3(signature = (path, pretty=false))]
    fn save(&self, path: PathBuf, pretty: bool) -> PyResult<()> {
        self.inner.save(path, pretty).map_err(to_py)
    }

    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.inner.vocab_size(true))
    }
}

/// Python module entry point.
#[pymodule(name = "morpheme")]
fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyEncoding>()?;
    m.add("MorphemeError", m.py().get_type::<MorphemeError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
