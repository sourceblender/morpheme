//! Download `tokenizer.json` files from the Hugging Face Hub (feature
//! `hub`).
//!
//! Files are cached in the standard `huggingface_hub` layout, so the
//! cache is shared with Python (`transformers`, `tokenizers`, …):
//!
//! ```text
//! <cache>/models--<org>--<name>/
//!     blobs/<etag>                         file contents
//!     refs/<revision>                      commit hash of a branch or tag
//!     snapshots/<commit>/tokenizer.json    → ../../blobs/<etag>
//! ```
//!
//! Environment variables (same meaning as in `huggingface_hub`):
//!
//! | Variable | Effect |
//! | --- | --- |
//! | `HF_ENDPOINT` | Hub URL (default `https://huggingface.co`) |
//! | `HF_HUB_CACHE` | Cache directory (default `$HF_HOME/hub`) |
//! | `HF_HOME` | Base directory (default `$XDG_CACHE_HOME/huggingface` or `~/.cache/huggingface`) |
//! | `HF_TOKEN` | Access token (otherwise read from `$HF_HOME/token`) |
//! | `HF_HUB_OFFLINE` | `1` to only use the cache |

use sha1::Digest;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::error::{Error, Result};

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
const FILENAME: &str = "tokenizer.json";

/// Options for [`crate::Tokenizer::from_pretrained`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FromPretrainedParameters {
    /// Branch, tag or commit hash. Defaults to `main`.
    pub revision: String,
    /// Extra `key/value` pairs appended to the `User-Agent` header.
    pub user_agent: HashMap<String, String>,
    /// Access token for private or gated repositories. Defaults to
    /// `HF_TOKEN`, then the token saved by `huggingface-cli login`.
    pub token: Option<String>,
    /// Cache directory. Defaults to the shared Hugging Face cache.
    pub cache_dir: Option<PathBuf>,
}

impl Default for FromPretrainedParameters {
    fn default() -> Self {
        Self {
            revision: "main".to_owned(),
            user_agent: HashMap::new(),
            token: None,
            cache_dir: None,
        }
    }
}

impl FromPretrainedParameters {
    /// Set the revision (branch, tag or commit hash).
    #[must_use]
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = revision.into();
        self
    }

    /// Set the access token.
    #[must_use]
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Set the cache directory.
    #[must_use]
    pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    /// Add a `key/value` pair to the `User-Agent` header.
    #[must_use]
    pub fn user_agent_entry(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.user_agent.insert(key.into(), value.into());
        self
    }
}

/// Settings resolved from the environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HubConfig {
    pub endpoint: String,
    pub cache_dir: PathBuf,
    pub token: Option<String>,
    pub offline: bool,
}

impl HubConfig {
    /// Resolve settings from the process environment.
    pub(crate) fn from_env() -> Self {
        Self::from_lookup(|key| std::env::var(key).ok().filter(|v| !v.is_empty()))
    }

    /// Resolve settings from `lookup` (an environment accessor), using
    /// the same precedence as `huggingface_hub`.
    pub(crate) fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Self {
        let home = || {
            lookup("HOME")
                .or_else(|| lookup("USERPROFILE"))
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
        };
        let hf_home = lookup("HF_HOME").map(PathBuf::from).unwrap_or_else(|| {
            lookup("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| home().join(".cache"))
                .join("huggingface")
        });
        let cache_dir = lookup("HF_HUB_CACHE")
            .or_else(|| lookup("HUGGINGFACE_HUB_CACHE"))
            .map(PathBuf::from)
            .unwrap_or_else(|| hf_home.join("hub"));
        let token = lookup("HF_TOKEN")
            .or_else(|| lookup("HUGGING_FACE_HUB_TOKEN"))
            .or_else(|| {
                let path = lookup("HF_TOKEN_PATH")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| hf_home.join("token"));
                std::fs::read_to_string(path).ok()
            })
            .map(|t| t.trim().to_owned())
            .filter(|t| !t.is_empty());
        let offline = lookup("HF_HUB_OFFLINE").is_some_and(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        });
        let endpoint = lookup("HF_ENDPOINT")
            .unwrap_or_else(|| DEFAULT_ENDPOINT.to_owned())
            .trim_end_matches('/')
            .to_owned();
        Self {
            endpoint,
            cache_dir,
            token,
            offline,
        }
    }
}

fn is_valid_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')
}

/// Validate a repository id (`name` or `org/name`).
pub(crate) fn validate_repo_id(id: &str) -> Result<()> {
    let parts: Vec<&str> = id.split('/').collect();
    let ok = (1..=2).contains(&parts.len())
        && parts.iter().all(|p| {
            !p.is_empty()
                && p.len() <= 96
                && p.chars().all(is_valid_char)
                && !p.starts_with(['.', '-'])
                && !p.contains("..")
        });
    if ok {
        Ok(())
    } else {
        Err(Error::Hub(format!(
            "invalid model id {id:?}: expected `name` or `org/name` using letters, digits, '-', '_' and '.'"
        )))
    }
}

/// Validate a revision (branch, tag, commit hash, or `refs/pr/N`).
pub(crate) fn validate_revision(revision: &str) -> Result<()> {
    let ok = !revision.is_empty()
        && revision
            .split('/')
            .all(|p| !p.is_empty() && p != "." && p != ".." && p.chars().all(is_valid_char));
    if ok {
        Ok(())
    } else {
        Err(Error::Hub(format!("invalid revision {revision:?}")))
    }
}

fn is_commit_hash(revision: &str) -> bool {
    revision.len() == 40 && revision.chars().all(|c| c.is_ascii_hexdigit())
}

/// `<cache>/models--org--name`.
pub(crate) fn repo_cache_dir(cache_dir: &Path, repo_id: &str) -> PathBuf {
    cache_dir.join(format!("models--{}", repo_id.replace('/', "--")))
}

/// The cached snapshot of `tokenizer.json` for `revision`, if present.
pub(crate) fn cached_file(repo_dir: &Path, revision: &str) -> Option<PathBuf> {
    let commit = if is_commit_hash(revision) {
        revision.to_owned()
    } else {
        std::fs::read_to_string(repo_dir.join("refs").join(revision))
            .ok()?
            .trim()
            .to_owned()
    };
    if !is_commit_hash(&commit) {
        return None;
    }
    let path = repo_dir.join("snapshots").join(commit).join(FILENAME);
    let (git, sha256) = file_hashes(&path).ok()?;
    // Symlinks carry the expected hash. Windows copies can be matched
    // against the content-addressed blob names in the shared HF cache.
    let valid = match std::fs::read_link(&path) {
        Ok(target) => target
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|etag| etag == git || etag == sha256),
        Err(_) => [git, sha256].iter().any(|etag| {
            let blob = repo_dir.join("blobs").join(etag);
            verify_file(&blob, etag).unwrap_or(false)
        }),
    };
    valid.then_some(path)
}

fn valid_etag(etag: &str) -> bool {
    matches!(etag.len(), 40 | 64) && etag.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Git objects include the blob header; LFS hashes are raw SHA-256.
fn file_hashes(path: &Path) -> std::io::Result<(String, String)> {
    let mut file = std::fs::File::open(path)?;
    let mut git = sha1::Sha1::new();
    git.update(format!("blob {}\0", file.metadata()?.len()).as_bytes());
    let mut sha256 = sha2::Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        git.update(&buffer[..n]);
        sha256.update(&buffer[..n]);
    }
    Ok((
        format!("{:x}", git.finalize()),
        format!("{:x}", sha256.finalize()),
    ))
}

fn verify_file(path: &Path, etag: &str) -> std::io::Result<bool> {
    let (git, sha256) = file_hashes(path)?;
    Ok(etag == git || etag == sha256)
}

fn publish_verified_file(
    file: tempfile::NamedTempFile,
    path: &Path,
    etag: &str,
) -> std::io::Result<()> {
    if !verify_file(file.path(), etag)? {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "integrity check failed: temporary contents do not match the expected hash",
        ));
    }
    if verify_file(path, etag).unwrap_or(false) {
        return Ok(());
    }
    match file.persist(path) {
        Ok(_) => Ok(()),
        Err(_) if verify_file(path, etag).unwrap_or(false) => Ok(()),
        Err(e) => Err(e.error),
    }
}

fn user_agent(params: &FromPretrainedParameters) -> String {
    let mut ua = format!("morpheme/{}; rust", crate::VERSION);
    let mut extra: Vec<_> = params.user_agent.iter().collect();
    extra.sort();
    for (k, v) in extra {
        ua.push_str(&format!("; {k}/{v}"));
    }
    ua
}

/// Percent-encode a revision for use as one URL path segment.
fn encode_segment(s: &str) -> String {
    s.chars()
        .map(|c| {
            if is_valid_char(c) {
                c.to_string()
            } else {
                c.to_string().bytes().map(|b| format!("%{b:02X}")).collect()
            }
        })
        .collect()
}

fn host_of(url: &str) -> Option<&str> {
    let rest = url.split_once("://")?.1;
    Some(rest.split(['/', '?', '#']).next().unwrap_or(rest))
}

fn header(resp: &ureq::http::Response<ureq::Body>, name: &str) -> Option<String> {
    resp.headers()
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
}

fn normalize_etag(etag: &str) -> String {
    etag.trim()
        .trim_start_matches("W/")
        .trim_matches('"')
        .to_owned()
}

/// Whether a request error means "the Hub is unreachable" (as opposed
/// to an answer from the Hub).
fn is_connection_error(e: &ureq::Error) -> bool {
    matches!(
        e,
        ureq::Error::Io(_)
            | ureq::Error::Timeout(_)
            | ureq::Error::HostNotFound
            | ureq::Error::ConnectionFailed
    )
}

fn status_error(status: u16, error_code: Option<String>, repo_id: &str, revision: &str) -> Error {
    let msg = match (status, error_code.as_deref()) {
        (401 | 403, _) | (404, Some("RepoNotFound")) => format!(
            "cannot access {repo_id:?} (HTTP {status}): the repository does not exist, \
             is private, or is gated. Pass a token (FromPretrainedParameters::token or \
             HF_TOKEN) from an account with access."
        ),
        (404, Some("RevisionNotFound")) => {
            format!("revision {revision:?} not found in {repo_id:?}")
        }
        (404, _) => format!("{repo_id:?} at revision {revision:?} has no {FILENAME}"),
        _ => format!("Hugging Face Hub returned HTTP {status} for {repo_id:?}"),
    };
    Error::Hub(msg)
}

/// Download (or reuse from cache) `tokenizer.json` for `repo_id` and
/// return its local path.
pub(crate) fn from_pretrained(
    repo_id: &str,
    params: Option<FromPretrainedParameters>,
) -> Result<PathBuf> {
    let params = params.unwrap_or_default();
    let mut config = HubConfig::from_env();
    if let Some(dir) = &params.cache_dir {
        config.cache_dir = dir.clone();
    }
    if let Some(token) = &params.token {
        config.token = Some(token.clone());
    }
    resolve(repo_id, &params, &config)
}

pub(crate) fn resolve(
    repo_id: &str,
    params: &FromPretrainedParameters,
    config: &HubConfig,
) -> Result<PathBuf> {
    validate_repo_id(repo_id)?;
    let revision = params.revision.as_str();
    validate_revision(revision)?;
    let repo_dir = repo_cache_dir(&config.cache_dir, repo_id);

    // A commit hash never changes: serve it from the cache if present.
    if is_commit_hash(revision) {
        if let Some(path) = cached_file(&repo_dir, revision) {
            return Ok(path);
        }
    }
    if config.offline {
        return cached_file(&repo_dir, revision).ok_or_else(|| {
            Error::Hub(format!(
                "{repo_id:?} at revision {revision:?} is not in the cache ({}) and \
                 HF_HUB_OFFLINE is set; missing or corrupt cache entries require an online download",
                config.cache_dir.display()
            ))
        });
    }

    match download(repo_id, revision, params, config, &repo_dir) {
        Ok(path) => Ok(path),
        Err(Fetch::Unreachable(e)) => cached_file(&repo_dir, revision).ok_or_else(|| {
            Error::Hub(format!(
                "cannot reach {} ({e}) and {repo_id:?} at revision {revision:?} is not cached",
                config.endpoint
            ))
        }),
        Err(Fetch::Failed(e)) => Err(e),
    }
}

enum Fetch {
    /// Network failure: the cache may still be used.
    Unreachable(ureq::Error),
    /// The Hub answered with an error, or writing the cache failed.
    Failed(Error),
}

impl From<std::io::Error> for Fetch {
    fn from(e: std::io::Error) -> Self {
        Fetch::Failed(Error::Hub(format!("writing to the cache failed: {e}")))
    }
}

fn agent(max_redirects: u32) -> ureq::Agent {
    ureq::Agent::config_builder()
        .max_redirects(max_redirects)
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(10)))
        .build()
        .into()
}

fn get(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
    ua: &str,
    range: bool,
) -> std::result::Result<ureq::http::Response<ureq::Body>, Fetch> {
    let mut req = agent.get(url).header("User-Agent", ua);
    if range {
        req = req.header("Range", "bytes=0-0");
    }
    if let Some(token) = token {
        req = req.header("Authorization", &format!("Bearer {token}"));
    }
    req.call().map_err(|e| {
        if is_connection_error(&e) {
            Fetch::Unreachable(e)
        } else {
            Fetch::Failed(Error::Hub(format!("request to {url} failed: {e}")))
        }
    })
}

fn download(
    repo_id: &str,
    revision: &str,
    params: &FromPretrainedParameters,
    config: &HubConfig,
    repo_dir: &Path,
) -> std::result::Result<PathBuf, Fetch> {
    let ua = user_agent(params);
    let token = config.token.as_deref();
    let mut url = format!(
        "{}/{repo_id}/resolve/{}/{FILENAME}",
        config.endpoint,
        encode_segment(revision)
    );

    // 1. Metadata: commit hash and etag, without following redirects
    //    (relative redirects, e.g. for renamed repos, are followed).
    let no_redirects = agent(0);
    let mut resp = get(&no_redirects, &url, token, &ua, true)?;
    for _ in 0..5 {
        let status = resp.status().as_u16();
        let location = header(&resp, "location");
        match location {
            Some(loc) if (300..400).contains(&status) && loc.starts_with('/') => {
                url = format!("{}{loc}", config.endpoint);
                resp = get(&no_redirects, &url, token, &ua, true)?;
            }
            _ => break,
        }
    }
    let status = resp.status().as_u16();
    if status >= 400 {
        return Err(Fetch::Failed(status_error(
            status,
            header(&resp, "x-error-code"),
            repo_id,
            revision,
        )));
    }
    let commit = header(&resp, "x-repo-commit").ok_or_else(|| {
        Fetch::Failed(Error::Hub(format!(
            "{} did not return a commit hash; is HF_ENDPOINT a Hugging Face Hub?",
            config.endpoint
        )))
    })?;
    if !is_commit_hash(&commit) {
        return Err(Fetch::Failed(Error::Hub(format!(
            "unexpected commit hash {commit:?} from the Hub"
        ))));
    }
    let etag = header(&resp, "x-linked-etag")
        .or_else(|| header(&resp, "etag"))
        .map(|e| normalize_etag(&e))
        .map(|e| e.to_ascii_lowercase())
        .filter(|e| valid_etag(e))
        .ok_or_else(|| Fetch::Failed(Error::Hub("the Hub did not return a valid ETag".into())))?;
    let download_url = match header(&resp, "location") {
        Some(loc) if (300..400).contains(&status) => loc,
        _ => url.clone(),
    };
    drop(resp);

    // 2. Blob (skipped if already cached).
    let blobs = repo_dir.join("blobs");
    let blob = blobs.join(&etag);
    if !verify_file(&blob, &etag).unwrap_or(false) {
        std::fs::create_dir_all(&blobs)?;
        // Only send the token to the Hub itself, never to a CDN.
        let same_host = host_of(&download_url) == host_of(&config.endpoint);
        let resp = get(
            &agent(10),
            &download_url,
            token.filter(|_| same_host),
            &ua,
            false,
        )?;
        let status = resp.status().as_u16();
        if status >= 400 {
            return Err(Fetch::Failed(status_error(status, None, repo_id, revision)));
        }
        let mut file = tempfile::NamedTempFile::new_in(&blobs)?;
        let mut reader = resp.into_body().into_reader();
        if let Err(e) = std::io::copy(&mut reader.by_ref(), &mut file) {
            return Err(Fetch::Failed(Error::Hub(format!(
                "download interrupted: {e}"
            ))));
        }
        file.as_file().sync_all()?;
        if !verify_file(file.path(), &etag)? {
            return Err(Fetch::Failed(Error::Hub(format!(
                "download integrity check failed for {repo_id:?}: contents do not match ETag {etag}"
            ))));
        }
        publish_verified_file(file, &blob, &etag)?;
    }

    // 3. Snapshot entry and ref.
    let snapshot_dir = repo_dir.join("snapshots").join(&commit);
    let snapshot = snapshot_dir.join(FILENAME);
    if !verify_file(&snapshot, &etag).unwrap_or(false) {
        std::fs::create_dir_all(&snapshot_dir)?;
        link_or_copy(&blob, &snapshot, &etag)?;
    }
    if revision != commit {
        let ref_path = repo_dir.join("refs").join(revision);
        if let Some(parent) = ref_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file =
            tempfile::NamedTempFile::new_in(ref_path.parent().expect("ref has a parent"))?;
        file.write_all(commit.as_bytes())?;
        file.as_file().sync_all()?;
        publish_revision_ref(file, &ref_path, &commit)?;
    }
    Ok(snapshot)
}

/// Refs must remain replaceable when a branch moves to another commit.
/// Accept an identical concurrent publication even if replacement fails.
fn publish_revision_ref(
    file: tempfile::NamedTempFile,
    path: &Path,
    commit: &str,
) -> std::io::Result<()> {
    if std::fs::read_to_string(path).is_ok_and(|existing| existing == commit) {
        return Ok(());
    }
    match file.persist(path) {
        Ok(_) => Ok(()),
        Err(_) if std::fs::read_to_string(path).is_ok_and(|existing| existing == commit) => Ok(()),
        Err(e) => Err(e.error),
    }
}

/// Point the snapshot at the blob with a relative symlink (as
/// `huggingface_hub` does), falling back to a copy where symlinks are
/// unavailable (e.g. Windows without developer mode).
fn link_or_copy(blob: &Path, snapshot: &Path, etag: &str) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let target = Path::new("..").join("..").join("blobs").join(etag);
        match std::os::unix::fs::symlink(&target, snapshot) {
            Ok(()) => return Ok(()),
            Err(e)
                if e.kind() == std::io::ErrorKind::AlreadyExists
                    && verify_file(snapshot, etag).unwrap_or(false) =>
            {
                return Ok(());
            }
            Err(_) => {}
        }
    }
    let _ = etag;
    let mut file =
        tempfile::NamedTempFile::new_in(snapshot.parent().expect("snapshot has a parent"))?;
    std::io::copy(&mut std::fs::File::open(blob)?, &mut file)?;
    file.as_file().sync_all()?;
    publish_verified_file(file, snapshot, etag)
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    fn git_hash(body: &[u8]) -> String {
        let mut hash = sha1::Sha1::new();
        hash.update(format!("blob {}\0", body.len()).as_bytes());
        hash.update(body);
        format!("{:x}", hash.finalize())
    }

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(key, _)| *key == k)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn repo_id_validation() {
        for ok in [
            "bert-base-uncased",
            "google-bert/bert-base-uncased",
            "Qwen/Qwen2.5-0.5B",
            "a_b.c",
        ] {
            assert!(validate_repo_id(ok).is_ok(), "{ok}");
        }
        for bad in [
            "", "/x", "x/", "a/b/c", "../x", "x/..", "a/../b", ".hidden", "-x", "a b", "a\\b",
            "C:x", "a?b", "a%2F..",
        ] {
            assert!(validate_repo_id(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn revision_validation() {
        for ok in ["main", "v1.0", COMMIT, "refs/pr/1"] {
            assert!(validate_revision(ok).is_ok(), "{ok}");
        }
        for bad in ["", "..", "../main", "a//b", "/abs", "a b", "a\\b"] {
            assert!(validate_revision(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn cache_dir_precedence() {
        let c = HubConfig::from_lookup(env(&[("HOME", "/home/u")]));
        assert_eq!(c.cache_dir, PathBuf::from("/home/u/.cache/huggingface/hub"));
        let c = HubConfig::from_lookup(env(&[("HOME", "/home/u"), ("XDG_CACHE_HOME", "/xdg")]));
        assert_eq!(c.cache_dir, PathBuf::from("/xdg/huggingface/hub"));
        let c = HubConfig::from_lookup(env(&[("HOME", "/home/u"), ("HF_HOME", "/hf")]));
        assert_eq!(c.cache_dir, PathBuf::from("/hf/hub"));
        let c = HubConfig::from_lookup(env(&[("HF_HOME", "/hf"), ("HF_HUB_CACHE", "/cache")]));
        assert_eq!(c.cache_dir, PathBuf::from("/cache"));
        assert_eq!(c.endpoint, DEFAULT_ENDPOINT);
        assert!(!c.offline);
    }

    #[test]
    fn token_precedence_and_offline_flag() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().to_str().unwrap();
        std::fs::write(dir.path().join("token"), "file-token\n").unwrap();
        let c = HubConfig::from_lookup(env(&[("HF_HOME", home)]));
        assert_eq!(c.token.as_deref(), Some("file-token"));
        let c = HubConfig::from_lookup(env(&[("HF_HOME", home), ("HF_TOKEN", "env-token")]));
        assert_eq!(c.token.as_deref(), Some("env-token"));
        for (v, want) in [
            ("1", true),
            ("TRUE", true),
            ("yes", true),
            ("0", false),
            ("no", false),
        ] {
            let c = HubConfig::from_lookup(env(&[("HF_HUB_OFFLINE", v)]));
            assert_eq!(c.offline, want, "{v}");
        }
        let c = HubConfig::from_lookup(env(&[("HF_ENDPOINT", "https://mirror.example/")]));
        assert_eq!(c.endpoint, "https://mirror.example");
    }

    fn populate(cache: &Path, repo: &str, revision: &str) -> PathBuf {
        let repo_dir = repo_cache_dir(cache, repo);
        let snap = repo_dir.join("snapshots").join(COMMIT);
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join(FILENAME), "{}").unwrap();
        let etag = file_hashes(&snap.join(FILENAME)).unwrap().0;
        std::fs::create_dir_all(repo_dir.join("blobs")).unwrap();
        std::fs::write(repo_dir.join("blobs").join(etag), "{}").unwrap();
        std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
        std::fs::write(repo_dir.join("refs").join(revision), COMMIT).unwrap();
        snap.join(FILENAME)
    }

    fn config(cache: &Path, endpoint: &str, offline: bool) -> HubConfig {
        HubConfig {
            endpoint: endpoint.to_owned(),
            cache_dir: cache.to_owned(),
            token: None,
            offline,
        }
    }

    #[test]
    fn offline_mode_serves_cache_only() {
        let dir = tempfile::tempdir().unwrap();
        let want = populate(dir.path(), "org/model", "main");
        let cfg = config(dir.path(), "http://127.0.0.1:9", true);
        let params = FromPretrainedParameters::default();
        assert_eq!(resolve("org/model", &params, &cfg).unwrap(), want);
        let by_commit = params.clone().revision(COMMIT);
        assert_eq!(resolve("org/model", &by_commit, &cfg).unwrap(), want);
        let err = resolve("org/other", &params, &cfg).unwrap_err();
        assert!(err.to_string().contains("HF_HUB_OFFLINE"), "{err}");
    }

    #[test]
    fn unreachable_hub_falls_back_to_cache() {
        let dir = tempfile::tempdir().unwrap();
        let want = populate(dir.path(), "org/model", "main");
        // Port 9 (discard) is closed on test machines: connection refused.
        let cfg = config(dir.path(), "http://127.0.0.1:9", false);
        let params = FromPretrainedParameters::default();
        assert_eq!(resolve("org/model", &params, &cfg).unwrap(), want);
        let err = resolve("org/missing", &params, &cfg).unwrap_err();
        assert!(err.to_string().contains("cannot reach"), "{err}");
    }

    /// A minimal HTTP/1.1 server answering each request with the next
    /// canned response. Returns the base URL and a request counter.
    fn serve(responses: Vec<String>) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let count = Arc::new(AtomicUsize::new(0));
        let c = count.clone();
        std::thread::spawn(move || {
            for (stream, response) in listener.incoming().zip(responses) {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap() > 0 && line != "\r\n" {
                    line.clear();
                }
                c.fetch_add(1, Ordering::SeqCst);
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (url, count)
    }

    fn response(status: &str, headers: &[(&str, &str)], body: &str) -> String {
        let mut r = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
            body.len()
        );
        for (k, v) in headers {
            r.push_str(&format!("{k}: {v}\r\n"));
        }
        r.push_str("\r\n");
        r.push_str(body);
        r
    }

    #[test]
    fn downloads_into_hf_cache_layout_and_reuses_it() {
        let body = r#"{"version":"1.0"}"#;
        let etag = git_hash(body.as_bytes());
        let headers = [("X-Repo-Commit", COMMIT), ("ETag", etag.as_str())];
        let (url, count) = serve(vec![
            response("206 Partial Content", &headers, "{"),
            response("200 OK", &headers, body),
            response("206 Partial Content", &headers, "{"),
        ]);
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path(), &url, false);
        let params = FromPretrainedParameters::default();

        let path = resolve("org/model", &params, &cfg).unwrap();
        let repo_dir = repo_cache_dir(dir.path(), "org/model");
        assert_eq!(path, repo_dir.join("snapshots").join(COMMIT).join(FILENAME));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), body);
        assert_eq!(
            std::fs::read_to_string(repo_dir.join("blobs").join(&etag)).unwrap(),
            body
        );
        assert_eq!(
            std::fs::read_to_string(repo_dir.join("refs/main")).unwrap(),
            COMMIT
        );
        assert_eq!(count.load(Ordering::SeqCst), 2);

        // Second call: metadata only, the blob is reused.
        assert_eq!(resolve("org/model", &params, &cfg).unwrap(), path);
        assert_eq!(count.load(Ordering::SeqCst), 3);

        // Pinned commit: no request at all.
        let pinned = params.clone().revision(COMMIT);
        assert_eq!(resolve("org/model", &pinned, &cfg).unwrap(), path);
        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn hash_verification_uses_git_object_headers_and_raw_lfs_sha256() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        std::fs::write(&path, b"hello\n").unwrap();
        // Known Git object hash, not raw SHA-1 of the contents.
        assert!(verify_file(&path, "ce013625030ba8dba906f756967f9e9ca394464a").unwrap());
        assert!(
            verify_file(
                &path,
                "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
            )
            .unwrap()
        );
        assert!(!verify_file(&path, &git_hash(b"different")).unwrap());
        for invalid in [
            "abc123",
            "../hash",
            "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
        ] {
            assert!(!valid_etag(invalid));
        }
    }

    #[test]
    fn mismatched_download_is_not_published_and_corrupt_cache_is_repaired() {
        let body = r#"{"version":"1.0"}"#;
        for etag in [
            git_hash(body.as_bytes()),
            format!("{:x}", sha2::Sha256::digest(body.as_bytes())),
        ] {
            let headers = [("X-Repo-Commit", COMMIT), ("ETag", etag.as_str())];
            let (url, _) = serve(vec![
                response("206 Partial Content", &headers, "{"),
                response("200 OK", &headers, "corrupt download"),
                response("206 Partial Content", &headers, "{"),
                response("200 OK", &headers, body),
                response("206 Partial Content", &headers, "{"),
                response("200 OK", &headers, body),
            ]);
            let dir = tempfile::tempdir().unwrap();
            let cfg = config(dir.path(), &url, false);
            let params = FromPretrainedParameters::default();
            let error = resolve("org/model", &params, &cfg).unwrap_err();
            assert!(error.to_string().contains("integrity check failed"));
            let repo = repo_cache_dir(dir.path(), "org/model");
            assert_eq!(std::fs::read_dir(repo.join("blobs")).unwrap().count(), 0);
            assert!(!repo.join("refs/main").exists());
            let path = resolve("org/model", &params, &cfg).unwrap();
            // On Unix this corrupts the linked blob; on Windows it corrupts
            // the snapshot copy. Repair both independently.
            std::fs::write(&path, "corrupt snapshot").unwrap();
            std::fs::write(repo.join("blobs").join(&etag), "corrupt blob").unwrap();
            let offline = config(dir.path(), &url, true);
            let pinned = params.clone().revision(COMMIT);
            assert!(
                resolve("org/model", &pinned, &offline)
                    .unwrap_err()
                    .to_string()
                    .contains("corrupt")
            );
            assert_eq!(resolve("org/model", &pinned, &cfg).unwrap(), path);
            assert_eq!(std::fs::read_to_string(&path).unwrap(), body);
            assert!(
                verify_file(&repo.join("blobs").join(etag), &git_hash(body.as_bytes())).unwrap()
            );
            assert_eq!(resolve("org/model", &pinned, &offline).unwrap(), path);
        }
    }

    #[test]
    fn concurrent_downloads_publish_complete_cache_entries() {
        use std::sync::Barrier;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let body = Arc::new("complete tokenizer contents ".repeat(4096));
        let barrier = Arc::new(Barrier::new(2));
        let server_body = body.clone();
        let server = std::thread::spawn(move || {
            let mut handlers = Vec::new();
            // Each caller requests metadata and then the same blob.
            for stream in listener.incoming().take(4) {
                let mut stream = stream.unwrap();
                let body = server_body.clone();
                let barrier = barrier.clone();
                handlers.push(std::thread::spawn(move || {
                    stream
                        .set_read_timeout(Some(Duration::from_secs(10)))
                        .unwrap();
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut metadata = false;
                    let mut line = String::new();
                    while reader.read_line(&mut line).unwrap() > 0 && line != "\r\n" {
                        metadata |= line.to_ascii_lowercase().starts_with("range:");
                        line.clear();
                    }
                    let etag = git_hash(body.as_bytes());
                    let headers = [("X-Repo-Commit", COMMIT), ("ETag", etag.as_str())];
                    if metadata {
                        stream
                            .write_all(response("206 Partial Content", &headers, "{").as_bytes())
                            .unwrap();
                    } else {
                        let reply = response("200 OK", &headers, &body);
                        let cut = reply.len() - body.len() / 2;
                        stream.write_all(&reply.as_bytes()[..cut]).unwrap();
                        stream.flush().unwrap();
                        // Neither blob can finish until both downloads are active.
                        barrier.wait();
                        stream.write_all(&reply.as_bytes()[cut..]).unwrap();
                    }
                }));
            }
            for handler in handlers {
                handler.join().unwrap();
            }
        });
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path(), &url, false);
        let params = FromPretrainedParameters::default();
        std::thread::scope(|scope| {
            let first = scope.spawn(|| resolve("org/model", &params, &cfg).unwrap());
            let second = scope.spawn(|| resolve("org/model", &params, &cfg).unwrap());
            let first = first.join().unwrap();
            let second = second.join().unwrap();
            assert_eq!(first, second);
            assert_eq!(std::fs::read_to_string(first).unwrap(), *body);
        });
        server.join().unwrap();
        let repo = repo_cache_dir(dir.path(), "org/model");
        assert_eq!(
            std::fs::read_to_string(repo.join("blobs").join(git_hash(body.as_bytes()))).unwrap(),
            *body
        );
        assert_eq!(
            std::fs::read_to_string(repo.join("refs/main")).unwrap(),
            COMMIT
        );
        assert_eq!(std::fs::read_dir(repo.join("blobs")).unwrap().count(), 1);
        assert_eq!(std::fs::read_dir(repo.join("refs")).unwrap().count(), 1);
    }

    #[test]
    fn cache_publication_does_not_replace_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob");
        std::fs::write(&path, "winner").unwrap();
        let mut file = tempfile::NamedTempFile::new_in(dir.path()).unwrap();
        file.write_all(b"winner").unwrap();
        publish_verified_file(file, &path, &git_hash(b"winner")).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "winner");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn revision_refs_accept_identical_publications_and_follow_branch_updates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("main");
        for commit in ["old", "old", "new", "new"] {
            let mut file = tempfile::NamedTempFile::new_in(dir.path()).unwrap();
            file.write_all(commit.as_bytes()).unwrap();
            publish_revision_ref(file, &path, commit).unwrap();
            assert_eq!(std::fs::read_to_string(&path).unwrap(), commit);
            assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        }
    }

    #[test]
    fn publication_rejects_corrupt_temporary_contents_without_replacing_destination() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob");
        for existing in [None, Some("existing")] {
            if let Some(contents) = existing {
                std::fs::write(&path, contents).unwrap();
            }
            let mut file = tempfile::NamedTempFile::new_in(dir.path()).unwrap();
            file.write_all(b"corrupt").unwrap();
            let error = publish_verified_file(file, &path, &git_hash(b"expected")).unwrap_err();
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
            match existing {
                Some(contents) => assert_eq!(std::fs::read_to_string(&path).unwrap(), contents),
                None => assert!(!path.exists()),
            }
            assert_eq!(
                std::fs::read_dir(dir.path()).unwrap().count(),
                usize::from(existing.is_some())
            );
        }
    }

    #[test]
    fn hub_errors_are_explained() {
        let (url, _) = serve(vec![
            response("401 Unauthorized", &[("X-Error-Code", "GatedRepo")], ""),
            response("404 Not Found", &[("X-Error-Code", "RevisionNotFound")], ""),
            response("404 Not Found", &[("X-Error-Code", "EntryNotFound")], ""),
        ]);
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path(), &url, false);
        let params = FromPretrainedParameters::default();
        let e1 = resolve("org/gated", &params, &cfg).unwrap_err().to_string();
        assert!(e1.contains("token"), "{e1}");
        let e2 = resolve("org/m", &params, &cfg).unwrap_err().to_string();
        assert!(e2.contains("revision"), "{e2}");
        let e3 = resolve("org/m", &params, &cfg).unwrap_err().to_string();
        assert!(e3.contains("no tokenizer.json"), "{e3}");
    }

    #[test]
    fn revision_with_slash_is_encoded() {
        assert_eq!(encode_segment("refs/pr/1"), "refs%2Fpr%2F1");
        assert_eq!(
            host_of("https://huggingface.co/x/y"),
            Some("huggingface.co")
        );
        assert_eq!(
            host_of("https://cdn-lfs.hf.co/abc?x=1"),
            Some("cdn-lfs.hf.co")
        );
    }
}
