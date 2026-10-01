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
/// Connect, response-header and body-read timeout (`huggingface_hub`
/// uses a 10 s read timeout as well).
const TIMEOUT: Duration = Duration::from_secs(10);
/// Pause before the single retry of a `429`/`5xx` metadata or blob request.
const RETRY_BACKOFF: Duration = Duration::from_millis(500);
/// Permissions of newly created cache files and saved tokenizers on Unix
/// (what `huggingface_hub` produces under the default umask).
#[cfg(unix)]
const NEW_FILE_MODE: u32 = 0o644;

/// Options for [`crate::Tokenizer::from_pretrained`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FromPretrainedParameters {
    /// Branch, tag or commit hash. Defaults to `main`.
    pub revision: String,
    /// Extra `key/value` pairs appended to the `User-Agent` header.
    pub user_agent: HashMap<String, String>,
    /// Access token for private or gated repositories. Defaults to
    /// `HF_TOKEN`, then the token saved by `huggingface-cli login`
    /// (unless [`anonymous`](Self::anonymous) is set).
    pub token: Option<String>,
    /// Do not read `HF_TOKEN` or the token file: send no token unless
    /// one is given explicitly with [`token`](Self::token).
    pub anonymous: bool,
    /// Cache directory. Defaults to the shared Hugging Face cache.
    pub cache_dir: Option<PathBuf>,
}

impl Default for FromPretrainedParameters {
    fn default() -> Self {
        Self {
            revision: "main".to_owned(),
            user_agent: HashMap::new(),
            token: None,
            anonymous: false,
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

    /// Do not read `HF_TOKEN` or the token file saved by
    /// `huggingface-cli login`: requests for public repositories carry
    /// no credentials. A token passed with [`token`](Self::token) is
    /// still used.
    #[must_use]
    pub fn anonymous(mut self) -> Self {
        self.anonymous = true;
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
                // `--` separates org and name in the cache directory, so
                // `acme--model` would collide with `acme/model`
                // (huggingface_hub rejects it too).
                && !p.contains("--")
        });
    if ok {
        Ok(())
    } else {
        Err(Error::Hub(format!(
            "invalid model id {id:?}: expected `name` or `org/name` using letters, digits, '-', '_' and '.' (no `..` or `--`)"
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

/// Give a temporary file the permissions its destination will need:
/// those of an existing destination, otherwise the shared-cache default
/// (`tempfile` creates files `0600`, which other users of a shared cache
/// or a later container uid cannot read). No-op on non-Unix platforms.
fn prepare_permissions(file: &std::fs::File, destination: &Path) -> std::io::Result<()> {
    let existing = std::fs::metadata(destination).ok().map(|m| m.permissions());
    #[cfg(unix)]
    let permissions = {
        use std::os::unix::fs::PermissionsExt;
        existing.unwrap_or_else(|| std::fs::Permissions::from_mode(NEW_FILE_MODE))
    };
    #[cfg(not(unix))]
    let Some(permissions) = existing else {
        return Ok(());
    };
    file.set_permissions(permissions)
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
    prepare_permissions(file.as_file(), path)?;
    match super::persist_with_retry(file, path) {
        Ok(()) => Ok(()),
        Err(_) if verify_file(path, etag).unwrap_or(false) => Ok(()),
        Err(e) => Err(e),
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

/// `(scheme, host, port)` of a URL, with the scheme's default port.
fn origin_of(url: &str) -> Option<(String, String, u16)> {
    let (scheme, rest) = url.split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    // Ignore user-info; a `host:port` after it is still an authority.
    let authority = authority.rsplit('@').next().unwrap_or(authority);
    let default_port = match scheme.as_str() {
        "https" => 443,
        "http" => 80,
        _ => return None,
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) if !host.ends_with(']') || host.starts_with('[') => {
            (host, port.parse().ok()?)
        }
        _ => (authority, default_port),
    };
    if host.is_empty() {
        return None;
    }
    Some((scheme, host.to_ascii_lowercase(), port))
}

/// Whether a bearer token sent to `endpoint` may also be sent to `url`:
/// only to the very same origin, and never after a downgrade to
/// cleartext (an `https` endpoint redirecting to `http://` must not see
/// the token again on port 80). A plain `http` endpoint (a local mirror
/// or test server) is allowed to redirect within itself.
fn same_target(url: &str, endpoint: &str) -> bool {
    match (origin_of(url), origin_of(endpoint)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

/// Strip the query string and fragment from a URL before embedding it in
/// an error message (presigned LFS/CDN URLs carry credentials there).
fn redact_url(url: &str) -> &str {
    url.split(['?', '#']).next().unwrap_or(url)
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
    if params.anonymous {
        config.token = None;
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
        // A rate limit or server error is served from the cache, as
        // huggingface_hub does; without a valid cache entry the HTTP
        // error is reported.
        Err(Fetch::Unavailable(e)) => cached_file(&repo_dir, revision).ok_or(e),
        Err(Fetch::Failed(e)) => Err(e),
    }
}

enum Fetch {
    /// Network failure: the cache may still be used.
    Unreachable(ureq::Error),
    /// The Hub answered the metadata request with a transient error
    /// (`429`, `5xx`, …) even after a retry: the cache may still be used.
    Unavailable(Error),
    /// The Hub answered with a definitive error, or writing the cache failed.
    Failed(Error),
}

/// Statuses worth one retry and, for metadata, a cache fallback: not the
/// definitive answers (`401`/`403` no access, `404` missing).
fn is_transient_status(status: u16) -> bool {
    status >= 400 && !matches!(status, 401 | 403 | 404)
}

impl From<std::io::Error> for Fetch {
    fn from(e: std::io::Error) -> Self {
        Fetch::Failed(Error::Hub(format!("writing to the cache failed: {e}")))
    }
}

/// An agent that follows at most `max_redirects` redirects. With
/// `https_only`, every request (including redirect targets) must use TLS.
fn agent(max_redirects: u32, https_only: bool) -> ureq::Agent {
    ureq::Agent::config_builder()
        .max_redirects(max_redirects)
        .http_status_as_error(false)
        .https_only(https_only)
        .timeout_connect(Some(TIMEOUT))
        .timeout_recv_response(Some(TIMEOUT))
        .timeout_recv_body(Some(TIMEOUT))
        .build()
        .into()
}

/// One request, retried once after a short pause on a transient status.
fn get(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
    ua: &str,
    range: bool,
) -> std::result::Result<ureq::http::Response<ureq::Body>, Fetch> {
    let resp = get_once(agent, url, token, ua, range)?;
    if !is_transient_status(resp.status().as_u16()) {
        return Ok(resp);
    }
    drop(resp);
    std::thread::sleep(RETRY_BACKOFF);
    get_once(agent, url, token, ua, range)
}

fn get_once(
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
            Fetch::Failed(Error::Hub(format!(
                "request to {} failed: {e}",
                redact_url(url)
            )))
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
    // Never let a redirect downgrade an https endpoint to cleartext.
    let https_only = config.endpoint.starts_with("https://");

    // 1. Metadata: commit hash and etag, without following redirects
    //    (relative redirects, e.g. for renamed repos, are followed).
    let no_redirects = agent(0, https_only);
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
        let error = status_error(status, header(&resp, "x-error-code"), repo_id, revision);
        return Err(if is_transient_status(status) {
            Fetch::Unavailable(error)
        } else {
            Fetch::Failed(error)
        });
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
    // A pinned commit must come back as itself; a mirror serving another
    // commit would otherwise be cached under a misleading ref.
    if is_commit_hash(revision) && commit != revision {
        return Err(Fetch::Failed(Error::Hub(format!(
            "{} returned commit {commit} for {repo_id:?} at pinned revision {revision}; \
             is HF_ENDPOINT a faithful mirror?",
            config.endpoint
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
        // Only send the token to the Hub itself over the same scheme:
        // never to a CDN, never over cleartext after a downgrade.
        let same_origin = same_target(&download_url, &config.endpoint);
        let resp = get(
            &agent(10, https_only),
            &download_url,
            token.filter(|_| same_origin),
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
    prepare_permissions(file.as_file(), path)?;
    match super::persist_with_retry(file, path) {
        Ok(()) => Ok(()),
        Err(_) if std::fs::read_to_string(path).is_ok_and(|existing| existing == commit) => Ok(()),
        Err(e) => Err(e),
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
    }

    #[test]
    fn repo_ids_with_double_dash_are_rejected() {
        // `acme--model` would share `models--acme--model` with `acme/model`.
        for bad in ["acme--model", "acme/mo--del", "a--b/c"] {
            assert!(validate_repo_id(bad).is_err(), "{bad}");
        }
        assert_eq!(
            repo_cache_dir(Path::new("/c"), "acme/model"),
            PathBuf::from("/c/models--acme--model")
        );
    }

    #[test]
    fn token_is_only_sent_to_the_same_origin() {
        let hub = "https://huggingface.co";
        assert!(same_target("https://huggingface.co/x/y?sig=1", hub));
        assert!(same_target("HTTPS://HuggingFace.co:443/x", hub));
        // Downgrade to cleartext on the same host: no token.
        assert!(!same_target("http://huggingface.co/x/y", hub));
        assert!(!same_target("http://huggingface.co:443/x/y", hub));
        // Other hosts and ports: no token.
        assert!(!same_target("https://cdn-lfs.hf.co/abc?x=1", hub));
        assert!(!same_target("https://huggingface.co:8443/x", hub));
        assert!(!same_target("https://huggingface.co.evil.example/x", hub));
        assert!(!same_target("https://huggingface.co@evil.example/x", hub));
        assert!(!same_target("ftp://huggingface.co/x", hub));
        assert!(!same_target("huggingface.co/x", hub));
        // A plain-http endpoint (local mirror) may redirect within itself.
        let local = "http://127.0.0.1:8080";
        assert!(same_target("http://127.0.0.1:8080/blob", local));
        assert!(!same_target("http://127.0.0.1:8081/blob", local));
        assert!(!same_target("http://127.0.0.1/blob", local));
        assert!(!same_target("https://127.0.0.1:8080/blob", local));
        assert!(same_target("http://localhost/blob", "http://localhost:80"));
        assert!(same_target("https://[::1]:8443/x", "https://[::1]:8443"));
        assert!(!same_target("https://[::1]:8443/x", "https://[::1]"));
    }

    #[test]
    fn error_messages_redact_presigned_urls() {
        let url = "https://cdn-lfs.hf.co/repos/ab/cd/file?X-Amz-Signature=SECRET&Expires=1#frag";
        assert_eq!(redact_url(url), "https://cdn-lfs.hf.co/repos/ab/cd/file");
        assert_eq!(redact_url("https://hf.co/x"), "https://hf.co/x");

        // A server answering with garbage is a protocol error (not
        // "unreachable"), so the URL lands in the message: without its
        // query string.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            while reader.read_line(&mut line).unwrap() > 0 && line != "\r\n" {
                line.clear();
            }
            stream.write_all(b"garbage\r\n\r\n").unwrap();
        });
        let url = format!("http://{addr}/blob?X-Amz-Signature=SECRET");
        let err = match get_once(&agent(0, false), &url, None, "ua", false) {
            Ok(_) => panic!("garbage is not a response"),
            Err(Fetch::Failed(e)) => e.to_string(),
            Err(Fetch::Unreachable(e)) => panic!("unexpected connection error: {e}"),
            Err(Fetch::Unavailable(e)) => panic!("unexpected transient error: {e}"),
        };
        assert!(err.contains("/blob"), "{err}");
        assert!(!err.contains("SECRET"), "{err}");
    }

    /// Like [`serve`], also counting requests carrying an `Authorization`
    /// header.
    fn serve_counting_auth(responses: Vec<String>) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let authorized = Arc::new(AtomicUsize::new(0));
        let seen = authorized.clone();
        std::thread::spawn(move || {
            for (stream, response) in listener.incoming().zip(responses) {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap() > 0 && line != "\r\n" {
                    if line.to_ascii_lowercase().starts_with("authorization:") {
                        seen.fetch_add(1, Ordering::SeqCst);
                    }
                    line.clear();
                }
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (url, authorized)
    }

    #[test]
    fn token_is_withheld_from_other_origins_after_a_redirect() {
        let body = r#"{"version":"1.0"}"#;
        let etag = git_hash(body.as_bytes());
        let headers = [("X-Repo-Commit", COMMIT), ("ETag", etag.as_str())];

        // Metadata on the endpoint redirects to a blob on another origin
        // (same scheme, different port): the token must not follow.
        let (blob_url, blob_authorized) =
            serve_counting_auth(vec![response("200 OK", &headers, body)]);
        let location = format!("{blob_url}/blob?X-Amz-Signature=SECRET");
        let redirect = [
            ("X-Repo-Commit", COMMIT),
            ("ETag", etag.as_str()),
            ("Location", location.as_str()),
        ];
        let (url, authorized) = serve_counting_auth(vec![response("302 Found", &redirect, "")]);
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = config(dir.path(), &url, false);
        cfg.token = Some("hf_secret".into());
        let params = FromPretrainedParameters::default();
        let path = resolve("org/model", &params, &cfg).unwrap();
        assert_eq!(std::fs::read_to_string(path).unwrap(), body);
        assert_eq!(authorized.load(Ordering::SeqCst), 1);
        assert_eq!(blob_authorized.load(Ordering::SeqCst), 0);

        // The blob served by the endpoint itself (same plain-http origin)
        // does receive the token.
        let (url, authorized) = serve_counting_auth(vec![
            response("206 Partial Content", &headers, "{"),
            response("200 OK", &headers, body),
        ]);
        let other = tempfile::tempdir().unwrap();
        let mut cfg = config(other.path(), &url, false);
        cfg.token = Some("hf_secret".into());
        resolve("org/model", &params, &cfg).unwrap();
        assert_eq!(authorized.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn server_errors_fall_back_to_cache_after_one_retry() {
        // 503 twice (initial + retry) with a valid cache: served from cache.
        let (url, count) = serve(vec![
            response("503 Service Unavailable", &[], ""),
            response("503 Service Unavailable", &[], ""),
        ]);
        let dir = tempfile::tempdir().unwrap();
        let want = populate(dir.path(), "org/model", "main");
        let cfg = config(dir.path(), &url, false);
        let params = FromPretrainedParameters::default();
        assert_eq!(resolve("org/model", &params, &cfg).unwrap(), want);
        assert_eq!(count.load(Ordering::SeqCst), 2);

        // 503 with an empty cache: the HTTP error is reported.
        let (url, count) = serve(vec![
            response("503 Service Unavailable", &[], ""),
            response("503 Service Unavailable", &[], ""),
        ]);
        let empty = tempfile::tempdir().unwrap();
        let cfg = config(empty.path(), &url, false);
        let err = resolve("org/model", &params, &cfg).unwrap_err().to_string();
        assert!(err.contains("503"), "{err}");
        assert_eq!(count.load(Ordering::SeqCst), 2);

        // 429 once, then success: the retry completes the download.
        let body = r#"{"version":"1.0"}"#;
        let etag = git_hash(body.as_bytes());
        let headers = [("X-Repo-Commit", COMMIT), ("ETag", etag.as_str())];
        let (url, count) = serve(vec![
            response("429 Too Many Requests", &[], ""),
            response("206 Partial Content", &headers, "{"),
            response("200 OK", &headers, body),
        ]);
        let cfg = config(empty.path(), &url, false);
        let path = resolve("org/model", &params, &cfg).unwrap();
        assert_eq!(std::fs::read_to_string(path).unwrap(), body);
        assert_eq!(count.load(Ordering::SeqCst), 3);

        // Definitive answers (404) are not retried and do not use the cache.
        let (url, count) = serve(vec![response(
            "404 Not Found",
            &[("X-Error-Code", "RevisionNotFound")],
            "",
        )]);
        let cfg = config(dir.path(), &url, false);
        let err = resolve("org/model", &params, &cfg).unwrap_err().to_string();
        assert!(err.contains("revision"), "{err}");
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn pinned_commit_must_match_the_served_commit() {
        let other = "fedcba9876543210fedcba9876543210fedcba98";
        let body = r#"{"version":"1.0"}"#;
        let etag = git_hash(body.as_bytes());
        let headers = [("X-Repo-Commit", other), ("ETag", etag.as_str())];
        let (url, count) = serve(vec![response("206 Partial Content", &headers, "{")]);
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path(), &url, false);
        let params = FromPretrainedParameters::default().revision(COMMIT);
        let err = resolve("org/model", &params, &cfg).unwrap_err().to_string();
        assert!(err.contains(other) && err.contains(COMMIT), "{err}");
        assert_eq!(count.load(Ordering::SeqCst), 1);
        let repo = repo_cache_dir(dir.path(), "org/model");
        assert!(!repo.join("snapshots").exists());
        assert!(!repo.join("refs").exists());
    }

    #[cfg(unix)]
    #[test]
    fn new_cache_files_are_readable_by_other_users() {
        use std::os::unix::fs::PermissionsExt;
        let body = r#"{"version":"1.0"}"#;
        let etag = git_hash(body.as_bytes());
        let headers = [("X-Repo-Commit", COMMIT), ("ETag", etag.as_str())];
        let (url, _) = serve(vec![
            response("206 Partial Content", &headers, "{"),
            response("200 OK", &headers, body),
        ]);
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path(), &url, false);
        resolve("org/model", &FromPretrainedParameters::default(), &cfg).unwrap();
        let repo = repo_cache_dir(dir.path(), "org/model");
        for file in [repo.join("blobs").join(&etag), repo.join("refs/main")] {
            let mode = std::fs::metadata(&file).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, NEW_FILE_MODE, "{}: {mode:o}", file.display());
            assert_ne!(mode & 0o044, 0, "{}: {mode:o}", file.display());
        }

        // Existing permissions are kept when a ref is replaced.
        let ref_path = repo.join("refs/main");
        std::fs::set_permissions(&ref_path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let mut file = tempfile::NamedTempFile::new_in(ref_path.parent().unwrap()).unwrap();
        file.write_all(b"other").unwrap();
        publish_revision_ref(file, &ref_path, "other").unwrap();
        let mode = std::fs::metadata(&ref_path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }

    #[cfg(unix)]
    #[test]
    fn saved_tokenizer_is_not_private_to_the_creator() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = crate::Tokenizer::new(crate::models::WordLevel::default());
        let path = dir.path().join("tokenizer.json");
        tokenizer.save(&path, false).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o644, "{mode:o}");

        // Existing permissions are preserved on replacement.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        tokenizer.save(&path, true).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "{mode:o}");
    }
}
