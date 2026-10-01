#!/usr/bin/env bash
# Download the pinned Hugging Face tokenizer.json files used by the
# golden interop tests (crates/splinter/tests/hf_golden.rs).
#
# Files land in crates/splinter/tests/data/hf/ (gitignored). Revisions
# are pinned in scripts/hf-fixtures.txt so results are reproducible.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/crates/splinter/tests/data/hf"
mkdir -p "$out"
grep -v '^#' "$root/scripts/hf-fixtures.txt" | while read -r name repo rev; do
  [ -z "${name:-}" ] && continue
  dest="$out/$name.json"
  if [ -s "$dest" ]; then continue; fi
  echo "fetching $repo@${rev:0:8} -> $name.json"
  curl -fsSL --retry 3 -o "$dest.tmp" "https://huggingface.co/$repo/resolve/$rev/tokenizer.json"
  mv "$dest.tmp" "$dest"
done
