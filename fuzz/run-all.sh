#!/usr/bin/env bash
# Run every fuzz target in parallel for N seconds (default 60).
#
#   ./fuzz/run-all.sh 600
#
# Needs nightly + cargo-fuzz. Seeds come from fuzz/make_corpus.py; logs go
# to fuzz/logs/<target>.log and crashes to fuzz/artifacts/<target>/.
# Exits non-zero if any target finds a crash or timeout.
set -uo pipefail
cd "$(dirname "$0")/.."
seconds="${1:-60}"
targets=(load_json encode normalized_string components_json)

[ -d fuzz/corpus ] || python3 fuzz/make_corpus.py
mkdir -p fuzz/logs
cargo +nightly fuzz build

pids=()
for t in "${targets[@]}"; do
  mkdir -p "fuzz/corpus/$t"
  cargo +nightly fuzz run "$t" "fuzz/corpus/$t" -- \
    -max_total_time="$seconds" -timeout=10 -rss_limit_mb=2048 \
    >"fuzz/logs/$t.log" 2>&1 &
  pids+=($!)
done

status=0
for i in "${!targets[@]}"; do
  if wait "${pids[$i]}"; then
    echo "ok    ${targets[$i]}"
  else
    echo "FAIL  ${targets[$i]} (see fuzz/logs/${targets[$i]}.log)"
    status=1
  fi
done
exit $status
