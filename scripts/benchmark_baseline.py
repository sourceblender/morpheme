#!/usr/bin/env python3
"""Record isolated timing/RSS samples and compare only compatible local baselines."""
import argparse
import hashlib
import json
import os
import pathlib
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ("bert-base-uncased", "gpt2", "llama", "t5-small")
OPERATIONS = ("encode_seq", "encode_batch", "decode_seq", "decode_batch")


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, encoding="utf-8").strip()


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def medians(current, baseline):
    """Yield (workload, metric, baseline median, current median) for every comparable pair."""
    # Revision is deliberately excluded: comparing different commits is the purpose.
    for field in ("schema_version", "machine", "settings", "inputs", "probe_sha256"):
        if current[field] != baseline[field]:
            raise ValueError(f"incompatible baseline: {field} differs")
    if current["results"].keys() != baseline["results"].keys():
        raise ValueError("incompatible baseline: workloads differ")
    for name, result in current["results"].items():
        old = baseline["results"][name]
        for metric in ("seconds", "peak_rss_bytes"):
            new_values = [sample[metric] for sample in result["samples"] if sample[metric] is not None]
            old_values = [sample[metric] for sample in old["samples"] if sample[metric] is not None]
            if not new_values and not old_values:
                continue
            if not new_values or not old_values:
                raise ValueError(f"incompatible baseline: {name} {metric} availability differs")
            yield name, metric, statistics.median(old_values), statistics.median(new_values)


def compare(current, baseline, limit):
    regressions = []
    for name, metric, before, after in medians(current, baseline):
        if before > 0 and after > before * (1 + limit / 100):
            regressions.append(f"{name} {metric}: {before:.6g} -> {after:.6g} (+{(after / before - 1) * 100:.1f}%)")
    return regressions


def summarize(current, baseline, limit):
    """Markdown table of medians per workload; rows over the threshold are marked."""
    rows = {}
    for name, metric, before, after in medians(current, baseline):
        rows.setdefault(name, {})[metric] = (before, after)

    def cell(pair, scale, unit):
        if pair is None:
            return "n/a | n/a | n/a"
        before, after = pair
        delta = (after / before - 1) * 100 if before > 0 else 0.0
        flag = " **(!)**" if before > 0 and after > before * (1 + limit / 100) else ""
        return f"{before / scale:.4g} {unit} | {after / scale:.4g} {unit} | {delta:+.1f}%{flag}"

    lines = ["| Workload | Baseline time | Current time | Δ time | Baseline RSS | Current RSS | Δ RSS |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, metrics in rows.items():
        lines.append(f"| {name} | {cell(metrics.get('seconds'), 1, 's')} | {cell(metrics.get('peak_rss_bytes'), 1 << 20, 'MiB')} |")
    return "\n".join(lines) + "\n"


def sample(probe, operation, source, corpus, batch, env):
    args = [str(probe), operation, str(source), str(corpus), str(batch)]
    if sys.platform == "darwin":
        args = ["/usr/bin/time", "-l", *args]
    elif sys.platform.startswith("linux"):
        args = ["/usr/bin/time", "-v", *args]
    proc = subprocess.run(args, cwd=ROOT, env=env, check=True, capture_output=True, encoding="utf-8")
    result = json.loads(proc.stdout)
    rss = None
    if sys.platform == "darwin":
        match = re.search(r"(\d+)\s+maximum resident set size", proc.stderr)
        if match:
            rss = int(match[1])
    elif sys.platform.startswith("linux"):
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", proc.stderr)
        if match:
            rss = int(match[1]) * 1024
    if (sys.platform == "darwin" or sys.platform.startswith("linux")) and rss is None:
        raise RuntimeError("could not parse process peak RSS from /usr/bin/time")
    result["peak_rss_bytes"] = rss
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--compare", type=pathlib.Path)
    parser.add_argument("--threshold-percent", type=float, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--lines", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--host-label", help="name of the machine recorded in the report (informational, not compared)")
    parser.add_argument("--summary", type=pathlib.Path, help="with --compare: write a Markdown comparison table here")
    args = parser.parse_args()
    if args.summary and not args.compare:
        parser.error("--summary requires --compare")
    if min(args.repeats, args.lines, args.batch_size, args.threads) <= 0 or not 0 <= args.threshold_percent < float("inf"):
        parser.error("counts must be positive and threshold must be finite and nonnegative")
    fixtures = {model: ROOT / f"crates/morpheme/tests/data/hf/{model}.json" for model in MODELS}
    for path in fixtures.values():
        if not path.is_file():
            parser.error(f"missing {path}; run scripts/fetch-hf-fixtures.sh first")
    env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads))
    # Remove unrecorded compiler overrides to keep the release build reproducible.
    for key in list(env):
        if key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "CARGO_TARGET_DIR", "RUSTC") or key.startswith("CARGO_PROFILE_RELEASE_"):
            del env[key]
    subprocess.run(["cargo", "build", "--release", "--locked", "-p", "morpheme", "--example", "benchmark_probe"], cwd=ROOT, env=env, check=True)
    probe = ROOT / "target/release/examples" / ("benchmark_probe.exe" if os.name == "nt" else "benchmark_probe")
    cpu = command("sysctl", "-n", "machdep.cpu.brand_string") if sys.platform == "darwin" else platform.processor()
    if sys.platform.startswith("linux"):
        match = re.search(r"^model name\s*:\s*(.+)$", pathlib.Path("/proc/cpuinfo").read_text(), re.MULTILINE)
        cpu = match[1] if match else cpu
    report = {
        "schema_version": 1,
        "machine": {"system": platform.system(), "release": platform.release(), "arch": platform.machine(), "cpu": cpu, "cores": os.cpu_count()},
        "settings": {"rustc": command("rustc", "-Vv"), "workspace_manifest_sha256": fingerprint(ROOT / "Cargo.toml"), "package_manifest_sha256": fingerprint(ROOT / "crates/morpheme/Cargo.toml"), "cargo_config_sha256": fingerprint(ROOT / ".cargo/config.toml"), "lock_sha256": fingerprint(ROOT / "Cargo.lock"), "lines": args.lines, "batch_size": args.batch_size, "threads": args.threads, "corpus_generator": 1, "seed": 42, "profile": "release", "padding": False, "truncation": False, "special_tokens": True, "train_vocab": 4000},
        "revision": command("git", "rev-parse", "HEAD"),
        "host_label": args.host_label,
        "probe_sha256": fingerprint(ROOT / "crates/morpheme/examples/benchmark_probe.rs"),
        "dirty": bool(command("git", "status", "--porcelain")),
        "inputs": {"fixtures": {model: fingerprint(path) for model, path in fixtures.items()}},
        "results": {},
    }
    with tempfile.TemporaryDirectory(prefix="morpheme-bench-") as temporary:
        work = pathlib.Path(temporary)
        rng = random.Random(42)
        corpora = {}
        words = (ROOT / "examples/corpus.txt").read_text(encoding="utf-8").split()
        for name in ("repeated", "diverse"):
            path = work / f"{name}.txt"
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for _ in range(args.lines):
                    line = [rng.choice(words) if name == "repeated" else "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(12)) for _ in range(24)]
                    stream.write(" ".join(line) + " café 世界 😀\n")
            corpora[name] = path
        report["inputs"]["corpora"] = {name: fingerprint(path) for name, path in corpora.items()}
        workloads = []
        for model, source in fixtures.items():
            workloads.append((f"{model}/load", "load", source, corpora["repeated"]))
            for name, corpus in corpora.items():
                for operation in OPERATIONS:
                    workloads.append((f"{model}/{name}/{operation}", operation, source, corpus))
        for model in ("bpe", "wordpiece", "unigram"):
            workloads.append((f"{model}/train", "train", model, corpora["diverse"]))
        for name, operation, source, corpus in workloads:
            print(name, flush=True)
            samples = [sample(probe, operation, source, corpus, args.batch_size, env) for _ in range(args.repeats)]
            # Counters must agree across repetitions; timing/RSS can vary.
            counters = [{k: v for k, v in row.items() if k not in ("seconds", "peak_rss_bytes")} for row in samples]
            if any(row != counters[0] for row in counters):
                raise RuntimeError(f"nonrepeatable counters: {name}")
            report["results"][name] = {"samples": samples}
        for model in MODELS:
            for name in corpora:
                counts = [report["results"][f"{model}/{name}/{op}"]["samples"][0]["tokens"] for op in OPERATIONS]
                if len(set(counts)) != 1:
                    raise RuntimeError(f"sequential/batch token mismatch: {model}/{name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.compare:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        try:
            regressions = compare(report, baseline, args.threshold_percent)
            if args.summary:
                args.summary.write_text(summarize(report, baseline, args.threshold_percent), encoding="utf-8")
        except ValueError as error:
            # Exit 2 distinguishes "not comparable" from "regressed" for callers that record both.
            print(error, file=sys.stderr)
            if args.summary:
                args.summary.write_text(f"Not compared: {error}.\n", encoding="utf-8")
            return 2
        for line in regressions:
            print(line, file=sys.stderr)
        if regressions:
            return 1
        print("No regressions above the selected threshold")
    print(f"Recorded {len(report['results'])} workloads in {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
