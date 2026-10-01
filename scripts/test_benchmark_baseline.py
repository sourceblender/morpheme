#!/usr/bin/env python3
"""Deterministic checks for the performance comparison gate, without timing tests."""
import copy
import unittest

from benchmark_baseline import compare, summarize


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "schema_version": 1, "machine": {"cpu": "test"}, "probe_sha256": "probe-a",
            "settings": {"threads": 4, "package_manifest_sha256": "package-a"}, "inputs": {"sha256": "abc"},
            "results": {"encode": {"samples": [
                {"seconds": 1, "peak_rss_bytes": 100},
                {"seconds": 1, "peak_rss_bytes": 100},
                {"seconds": 100, "peak_rss_bytes": 10000},
            ]}},
        }

    def test_identical_and_outliers_use_median(self):
        self.assertEqual(compare(self.baseline, self.baseline, 20), [])
        current = copy.deepcopy(self.baseline)
        current["results"]["encode"]["samples"][2]["seconds"] = 10000
        self.assertEqual(compare(current, self.baseline, 20), [])

    def test_time_and_memory_regressions(self):
        current = copy.deepcopy(self.baseline)
        for row in current["results"]["encode"]["samples"]:
            row.update(seconds=2, peak_rss_bytes=200)
        self.assertEqual(len(compare(current, self.baseline, 20)), 2)

    def test_incompatible_measurements_rejected(self):
        for field in ("schema_version", "machine", "settings", "inputs", "probe_sha256"):
            current = copy.deepcopy(self.baseline)
            current[field] = None
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare(current, self.baseline, 20)
        current = copy.deepcopy(self.baseline)
        current["results"]["extra"] = current["results"]["encode"]
        with self.assertRaises(ValueError):
            compare(current, self.baseline, 20)

    def test_package_manifest_mismatch(self):
        current = copy.deepcopy(self.baseline)
        current["settings"]["package_manifest_sha256"] = "package-b"
        with self.assertRaisesRegex(ValueError, "settings differs"):
            compare(current, self.baseline, 20)

    def test_missing_rss(self):
        current = copy.deepcopy(self.baseline)
        for row in current["results"]["encode"]["samples"]:
            row["peak_rss_bytes"] = None
        with self.assertRaises(ValueError):
            compare(current, self.baseline, 20)
        self.assertEqual(compare(current, current, 20), [])


class SummaryTests(unittest.TestCase):
    setUp = ComparisonTests.setUp

    def test_identical_rows_are_unflagged(self):
        table = summarize(self.baseline, self.baseline, 20)
        lines = table.strip().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("| Workload |"))
        self.assertIn("| encode | 1 s | 1 s | +0.0% | ", lines[2])
        self.assertNotIn("(!)", table)

    def test_regressions_are_flagged_consistently_with_compare(self):
        current = copy.deepcopy(self.baseline)
        for row in current["results"]["encode"]["samples"]:
            row["seconds"] = 2
        table = summarize(current, self.baseline, 20)
        self.assertIn("| 1 s | 2 s | +100.0% **(!)** |", table)
        self.assertEqual(table.count("(!)"), len(compare(current, self.baseline, 20)))
        self.assertNotIn("(!)", summarize(current, self.baseline, 100))

    def test_missing_rss_is_reported_not_compared(self):
        current = copy.deepcopy(self.baseline)
        for row in current["results"]["encode"]["samples"]:
            row["peak_rss_bytes"] = None
        self.assertIn("| n/a | n/a | n/a |", summarize(current, current, 20))
        with self.assertRaises(ValueError):
            summarize(current, self.baseline, 20)

    def test_incompatible_baseline_rejected(self):
        current = copy.deepcopy(self.baseline)
        current["probe_sha256"] = "probe-b"
        with self.assertRaisesRegex(ValueError, "probe_sha256 differs"):
            summarize(current, self.baseline, 20)


if __name__ == "__main__":
    unittest.main()
