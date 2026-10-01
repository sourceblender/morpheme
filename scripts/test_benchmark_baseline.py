#!/usr/bin/env python3
"""Deterministic checks for the performance comparison gate, without timing tests."""
import copy
import unittest

from benchmark_baseline import compare


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


if __name__ == "__main__":
    unittest.main()
