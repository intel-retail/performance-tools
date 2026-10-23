#
# Apache v2 license
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for Web UI benchmark event parsing and capture."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark_signage_web_ui as web_ui


class TestWebUiEventHelpers(unittest.TestCase):

    def test_parse_log_line_recognizes_markers_and_ignores_bad_events(self):
        parsed = web_ui.parse_log_line("2026-01-01T00:00:00Z WEB_UI_SELECTION {\"frame_id\":1}")
        self.assertEqual(parsed, (web_ui.CALLBACK_MARKER, "2026-01-01T00:00:00Z", {"frame_id": 1}))
        self.assertIsNone(web_ui.parse_log_line("unrelated output"))
        self.assertIsNone(web_ui.parse_log_line("ts WEB_UI_SELECTION {bad}"))

    def test_combine_events_attaches_selector_queue_and_cooldown_data(self):
        events = [
            (web_ui.CALLBACK_MARKER, "t1", {
                "frame_id": 1, "callback_total_ms": 2, "recency_frames": 8,
                "confidence_threshold": 0.5,
            }),
            (web_ui.SELECTOR_MARKER, "t2", {
                "frame_id": 1, "selector_total_ms": 3, "candidate_count": 2,
                "product_key_normalization_ms": 1, "csv_lookup_ms": 1,
                "product_deduplication_ms": 1,
            }),
            (web_ui.AD_TRACE_MARKER, "t3", {"frame_id": 1, "queue_wait_ms": 4}),
            (web_ui.COOLDOWN_MARKER, "t4", {
                "frame_id": 1, "queue_wait_ms": 5, "cooldown_remaining_ms": 6,
            }),
            (web_ui.CALLBACK_MARKER, "t5", {"frame_id": None, "callback_total_ms": 1}),
        ]

        frames = web_ui.combine_events(events)

        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0]["total_selection_logic_ms"], 5)
        self.assertEqual(frames[0]["queue_wait_ms"], 5)
        self.assertTrue(frames[0]["cooldown_suppressed"])
        self.assertEqual(frames[0]["cooldown_remaining_ms"], 6)
        self.assertIsNone(frames[1]["cooldown_suppressed"])

    def test_combine_ad_events_merges_served_and_rendered_timings(self):
        events = [
            (web_ui.AD_TRACE_MARKER, "t1", {
                "ad_id": 7, "path": "dynamic", "frame_id": 1, "ase_query_ms": 2,
                "frame_to_ad_ready_ms": 3,
            }),
            (web_ui.AD_SERVED_MARKER, "t2", {
                "ad_id": 7, "delivery_id": "request-a", "ready_to_served_ms": 3,
            }),
            (web_ui.AD_RENDERED_MARKER, "t3", {
                "ad_id": 7, "delivery_id": "request-a",
                "browser_fetch_ms": 4, "browser_render_ms": 5,
            }),
            (web_ui.AD_TRACE_MARKER, "t4", {"ad_id": 8, "path": "predefined"}),
        ]

        ads = web_ui.combine_ad_events(events)

        self.assertEqual(len(ads), 2)
        self.assertIsNone(ads[0]["frame_to_render_ms"])
        self.assertEqual(ads[0]["matched_deliveries"], 1)
        self.assertEqual(ads[0]["unmatched_served_deliveries"], 0)
        self.assertEqual(ads[0]["ready_to_served_ms"], 3)
        self.assertEqual(ads[0]["browser_fetch_ms"], 4)
        self.assertEqual(ads[0]["browser_render_ms"], 5)
        self.assertEqual(ads[0]["path"], "dynamic")
        self.assertIsNone(ads[1]["frame_to_render_ms"])

    def test_combine_ad_events_correlates_client_deliveries_before_averaging(self):
        events = [
            (web_ui.AD_TRACE_MARKER, "t0", {
                "ad_id": 7, "path": "dynamic", "frame_to_ad_ready_ms": 10,
            }),
            (web_ui.AD_SERVED_MARKER, "t1", {
                "ad_id": 7, "delivery_id": "client-a", "ready_to_served_ms": 2,
            }),
            (web_ui.AD_SERVED_MARKER, "t2", {
                "ad_id": 7, "delivery_id": "client-b", "ready_to_served_ms": 20,
            }),
            (web_ui.AD_SERVED_MARKER, "t3", {
                "ad_id": 7, "delivery_id": "client-c", "ready_to_served_ms": 30,
            }),
            (web_ui.AD_RENDERED_MARKER, "t4", {
                "ad_id": 7, "delivery_id": "client-b",
                "browser_fetch_ms": 100, "browser_render_ms": 10,
            }),
            (web_ui.AD_RENDERED_MARKER, "t5", {
                "ad_id": 7, "delivery_id": "client-a",
                "browser_fetch_ms": 200, "browser_render_ms": 20,
            }),
            (web_ui.AD_RENDERED_MARKER, "t6", {
                "ad_id": 7, "delivery_id": "orphan",
                "browser_fetch_ms": 300, "browser_render_ms": 30,
            }),
        ]

        ad = web_ui.combine_ad_events(events)[0]

        self.assertEqual(ad["served_deliveries"], 3)
        self.assertEqual(ad["matched_deliveries"], 2)
        self.assertEqual(ad["unmatched_served_deliveries"], 1)
        self.assertEqual(ad["unmatched_rendered_deliveries"], 1)
        self.assertEqual(ad["ready_to_served_ms"], 17.333)
        self.assertEqual(ad["browser_fetch_ms"], 150)
        self.assertEqual(ad["browser_render_ms"], 15)
        self.assertIsNone(ad["frame_to_render_ms"])

    def test_combine_ad_events_does_not_pair_legacy_events_without_delivery_ids(self):
        events = [
            (web_ui.AD_TRACE_MARKER, "t0", {
                "ad_id": 7, "path": "dynamic", "frame_to_ad_ready_ms": 10,
            }),
            (web_ui.AD_SERVED_MARKER, "t1", {
                "ad_id": 7, "ready_to_served_ms": 2,
            }),
            (web_ui.AD_RENDERED_MARKER, "t2", {
                "ad_id": 7, "browser_fetch_ms": 4, "browser_render_ms": 5,
            }),
        ]

        ad = web_ui.combine_ad_events(events)[0]

        self.assertEqual(ad["matched_deliveries"], 0)
        self.assertEqual(ad["unmatched_served_deliveries"], 1)
        self.assertEqual(ad["unmatched_rendered_deliveries"], 1)
        self.assertIsNone(ad["frame_to_render_ms"])


class TestWebUiRun(unittest.TestCase):

    @staticmethod
    def event_line():
        data = [
            (web_ui.CALLBACK_MARKER, {"frame_id": 1, "callback_total_ms": 2,
                                      "recency_frames": 5, "confidence_threshold": 0.5}),
            (web_ui.SELECTOR_MARKER, {"frame_id": 1, "selector_total_ms": 3}),
            (web_ui.AD_TRACE_MARKER, {
                "ad_id": 9, "frame_id": 1, "path": "dynamic",
                "frame_to_ad_ready_ms": 10, "ase_query_ms": 4, "aig_generation_ms": 6,
            }),
            (web_ui.AD_SERVED_MARKER, {
                "ad_id": 9, "delivery_id": "request-9", "ready_to_served_ms": 2,
            }),
            (web_ui.AD_RENDERED_MARKER, {
                "ad_id": 9, "delivery_id": "request-9",
                "browser_fetch_ms": 3, "browser_render_ms": 4,
            }),
        ]
        return b"".join(
            f"2026-01-01T00:00:00Z {marker}{json.dumps(payload)}\n".encode()
            for marker, payload in data
        )

    def test_run_writes_summary_frames_and_ads(self):
        with tempfile.TemporaryDirectory() as directory:
            process = mock.Mock(stdout=mock.Mock())
            with (
                mock.patch.object(web_ui.subprocess, "check_output", return_value="true"),
                mock.patch.object(web_ui.subprocess, "Popen", return_value=process),
                mock.patch.object(web_ui.select, "select", return_value=([process.stdout], [], [])),
                mock.patch.object(web_ui.os, "read", return_value=self.event_line()),
                mock.patch.object(web_ui.time, "monotonic", side_effect=[0, 0, 1, 1]),
                mock.patch("builtins.print"),
            ):
                web_ui.run(0.5, Path(directory))

            summary_path = next(
                path for path in Path(directory).glob("web_ui_benchmark_*.json")
                if "_frames" not in path.name and "_ads" not in path.name
            )
            frame_path = next(Path(directory).glob("web_ui_benchmark_*_frames.json"))
            ad_path = next(Path(directory).glob("web_ui_benchmark_*_ads.json"))
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["frames"], 1)
            self.assertEqual(summary["ads"], 1)
            self.assertEqual(summary["ads_dynamic"], 1)
            self.assertEqual(json.loads(frame_path.read_text())[0]["selector_total_ms"], 3)
            ad = json.loads(ad_path.read_text())[0]
            self.assertIsNone(ad["frame_to_render_ms"])
            self.assertEqual(ad["browser_fetch_ms"], 3)
            self.assertEqual(ad["browser_render_ms"], 4)
            self.assertEqual(summary["ad_delivery_matches"], 1)
            self.assertEqual(summary["ad_unmatched_served_deliveries"], 0)
            self.assertEqual(summary["ad_unmatched_rendered_deliveries"], 0)
            process.terminate.assert_called_once()
            process.communicate.assert_called_once_with(timeout=3)

    def test_run_reports_stopped_container_and_missing_selection_events(self):
        with mock.patch.object(web_ui.subprocess, "check_output", return_value="false"):
            with self.assertRaisesRegex(RuntimeError, "web-ui is not running"):
                web_ui.run(1, Path(tempfile.gettempdir()))

        process = mock.Mock(stdout=mock.Mock())
        with (
            mock.patch.object(web_ui.subprocess, "check_output", return_value="true"),
            mock.patch.object(web_ui.subprocess, "Popen", return_value=process),
            mock.patch.object(web_ui.select, "select", return_value=([], [], [])),
            mock.patch.object(web_ui.time, "monotonic", side_effect=[0, 1]),
        ):
            with self.assertRaisesRegex(RuntimeError, "No WEB_UI_SELECTION frames"):
                web_ui.run(0.5, Path(tempfile.gettempdir()))
        process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
