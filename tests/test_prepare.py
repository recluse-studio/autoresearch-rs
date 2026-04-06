import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import prepare
from prepare import (
    Benchmark,
    apply_conservative_discriminative_fallback,
    assemble_variables,
    build_discriminative_retry_prompt,
    build_json_repair_prompt,
    build_benchmark_synthesis_prompt,
    default_benchmark_payload,
    get_active_benchmark,
    is_protected_path,
    load_benchmark,
    measurement_copies_task_baseline,
    measurement_needs_discrimination,
    sanitize_synthesized_benchmark_payload,
    score_variables_python,
    validate_measurement,
)


EXAMPLE_BENCHMARK = {
    "benchmark_id": "feat-001",
    "app_name": "Example App",
    "feature_name": "Opportunity brief generator",
    "app_summary": "Example benchmark for task-time scoring.",
    "paths_of_interest": ["src/"],
    "notes": ["Frozen inferred layer for the run."],
    "build_commands": ["cargo test"],
    "mandatory_criteria": [
        {"id": "wcag-2.1.1", "description": "Keyboard access passes."},
        {"id": "wcag-1.4.3", "description": "Contrast passes."},
    ],
    "taus": {"k": 0.28, "p": 1.1, "h": 0.4, "d": 1.2, "m": 1.35, "r": 0.5},
    "persona_weights": [
        {"persona_id": "professional-services-marketer", "w": 0.6},
        {"persona_id": "work-winner", "w": 0.4},
    ],
    "tasks": [
        {
            "persona_id": "professional-services-marketer",
            "task_id": "create-brief",
            "q": 1.0,
            "task_description": "Create a brief.",
            "baseline": {"k": 36, "p": 18, "h": 3, "d": 0, "m": 10, "r": 6},
        },
        {
            "persona_id": "work-winner",
            "task_id": "review-brief",
            "q": 1.0,
            "task_description": "Review a brief.",
            "baseline": {"k": 20, "p": 12, "h": 2, "d": 0, "m": 7, "r": 3},
        },
    ],
}


MEASUREMENT = {
    "status": "resolved",
    "g": 1,
    "cosmic": {"entries": 4, "exits": 3, "reads": 5, "writes": 2},
    "criteria": [
        {"id": "wcag-2.1.1", "result": "pass", "reason": "All controls reachable."},
        {"id": "wcag-1.4.3", "result": "pass", "reason": "Contrast meets threshold."},
    ],
    "feature_rows": [
        {
            "persona_id": "professional-services-marketer",
            "task_id": "create-brief",
            "feature": {"k": 20, "p": 9, "h": 2, "d": 0, "m": 7, "r": 4},
        },
        {
            "persona_id": "work-winner",
            "task_id": "review-brief",
            "feature": {"k": 12, "p": 7, "h": 1, "d": 0, "m": 5, "r": 2},
        },
    ],
}


DIMENSION_BENCHMARK = {
    "benchmark_id": "dim-001",
    "app_name": "Dimension App",
    "feature_name": "Dimension benchmark",
    "app_summary": "Example benchmark for dimension scoring.",
    "paths_of_interest": ["src/"],
    "notes": ["Frozen inferred layer for the run."],
    "build_commands": ["cargo test"],
    "mandatory_criteria": [
        {"id": "build-success", "description": "Build succeeds."},
    ],
    "dimensions": [
        {
            "id": "overall-clarity",
            "description": "Overall clarity improves.",
            "baseline_anchor": "Current baseline behavior.",
            "w": 1.0,
        }
    ],
    "taus": {"k": 0.28, "p": 1.1, "h": 0.4, "d": 1.2, "m": 1.35, "r": 0.5},
    "persona_weights": [],
    "tasks": [],
}


DIMENSION_MEASUREMENT_ZERO = {
    "status": "resolved",
    "g": 1,
    "cosmic": {"entries": 0, "exits": 0, "reads": 0, "writes": 0},
    "criteria": [
        {"id": "build-success", "result": "pass", "reason": "Build passed."},
    ],
    "dimension_rows": [
        {
            "id": "overall-clarity",
            "delta": 0.0,
            "confidence": 0.7,
            "observability": 0.7,
            "reason": "No call made.",
        }
    ],
}


class PrepareTests(unittest.TestCase):
    def test_default_benchmark_payload_uses_static_twenty_dimension_shape(self) -> None:
        payload = default_benchmark_payload()

        self.assertEqual(len(payload["dimensions"]), 20)
        self.assertEqual(payload["persona_weights"], [])
        self.assertEqual(payload["tasks"], [])
        self.assertEqual(payload["dimensions"][0]["id"], "core-task-effectiveness")
        self.assertEqual(payload["dimensions"][-1]["id"], "holistic-improvement")
        self.assertAlmostEqual(sum(row["w"] for row in payload["dimensions"]), 1.0)

    def test_benchmark_prompt_requests_fixed_twenty_dimension_scorecard(self) -> None:
        prompt = build_benchmark_synthesis_prompt()

        self.assertIn("Use all 20 fixed dimensions exactly as given.", prompt)
        self.assertIn("holistic-improvement", prompt)
        self.assertIn('"dimensions"', prompt)
        self.assertIn('"persona_weights": []', prompt)
        self.assertIn('"tasks": []', prompt)

    def test_load_benchmark_and_measurement_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "benchmark.json"
            path.write_text(json.dumps(EXAMPLE_BENCHMARK), encoding="utf-8")
            benchmark = load_benchmark(path)
        self.assertIsInstance(benchmark, Benchmark)
        measurement = validate_measurement(MEASUREMENT, benchmark)
        variables = assemble_variables(benchmark, measurement)
        self.assertEqual(variables["feature_id"], "feat-001")
        self.assertEqual(len(variables["tasks"]), 2)

    def test_python_scorer_matches_precedent_example(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "benchmark.json"
            path.write_text(json.dumps(EXAMPLE_BENCHMARK), encoding="utf-8")
            benchmark = load_benchmark(path)
        measurement = validate_measurement(MEASUREMENT, benchmark)
        variables = assemble_variables(benchmark, measurement)
        report = score_variables_python(variables)
        self.assertAlmostEqual(report["score"]["score_pct"], 40.31102825745683)
        self.assertAlmostEqual(report["score"]["weighted_delta_seconds"], 16.434)
        self.assertEqual(report["score"]["cosmic_function_points"], 14)

    def test_protected_paths_cover_fixed_harness_and_support(self) -> None:
        self.assertTrue(is_protected_path("prepare.py"))
        self.assertTrue(is_protected_path("support_docs/feature_assessment/framework.md"))
        self.assertTrue(is_protected_path("support_scripts/feature_assessor/src/main.rs"))
        self.assertFalse(is_protected_path("src/routes/app.rs"))

    def test_validate_measurement_normalizes_wrapped_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "benchmark.json"
            path.write_text(json.dumps(EXAMPLE_BENCHMARK), encoding="utf-8")
            benchmark = load_benchmark(path)

        payload = {
            "status": "resolved",
            "g": 1,
            "cosmic": {"entries": 4, "exits": 3, "reads": 5, "writes   ": 2},
            "criteria": [
                {"id": "wcag-2.1.1", "result": "pass", "reason": "All controls reachable."},
                {"id": "wcag-1.4.3", "result": "pass", "reason": "Contrast meets threshold."},
            ],
            "feature_rows": [
                {
                    "persona_id": "professional-services-marketer",
                    "task_id": "create-br ief",
                    "feature": {"k": 20, "p": 9, "h": 2, "d": 0, "m": 7, "r": 4},
                },
                {
                    "persona_id": "work-winner",
                    "task_id": "review-br ief",
                    "feature": {"k": 12, "p": 7, "h": 1, "d": 0, "m": 5, "r": 2},
                },
            ],
        }

        measurement = validate_measurement(payload, benchmark)
        self.assertEqual(measurement["cosmic"]["writes"], 2)
        self.assertEqual(measurement["feature_rows"][0]["task_id"], "create-brief")
        self.assertEqual(measurement["feature_rows"][1]["task_id"], "review-brief")

    def test_sanitize_synthesized_benchmark_payload_merges_static_anchors(self) -> None:
        payload = sanitize_synthesized_benchmark_payload(
            {
                "benchmark_id": " auto repo v1 ",
                "app_name": " Auto Repo ",
                "feature_name": " Core Flow ",
                "dimensions": [
                    {
                        "id": "core-task-effectiveness",
                        "description": "ignored",
                        "baseline_anchor": "Current repo baseline for the main job is already useful but still fairly manual.",
                        "w": 0.99,
                    },
                    {
                        "id": "holistic-improvement",
                        "description": "ignored",
                        "baseline_anchor": "Baseline currently hangs together, but still feels more adjacent than unified.",
                        "w": 0.01,
                    },
                ],
            }
        )

        self.assertEqual(payload["benchmark_id"], "autorepov1")
        self.assertEqual(len(payload["dimensions"]), 20)
        self.assertAlmostEqual(sum(row["w"] for row in payload["dimensions"]), 1.0)
        self.assertEqual(payload["dimensions"][0]["id"], "core-task-effectiveness")
        self.assertIn("already useful but still fairly manual", payload["dimensions"][0]["baseline_anchor"])
        self.assertEqual(payload["dimensions"][-1]["id"], "holistic-improvement")
        self.assertIn("adjacent than unified", payload["dimensions"][-1]["baseline_anchor"])

    def test_get_active_benchmark_resynthesizes_nonstatic_cache(self) -> None:
        dimension_benchmark = {
            "benchmark_id": "dim-bench",
            "app_name": "Dimension App",
            "feature_name": "Dimension benchmark",
            "app_summary": "Dimension-based benchmark.",
            "paths_of_interest": ["src/"],
            "notes": ["Dimension cache."],
            "build_commands": ["cargo test"],
            "mandatory_criteria": [
                {"id": "build-success", "description": "Build succeeds."},
            ],
            "dimensions": [
                {
                    "id": "overall-improvement",
                    "description": "Overall improvement.",
                    "baseline_anchor": "Current baseline behavior.",
                    "w": 1.0,
                }
            ],
            "taus": {"k": 0.28, "p": 1.1, "h": 0.4, "d": 1.2, "m": 1.35, "r": 0.5},
            "persona_weights": [],
            "tasks": [],
        }

        with tempfile.TemporaryDirectory() as tempdir:
            tempdir_path = Path(tempdir)
            cache_path = tempdir_path / "frozen-benchmark.json"
            override_path = tempdir_path / "benchmark.json"
            cache_path.write_text(json.dumps(dimension_benchmark), encoding="utf-8")
            synthesized_path = tempdir_path / "synthesized.json"
            synthesized_path.write_text(json.dumps(default_benchmark_payload()), encoding="utf-8")
            synthesized_benchmark = load_benchmark(synthesized_path)

            with mock.patch.object(prepare, "CACHED_BENCHMARK_PATH", cache_path), mock.patch.object(
                prepare, "BENCHMARK_PATH", override_path
            ), mock.patch.object(prepare, "synthesize_benchmark", return_value=synthesized_benchmark):
                benchmark, source = get_active_benchmark(allow_synthesis=True)

        self.assertEqual(source, "synthesized")
        self.assertEqual(len(benchmark.dimensions), 20)

    def test_measurement_needs_discrimination_for_all_zero_dimension_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "benchmark.json"
            path.write_text(json.dumps(DIMENSION_BENCHMARK), encoding="utf-8")
            benchmark = load_benchmark(path)

        measurement = validate_measurement(DIMENSION_MEASUREMENT_ZERO, benchmark)
        self.assertTrue(
            measurement_needs_discrimination(
                benchmark,
                measurement,
                require_nonzero_delta=True,
            )
        )

    def test_apply_conservative_discriminative_fallback_for_dimensions_produces_signed_score(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "benchmark.json"
            path.write_text(json.dumps(DIMENSION_BENCHMARK), encoding="utf-8")
            benchmark = load_benchmark(path)

        measurement = validate_measurement(DIMENSION_MEASUREMENT_ZERO, benchmark)
        fallback = apply_conservative_discriminative_fallback(
            benchmark,
            measurement,
            reason="Evaluator would not discriminate.",
        )
        self.assertLess(fallback["dimension_rows"][0]["delta"], 0.0)
        variables = assemble_variables(benchmark, fallback)
        report = score_variables_python(variables)
        self.assertLess(report["score"]["score_pct"], 0.0)

    def test_measurement_copies_task_baseline_detects_baseline_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "benchmark.json"
            path.write_text(json.dumps(EXAMPLE_BENCHMARK), encoding="utf-8")
            benchmark = load_benchmark(path)

        copied = {
            "status": "resolved",
            "g": 1,
            "cosmic": {"entries": 0, "exits": 0, "reads": 0, "writes": 0},
            "criteria": [],
            "feature_rows": [
                {
                    "persona_id": task.persona_id,
                    "task_id": task.task_id,
                    "feature": task.baseline.to_dict(),
                }
                for task in benchmark.tasks
            ],
        }
        measurement = validate_measurement(copied, benchmark)
        self.assertTrue(measurement_copies_task_baseline(benchmark, measurement))

    def test_build_discriminative_retry_prompt_for_dimensions_forbids_all_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "benchmark.json"
            path.write_text(json.dumps(DIMENSION_BENCHMARK), encoding="utf-8")
            benchmark = load_benchmark(path)
        prompt = build_discriminative_retry_prompt(benchmark, [{"command": "cargo test", "ok": True}])
        self.assertIn("all-zero dimension result is invalid", prompt)
        self.assertIn("At least one dimension row must carry a non-zero signed delta", prompt)

    def test_build_json_repair_prompt_demands_json_only(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "benchmark.json"
            path.write_text(json.dumps(DIMENSION_BENCHMARK), encoding="utf-8")
            benchmark = load_benchmark(path)
        prompt = build_json_repair_prompt(
            benchmark,
            [{"command": "cargo test", "ok": True}],
            prior_error="Failed to locate a JSON object in the evaluator output",
            final_pass=True,
        )
        self.assertIn("Return exactly one JSON object and nothing else.", prompt)
        self.assertIn("Failure reason: Failed to locate a JSON object in the evaluator output", prompt)


if __name__ == "__main__":
    unittest.main()
