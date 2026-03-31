import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import prepare
from prepare import (
    Benchmark,
    assemble_variables,
    build_benchmark_synthesis_prompt,
    default_benchmark_payload,
    get_active_benchmark,
    is_protected_path,
    load_benchmark,
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


class PrepareTests(unittest.TestCase):
    def test_default_benchmark_payload_prefers_task_scoring_shape(self) -> None:
        payload = default_benchmark_payload()

        self.assertEqual(payload["dimensions"], [])
        self.assertEqual(len(payload["persona_weights"]), 1)
        self.assertEqual(len(payload["tasks"]), 1)
        self.assertEqual(payload["tasks"][0]["persona_id"], payload["persona_weights"][0]["persona_id"])

    def test_benchmark_prompt_requests_personas_and_tasks(self) -> None:
        prompt = build_benchmark_synthesis_prompt()

        self.assertIn("Prefer 1 to 3 personas and 2 to 6 tasks total.", prompt)
        self.assertIn('"persona_weights"', prompt)
        self.assertIn('"tasks"', prompt)
        self.assertIn('"dimensions": []', prompt)

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

    def test_sanitize_synthesized_benchmark_payload_normalizes_weights(self) -> None:
        payload = sanitize_synthesized_benchmark_payload(
            {
                "benchmark_id": " auto repo v1 ",
                "app_name": " Auto Repo ",
                "feature_name": " Core Flow ",
                "persona_weights": [
                    {"persona_id": "primary-user", "w": 1.0},
                    {"persona_id": "secondary-user", "w": 0.6},
                ],
                "tasks": [
                    {
                        "persona_id": "primary-user",
                        "task_id": "complete-core-f low",
                        "q": 0.75,
                        "task_description": " Complete the main flow. ",
                        "baseline": {"k": 10, "p": 5, "h": 1, "d": 0, "m": 2, "r": 1},
                    },
                    {
                        "persona_id": "secondary-user",
                        "task_id": "review-flow",
                        "q": 0.25,
                        "task_description": "Review the flow.",
                        "baseline": {"k": 6, "p": 2, "h": 1, "d": 0, "m": 1, "r": 1},
                    },
                ],
            }
        )
        persona_sum = sum(row["w"] for row in payload["persona_weights"])
        primary_tasks = [row for row in payload["tasks"] if row["persona_id"] == "primary-user"]
        secondary_tasks = [row for row in payload["tasks"] if row["persona_id"] == "secondary-user"]

        self.assertAlmostEqual(persona_sum, 1.0)
        self.assertEqual(payload["benchmark_id"], "autorepov1")
        self.assertEqual(primary_tasks[0]["task_id"], "complete-core-flow")
        self.assertAlmostEqual(sum(row["q"] for row in primary_tasks), 1.0)
        self.assertAlmostEqual(sum(row["q"] for row in secondary_tasks), 1.0)

    def test_get_active_benchmark_resynthesizes_dimension_cache(self) -> None:
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
            synthesized_path = tempdir_path / "synthesized.json"

            cache_path.write_text(json.dumps(dimension_benchmark), encoding="utf-8")
            synthesized_path.write_text(json.dumps(EXAMPLE_BENCHMARK), encoding="utf-8")
            synthesized_benchmark = load_benchmark(synthesized_path)

            with mock.patch.object(prepare, "CACHED_BENCHMARK_PATH", cache_path), mock.patch.object(
                prepare, "BENCHMARK_PATH", override_path
            ), mock.patch.object(prepare, "synthesize_benchmark", return_value=synthesized_benchmark):
                benchmark, source = get_active_benchmark(allow_synthesis=True)

        self.assertEqual(source, "synthesized")
        self.assertEqual(len(benchmark.tasks), 2)
        self.assertFalse(cache_path.exists())


if __name__ == "__main__":
    unittest.main()
