"""
Preparation, benchmarking, and scoring utilities for app-focused autoresearch.

Usage:
    python prepare.py check
    python prepare.py score
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field, is_dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


HARNESS_ROOT = Path(__file__).resolve().parent
PROGRAM_PATH = HARNESS_ROOT / "program.md"
DEFAULT_BACKEND = "auto"
DEFAULT_TOOL = "copilot"
DEFAULT_COPILOT_MODEL = ""
AGENT_PROMPT_INLINE_LIMIT = 28_000
COPILOT_TIMEOUT_SECONDS = 300
BUILD_COMMAND_TIMEOUT_SECONDS = 300
BENCHMARK_SYNTHESIS_TIMEOUT_SECONDS = 120
EVALUATOR_TIMEOUT_SECONDS = 90
SCORER_TIMEOUT_SECONDS = 60
ALL_ZERO_RETRY_TIMEOUT_SECONDS = 60
EPSILON = 1e-9
MIN_DIMENSION_EFFECTIVE_WEIGHT = 0.35
DIMENSION_WEIGHT_EXPONENT = 0.5
DIMENSION_EVIDENCE_EXPONENT = 0.5
RESCUE_SIGNED_DELTA = 0.01
RESCUE_MIN_CONFIDENCE = 0.2
RESCUE_MIN_OBSERVABILITY = 0.2

PROTECTED_FILES = {
    "README.md",
    "benchmark.json",
    "prepare.py",
    "program.md",
    "train.py",
}

STATIC_MANDATORY_CRITERIA = [
    {
        "id": "build-success",
        "description": "All configured build and test commands complete successfully.",
    },
    {
        "id": "core-purpose-intact",
        "description": "The app remains coherent, usable, and recognizably aligned with its core purpose.",
    },
    {
        "id": "data-and-state-safe",
        "description": "Changes must not obviously damage saved data, continuity, or state restoration.",
    },
]

STATIC_DIMENSION_SPECS = [
    {"id": "core-task-effectiveness", "description": "How well the app helps users accomplish its main job.", "w": 0.10, "default_anchor": "Baseline frozen at run start for how well the app helps users accomplish its main job."},
    {"id": "workflow-efficiency", "description": "How little friction, repetition, and wasted effort the main workflow requires.", "w": 0.08, "default_anchor": "Baseline frozen at run start for workflow friction, repetition, and effort."},
    {"id": "navigation-clarity", "description": "How easy it is to know where to go and how to move through the app.", "w": 0.05, "default_anchor": "Baseline frozen at run start for wayfinding, movement, and navigation clarity."},
    {"id": "information-hierarchy", "description": "How clearly the app emphasizes the most important information first.", "w": 0.05, "default_anchor": "Baseline frozen at run start for how well the interface emphasizes what matters most."},
    {"id": "scanability", "description": "How quickly a user can scan a screen and understand what matters.", "w": 0.05, "default_anchor": "Baseline frozen at run start for how quickly important information can be scanned and understood."},
    {"id": "interaction-clarity", "description": "How obvious the app's controls, actions, and next steps are.", "w": 0.05, "default_anchor": "Baseline frozen at run start for how obvious controls, actions, and next steps feel."},
    {"id": "input-ergonomics", "description": "How easy forms, editing, and data entry feel in real use.", "w": 0.05, "default_anchor": "Baseline frozen at run start for form, editing, and data-entry ergonomics."},
    {"id": "feedback-and-system-response", "description": "How clearly the app communicates status, progress, success, failure, and system response.", "w": 0.04, "default_anchor": "Baseline frozen at run start for status visibility, progress feedback, and response clarity."},
    {"id": "visual-coherence", "description": "How consistent, intentional, and well-composed the visual system feels.", "w": 0.04, "default_anchor": "Baseline frozen at run start for visual consistency, composition, and design coherence."},
    {"id": "brand-expression", "description": "How well the product's tone, personality, and design character come through.", "w": 0.03, "default_anchor": "Baseline frozen at run start for how clearly the product expresses its intended tone and character."},
    {"id": "accessibility-and-inclusion", "description": "How usable the app is across different abilities, constraints, and environments.", "w": 0.04, "default_anchor": "Baseline frozen at run start for accessibility, inclusion, and adaptability to different user constraints."},
    {"id": "performance-and-responsiveness", "description": "How fast, responsive, and lightweight the app feels.", "w": 0.06, "default_anchor": "Baseline frozen at run start for app speed, responsiveness, and perceived performance."},
    {"id": "reliability-and-stability", "description": "How consistently the app works without errors, crashes, or broken states.", "w": 0.06, "default_anchor": "Baseline frozen at run start for runtime stability and freedom from broken behavior."},
    {"id": "data-integrity-and-safety", "description": "How safely and correctly the app stores, retrieves, and preserves information.", "w": 0.06, "default_anchor": "Baseline frozen at run start for correctness, safety, and trustworthiness of stored information."},
    {"id": "state-continuity-and-resume", "description": "How well the app preserves context across restart, reload, interruption, and resume.", "w": 0.05, "default_anchor": "Baseline frozen at run start for preserving context and resuming work after interruption or restart."},
    {"id": "feature-usefulness", "description": "How valuable and meaningful the available capabilities actually are.", "w": 0.06, "default_anchor": "Baseline frozen at run start for how useful and meaningful the current capabilities feel."},
    {"id": "ai-usefulness", "description": "Whether any AI capabilities provide meaningful help instead of noise.", "w": 0.04, "default_anchor": "Baseline frozen at run start for whether AI contributes real usefulness rather than decorative or noisy behavior."},
    {"id": "ai-integration-quality", "description": "How naturally AI fits into the workflow, interface, and product identity.", "w": 0.03, "default_anchor": "Baseline frozen at run start for how naturally AI is integrated into workflow, interface, and product identity."},
    {"id": "ai-trustworthiness-and-restraint", "description": "Whether AI behaves in a dependable, legible, non-gimmicky way and knows when not to intrude.", "w": 0.02, "default_anchor": "Baseline frozen at run start for whether AI feels dependable, legible, and appropriately restrained."},
    {"id": "holistic-improvement", "description": "Whether the change forms one coherent, higher-quality improvement instead of many scattered small tweaks.", "w": 0.04, "default_anchor": "Baseline frozen at run start for how well the product hangs together as a coherent whole rather than a pile of disconnected tweaks."},
]

STATIC_DIMENSION_IDS = {row["id"] for row in STATIC_DIMENSION_SPECS}


def discover_target_root() -> Path:
    override = os.environ.get("AUTORESEARCH_TARGET_ROOT", "").strip()
    if override:
        return Path(override).resolve()

    cwd = Path.cwd().resolve()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return cwd

    root_text = completed.stdout.strip()
    return Path(root_text).resolve() if root_text else cwd


ROOT = discover_target_root()
CACHE_DIR = ROOT / ".autoresearch-cache"
BENCHMARK_PATH = ROOT / "benchmark.json"
CACHED_BENCHMARK_PATH = CACHE_DIR / "frozen-benchmark.json"
LAST_VARIABLES_PATH = CACHE_DIR / "last-variables.json"
LAST_SCORE_PATH = CACHE_DIR / "last-score.json"
LAST_BUILD_RESULTS_PATH = CACHE_DIR / "last-build-results.json"
LAST_MEASUREMENT_PATH = CACHE_DIR / "last-measurement.json"
LAST_COPILOT_USAGE_PATH = CACHE_DIR / "last-copilot-usage.json"
LAST_BENCHMARK_MESSAGE = CACHE_DIR / "last-benchmark-message.txt"
LAST_EVALUATOR_MESSAGE = CACHE_DIR / "last-evaluator-message.txt"
RUST_ASSESSOR_MANIFEST = ROOT / "support_scripts" / "feature_assessor" / "Cargo.toml"
PROTECTED_PREFIXES = (
    ".git/",
    "karpathy-files/",
    "support_docs/",
    "support_scripts/",
)
INTERNAL_STATE_FILES = {
    "REFLECTION.md",
    "results.tsv",
    "TODO.md",
}
INTERNAL_STATE_PREFIXES = (
    ".autoresearch-cache/",
)


def discover_runner_root() -> Optional[Path]:
    candidates = [
        ROOT / "support_scripts" / "runner",
        HARNESS_ROOT.parent / "runner",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def prepend_env_path(paths: Sequence[Path]) -> None:
    current_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    additions: List[str] = []
    seen = set(current_entries)
    for path in paths:
        if not path or not path.exists():
            continue
        text = str(path)
        if text in seen:
            continue
        additions.append(text)
        seen.add(text)
    if additions:
        os.environ["PATH"] = os.pathsep.join(additions + current_entries)


def find_first_executable(root: Path, candidate_names: Sequence[str]) -> Optional[Path]:
    if not root.exists():
        return None
    for name in candidate_names:
        match = next(root.rglob(name), None)
        if match:
            return match
    return None


def configure_local_tooling_environment() -> None:
    runner_root = discover_runner_root()
    if not runner_root:
        return

    toolchain_root = runner_root / ".toolchain"
    path_entries: List[Path] = []

    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        npm_bin_dir = Path(appdata) / "npm"
        if npm_bin_dir.is_dir():
            path_entries.append(npm_bin_dir)

    node_root = toolchain_root / "node"
    if (node_root / "node.exe").exists():
        path_entries.append(node_root)

    cargo_home = toolchain_root / "cargo"
    if cargo_home.is_dir():
        os.environ.setdefault("CARGO_HOME", str(cargo_home))
        os.environ.setdefault("CARGO_TARGET_DIR", str(CACHE_DIR / "cargo-target"))
        cargo_bin = cargo_home / "bin"
        if cargo_bin.is_dir():
            path_entries.append(cargo_bin)

    rustup_home = toolchain_root / "rustup"
    if rustup_home.is_dir():
        os.environ.setdefault("RUSTUP_HOME", str(rustup_home))
        toolchains_root = rustup_home / "toolchains"
        preferred_toolchain = toolchains_root / "stable-x86_64-pc-windows-gnullvm"
        selected_toolchain = preferred_toolchain if preferred_toolchain.is_dir() else None
        if selected_toolchain is None and toolchains_root.is_dir():
            for candidate in sorted(toolchains_root.iterdir()):
                if (candidate / "bin" / "cargo.exe").exists():
                    selected_toolchain = candidate
                    break

        if selected_toolchain is not None:
            os.environ.setdefault("RUSTUP_TOOLCHAIN", selected_toolchain.name)
            toolchain_bin = selected_toolchain / "bin"
            rustlib_bin = selected_toolchain / "lib" / "rustlib" / "x86_64-pc-windows-gnullvm" / "bin"
            if toolchain_bin.is_dir():
                path_entries.append(toolchain_bin)
            if rustlib_bin.is_dir():
                path_entries.append(rustlib_bin)

    llvm_root = toolchain_root / "llvm-mingw"
    clang_exe = find_first_executable(llvm_root, ("x86_64-w64-mingw32-clang.exe", "clang.exe"))
    ar_exe = find_first_executable(llvm_root, ("llvm-ar.exe", "ar.exe"))
    if clang_exe:
        path_entries.append(clang_exe.parent)
        os.environ.setdefault("CARGO_TARGET_X86_64_PC_WINDOWS_GNULLVM_LINKER", str(clang_exe))
        os.environ.setdefault("CC_x86_64_pc_windows_gnullvm", str(clang_exe))
        os.environ.setdefault("CC_x86_64-pc-windows-gnullvm", str(clang_exe))
    if ar_exe:
        os.environ.setdefault("AR_x86_64_pc_windows_gnullvm", str(ar_exe))
        os.environ.setdefault("AR_x86_64-pc-windows-gnullvm", str(ar_exe))

    prepend_env_path(path_entries)


configure_local_tooling_environment()


def bounded_timeout(
    timeout_seconds: int,
    *,
    deadline_monotonic: Optional[float] = None,
    label: str = "operation",
) -> int:
    effective = max(1, int(timeout_seconds))
    if deadline_monotonic is None:
        return effective
    remaining = deadline_monotonic - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"{label} exceeded the remaining iteration deadline")
    return max(1, min(effective, int(math.ceil(remaining))))


def adjusted_dimension_weights(weights: Sequence[float]) -> List[float]:
    if not weights:
        return []
    flattened = [max(0.0, float(weight)) ** DIMENSION_WEIGHT_EXPONENT for weight in weights]
    total = sum(flattened)
    if total <= EPSILON:
        return [0.0 for _ in flattened]
    return [value / total for value in flattened]


def dimension_evidence_strength(confidence: float, observability: float) -> float:
    product = max(0.0, float(confidence)) * max(0.0, float(observability))
    return product ** DIMENSION_EVIDENCE_EXPONENT


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return
    process.kill()


def communicate_with_timeout(
    process: subprocess.Popen[str],
    *,
    timeout_seconds: int,
) -> Tuple[str, str, bool]:
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        terminate_process_tree(process)
        tail_stdout, tail_stderr = process.communicate()
        stdout = coerce_process_output(exc.stdout) + coerce_process_output(tail_stdout)
        stderr = coerce_process_output(exc.stderr) + coerce_process_output(tail_stderr)
    return stdout or "", stderr or "", timed_out

@dataclass(frozen=True)
class EnvironmentReport:
    harness_root: str
    repo_root: str
    benchmark_override_path: str
    benchmark_override_present: bool
    cached_benchmark_path: str
    cached_benchmark_present: bool
    active_benchmark_source: str
    active_benchmark_valid: bool
    benchmark_id: str
    app_name: str
    gh_path: str
    gh_authenticated: bool
    copilot_cli_available: bool
    copilot_cli_version: str
    cargo_path: str
    scorer_backend: str
    default_tool: str
    default_model: str
    protected_paths: List[str]


@dataclass(frozen=True)
class CopilotInvocation:
    tool: str
    model: str
    prompt_bytes: int
    duration_seconds: float
    timed_out: bool
    returncode: int
    output_bytes: int
    output_path: str
    usage_available: bool
    usage_source: str
    usage_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StatusEntry:
    status: str
    path: str
    original_path: Optional[str] = None


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_identifier(text: str) -> str:
    return "".join(str(text).split())


@dataclass(frozen=True)
class Cosmic:
    entries: int
    exits: int
    reads: int
    writes: int

    def validate(self) -> None:
        for field_name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"cosmic.{field_name} must be >= 0, got {value}")

    def cfp(self) -> int:
        return self.entries + self.exits + self.reads + self.writes

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Cosmic":
        normalized = {normalize_identifier(key): value for key, value in payload.items()}
        return cls(
            entries=int(normalized["entries"]),
            exits=int(normalized["exits"]),
            reads=int(normalized["reads"]),
            writes=int(normalized["writes"]),
        )


@dataclass(frozen=True)
class Taus:
    k: float
    p: float
    h: float
    d: float
    m: float
    r: float

    def validate(self) -> None:
        for field_name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"taus.{field_name} must be > 0, got {value}")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Taus":
        normalized = {normalize_identifier(key): value for key, value in payload.items()}
        return cls(
            k=float(normalized["k"]),
            p=float(normalized["p"]),
            h=float(normalized["h"]),
            d=float(normalized["d"]),
            m=float(normalized["m"]),
            r=float(normalized["r"]),
        )


@dataclass(frozen=True)
class Scenario:
    k: int
    p: int
    h: int
    d: int
    m: int
    r: int
    response_seconds_total: Optional[float] = None

    def validate(self, label: str) -> None:
        for field_name in ("k", "p", "h", "d", "m", "r"):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{label}.{field_name} must be >= 0, got {value}")
        if self.response_seconds_total is not None and self.response_seconds_total < 0:
            raise ValueError(
                f"{label}.response_seconds_total must be >= 0, got {self.response_seconds_total}"
            )

    def predicted_time_seconds(self, taus: Taus) -> float:
        response_seconds = self.response_seconds_total
        if response_seconds is None:
            response_seconds = self.r * taus.r
        return (
            self.k * taus.k
            + self.p * taus.p
            + self.h * taus.h
            + self.d * taus.d
            + self.m * taus.m
            + response_seconds
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "k": self.k,
            "p": self.p,
            "h": self.h,
            "d": self.d,
            "m": self.m,
            "r": self.r,
        }
        if self.response_seconds_total is not None:
            payload["response_seconds_total"] = self.response_seconds_total
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Scenario":
        normalized = {normalize_identifier(key): value for key, value in payload.items()}
        return cls(
            k=int(normalized["k"]),
            p=int(normalized["p"]),
            h=int(normalized["h"]),
            d=int(normalized["d"]),
            m=int(normalized["m"]),
            r=int(normalized["r"]),
            response_seconds_total=(
                float(normalized["response_seconds_total"])
                if normalized.get("response_seconds_total") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class PersonaWeight:
    persona_id: str
    w: float

    def validate(self) -> None:
        if not self.persona_id.strip():
            raise ValueError("persona_id must not be empty")
        if not (0.0 <= self.w <= 1.0):
            raise ValueError(f"w for persona `{self.persona_id}` must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        return {"persona_id": self.persona_id, "w": self.w}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PersonaWeight":
        return cls(persona_id=str(payload["persona_id"]), w=float(payload["w"]))


@dataclass(frozen=True)
class CriterionSpec:
    id: str
    description: str

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("mandatory criterion id must not be empty")
        if not self.description.strip():
            raise ValueError(f"mandatory criterion `{self.id}` must have a description")

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "description": self.description}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CriterionSpec":
        return cls(id=str(payload["id"]), description=str(payload["description"]))


@dataclass(frozen=True)
class BenchmarkDimension:
    id: str
    description: str
    w: float
    baseline_anchor: str = ""

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("dimension id must not be empty")
        if not self.description.strip():
            raise ValueError(f"dimension `{self.id}` must have a description")
        if not self.baseline_anchor.strip():
            raise ValueError(f"dimension `{self.id}` must have a baseline_anchor")
        if not (0.0 <= self.w <= 1.0):
            raise ValueError(f"weight for dimension `{self.id}` must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "w": self.w,
            "baseline_anchor": self.baseline_anchor,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BenchmarkDimension":
        return cls(
            id=str(payload["id"]),
            description=str(payload["description"]),
            w=float(payload["w"]),
            baseline_anchor=collapse_whitespace(
                str(payload.get("baseline_anchor", payload.get("baseline_reference", payload["description"])))
            ),
        )


@dataclass(frozen=True)
class BenchmarkTask:
    persona_id: str
    task_id: str
    q: float
    task_description: str
    baseline: Scenario

    def validate(self) -> None:
        if not self.persona_id.strip():
            raise ValueError("task persona_id must not be empty")
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.task_description.strip():
            raise ValueError(f"task_description must not be empty for `{self.persona_id}` / `{self.task_id}`")
        if not (0.0 <= self.q <= 1.0):
            raise ValueError(
                f"q for persona `{self.persona_id}` task `{self.task_id}` must be between 0.0 and 1.0"
            )
        self.baseline.validate(f"tasks[{self.persona_id}/{self.task_id}].baseline")

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "task_id": self.task_id,
            "q": self.q,
            "task_description": self.task_description,
            "baseline": self.baseline.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BenchmarkTask":
        return cls(
            persona_id=str(payload["persona_id"]),
            task_id=str(payload["task_id"]),
            q=float(payload["q"]),
            task_description=str(payload["task_description"]),
            baseline=Scenario.from_dict(payload["baseline"]),
        )


@dataclass(frozen=True)
class Benchmark:
    benchmark_id: str
    app_name: str
    feature_name: str
    app_summary: str = ""
    paths_of_interest: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    build_commands: List[str] = field(default_factory=list)
    mandatory_criteria: List[CriterionSpec] = field(default_factory=list)
    dimensions: List[BenchmarkDimension] = field(default_factory=list)
    taus: Taus = field(default_factory=lambda: Taus(0.28, 1.1, 0.4, 1.2, 1.35, 0.5))
    persona_weights: List[PersonaWeight] = field(default_factory=list)
    tasks: List[BenchmarkTask] = field(default_factory=list)

    def validate(self) -> None:
        if not self.benchmark_id.strip():
            raise ValueError("benchmark_id must not be empty")
        if not self.app_name.strip():
            raise ValueError("app_name must not be empty")
        if not self.feature_name.strip():
            raise ValueError("feature_name must not be empty")
        if not self.mandatory_criteria:
            raise ValueError("at least one mandatory criterion is required")
        if not self.dimensions and not self.tasks:
            raise ValueError("at least one benchmark dimension or task is required")

        self.taus.validate()
        for criterion in self.mandatory_criteria:
            criterion.validate()
        for dimension in self.dimensions:
            dimension.validate()
        if self.dimensions:
            dimension_sum = sum(dimension.w for dimension in self.dimensions)
            if abs(dimension_sum - 1.0) > EPSILON:
                raise ValueError(f"dimension weights must sum to 1.0, got {dimension_sum:.12f}")

        if self.tasks:
            if not self.persona_weights:
                raise ValueError("at least one persona weight is required when tasks are present")
            for weight in self.persona_weights:
                weight.validate()
            for task in self.tasks:
                task.validate()

            persona_map: Dict[str, float] = {}
            for weight in self.persona_weights:
                if weight.persona_id in persona_map:
                    raise ValueError(f"duplicate persona weight for `{weight.persona_id}`")
                persona_map[weight.persona_id] = weight.w

            persona_sum = sum(weight.w for weight in self.persona_weights)
            if abs(persona_sum - 1.0) > EPSILON:
                raise ValueError(f"persona weights must sum to 1.0, got {persona_sum:.12f}")

            q_sums: Dict[str, float] = {}
            seen_pairs: set[tuple[str, str]] = set()
            for task in self.tasks:
                if task.persona_id not in persona_map:
                    raise ValueError(
                        f"task `{task.task_id}` references persona `{task.persona_id}` with no persona weight"
                    )
                key = (task.persona_id, task.task_id)
                if key in seen_pairs:
                    raise ValueError(f"duplicate persona/task pair: `{task.persona_id}` / `{task.task_id}`")
                seen_pairs.add(key)
                q_sums[task.persona_id] = q_sums.get(task.persona_id, 0.0) + task.q

            for weight in self.persona_weights:
                total = q_sums.get(weight.persona_id, 0.0)
                if abs(total - 1.0) > EPSILON:
                    raise ValueError(
                        f"task weights q for persona `{weight.persona_id}` must sum to 1.0, got {total:.12f}"
                    )

    def to_prompt_payload(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "app_name": self.app_name,
            "feature_name": self.feature_name,
            "app_summary": self.app_summary,
            "paths_of_interest": self.paths_of_interest,
            "notes": self.notes,
            "build_commands": self.build_commands,
            "mandatory_criteria": [criterion.to_dict() for criterion in self.mandatory_criteria],
            "dimensions": [dimension.to_dict() for dimension in self.dimensions],
            "taus": asdict(self.taus),
            "persona_weights": [weight.to_dict() for weight in self.persona_weights],
            "tasks": [task.to_prompt_dict() for task in self.tasks],
        }


def command_path(binary: str) -> str:
    return shutil.which(binary) or ""


def resolve_copilot_cli_command() -> str:
    candidates: List[Path] = []

    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        npm_bin = Path(appdata) / "npm"
        candidates.extend(
            [
                npm_bin / "copilot.cmd",
                npm_bin / "copilot",
                npm_bin / "node_modules" / "@github" / "copilot" / "node_modules" / "@github" / "copilot-win32-x64" / "copilot.exe",
            ]
        )

    runner_root = discover_runner_root()
    if runner_root:
        candidates.extend(
            [
                runner_root / "copilot.cmd",
                runner_root / "copilot.ps1",
            ]
        )

    command_candidates = [
        command_path("copilot.cmd"),
        command_path("copilot"),
    ]
    for candidate in command_candidates:
        if candidate:
            candidates.append(Path(candidate))

    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)
    return ""


def shell_command(command: str) -> List[str]:
    if os.name == "nt":
        return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command]
    return ["/bin/zsh", "-lc", command]


def run_command(
    args: Sequence[str],
    *,
    cwd: Path = ROOT,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    kwargs: Dict[str, Any] = {
        "cwd": cwd,
        "check": check,
        "capture_output": capture_output,
        "text": text,
    }
    if text:
        kwargs["encoding"] = "utf-8"
        kwargs["errors"] = "replace"
    return subprocess.run(list(args), **kwargs)


def run_git(args: Sequence[str], *, cwd: Path = ROOT, check: bool = True) -> str:
    return run_command(["git", *args], cwd=cwd, check=check).stdout.strip()


def run_noninteractive_command(args: Sequence[str], *, cwd: Path = ROOT, timeout_seconds: int = 5) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=timeout_seconds,
    )


def run_git_paths(args: Sequence[str], *, cwd: Path = ROOT) -> List[Path]:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=False,
    )
    return [Path(chunk.decode("utf-8")) for chunk in result.stdout.split(b"\0") if chunk]


def parse_status_entries(*, cwd: Path = ROOT) -> List[StatusEntry]:
    raw = run_command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=cwd,
    ).stdout.splitlines()
    entries: List[StatusEntry] = []
    for line in raw:
        status = line[:2]
        path_text = line[3:]
        if " -> " in path_text:
            original_path, path_text = path_text.split(" -> ", 1)
            entry = StatusEntry(status=status, path=path_text, original_path=original_path)
        else:
            entry = StatusEntry(status=status, path=path_text)
        if any(is_internal_state_path(item) for item in entry_paths(entry)):
            continue
        entries.append(entry)
    return entries


def entry_paths(entry: StatusEntry) -> List[str]:
    paths = [entry.path]
    if entry.original_path and entry.original_path != entry.path:
        paths.append(entry.original_path)
    return paths


def remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()


def restore_entries(start_commit: str, entries: Sequence[StatusEntry], *, cwd: Path = ROOT) -> None:
    tracked_paths = sorted(
        {
            path_text
            for entry in entries
            if entry.status != "??"
            for path_text in entry_paths(entry)
        }
    )
    if tracked_paths:
        run_command(
            ["git", "restore", "--source", start_commit, "--staged", "--worktree", "--", *tracked_paths],
            cwd=cwd,
        )
    for entry in entries:
        if entry.status == "??":
            remove_path(cwd / entry.path)


def is_protected_path(path_text: str) -> bool:
    normalized = path_text.replace("\\", "/")
    if normalized in PROTECTED_FILES:
        return True
    return any(normalized.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def is_internal_state_path(path_text: str) -> bool:
    normalized = path_text.replace("\\", "/")
    if normalized in INTERNAL_STATE_FILES:
        return True
    return any(normalized.startswith(prefix) for prefix in INTERNAL_STATE_PREFIXES)


def load_benchmark(path: Path = BENCHMARK_PATH) -> Benchmark:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Benchmark file `{path}` was not found.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse benchmark JSON at `{path}`: {exc}") from exc

    benchmark = Benchmark(
        benchmark_id=str(payload["benchmark_id"]),
        app_name=str(payload["app_name"]),
        feature_name=str(payload["feature_name"]),
        app_summary=str(payload.get("app_summary", "")),
        paths_of_interest=[str(item) for item in payload.get("paths_of_interest", [])],
        notes=[str(item) for item in payload.get("notes", [])],
        build_commands=[str(item) for item in payload.get("build_commands", [])],
        mandatory_criteria=[
            CriterionSpec.from_dict(item) for item in payload.get("mandatory_criteria", [])
        ],
        dimensions=[BenchmarkDimension.from_dict(item) for item in payload.get("dimensions", [])],
        taus=Taus.from_dict(payload.get("taus", {"k": 0.28, "p": 1.1, "h": 0.4, "d": 1.2, "m": 1.35, "r": 0.5})),
        persona_weights=[PersonaWeight.from_dict(item) for item in payload.get("persona_weights", [])],
        tasks=[BenchmarkTask.from_dict(item) for item in payload.get("tasks", [])],
    )
    benchmark.validate()
    return benchmark


def default_benchmark_payload() -> Dict[str, Any]:
    repo_name = ROOT.name.replace("_", "-").replace(" ", "-").lower() or "app"
    app_name = ROOT.name.replace("-", " ").replace("_", " ").strip() or "Application"
    return {
        "benchmark_id": f"{repo_name}-static-scorecard-v1",
        "app_name": app_name,
        "feature_name": "Static multi-parameter application benchmark",
        "app_summary": f"Autogenerated static multi-parameter benchmark for {app_name}.",
        "paths_of_interest": detect_paths_of_interest(),
        "notes": [
            "Autogenerated hidden benchmark for this run using a fixed 20-parameter application scorecard.",
            "The harness and support assets are excluded from product scoring.",
        ],
        "build_commands": detect_build_commands(),
        "mandatory_criteria": list(STATIC_MANDATORY_CRITERIA),
        "dimensions": [
            {
                "id": row["id"],
                "description": row["description"],
                "baseline_anchor": row["default_anchor"],
                "w": row["w"],
            }
            for row in STATIC_DIMENSION_SPECS
        ],
        "taus": {"k": 0.28, "p": 1.1, "h": 0.4, "d": 1.2, "m": 1.35, "r": 0.5},
        "persona_weights": [],
        "tasks": [],
    }


def detect_paths_of_interest() -> List[str]:
    candidates = [
        "src/",
        "app/",
        "crates/",
        "templates/",
        "static/",
        "web/",
        "frontend/",
        "backend/",
        "Sources/",
        "Tests/",
        "tests/",
    ]
    detected = [path for path in candidates if (ROOT / path.rstrip("/")).exists()]
    return detected or ["."]


def detect_build_commands() -> List[str]:
    cargo_toml = ROOT / "Cargo.toml"
    if cargo_toml.exists():
        return ["cargo test --all-features"]

    xcode_projects = sorted(ROOT.glob("*.xcodeproj"))
    if xcode_projects:
        project_name = xcode_projects[0].name
        schemes = detect_xcode_schemes(xcode_projects[0])
        if schemes:
            scheme = schemes[0]
            return [
                f"xcodebuild -project {project_name} -scheme {scheme} -configuration Release -destination 'generic/platform=iOS Simulator' clean build"
            ]

    package_json = ROOT / "package.json"
    if package_json.exists():
        try:
            package_payload = json.loads(package_json.read_text(encoding="utf-8"))
        except Exception:
            package_payload = {}
        scripts = package_payload.get("scripts") or {}
        if isinstance(scripts, dict) and "test" in scripts:
            return ["npm test -- --runInBand"]

    pyproject = ROOT / "pyproject.toml"
    pyproject_text = pyproject.read_text(encoding="utf-8") if pyproject.exists() else ""
    if "pytest" in pyproject_text.lower() or (ROOT / "pytest.ini").exists():
        return ["python3 -m pytest -q"]
    if (ROOT / "tests").is_dir():
        return ["python3 -m unittest discover -s tests -p 'test_*.py'"]

    return []


def detect_xcode_schemes(project_path: Path) -> List[str]:
    if not command_path("xcodebuild"):
        return []
    completed = run_command(
        ["xcodebuild", "-list", "-project", project_path.name],
        cwd=project_path.parent,
        check=False,
    )
    if completed.returncode != 0:
        return []

    schemes: List[str] = []
    in_schemes = False
    for raw_line in completed.stdout.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == "Schemes:":
            in_schemes = True
            continue
        if in_schemes:
            if not stripped:
                break
            schemes.append(stripped)

    preferred = [scheme for scheme in schemes if not scheme.endswith(("Tests", "UITests"))]
    return preferred or schemes


def normalize_weight_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        persona_id = normalize_identifier(row.get("persona_id", ""))
        if not persona_id:
            continue
        try:
            weight = float(row.get("w", 0.0))
        except (TypeError, ValueError):
            continue
        if weight > 0:
            cleaned.append({"persona_id": persona_id, "w": weight})

    if not cleaned:
        return [{"persona_id": "primary-user", "w": 1.0}]

    total = sum(row["w"] for row in cleaned)
    if total <= 0:
        return [{"persona_id": "primary-user", "w": 1.0}]

    return [{"persona_id": row["persona_id"], "w": row["w"] / total} for row in cleaned]


def normalize_dimension_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        dimension_id = normalize_identifier(row.get("id", ""))
        description = collapse_whitespace(str(row.get("description", "")))
        baseline_anchor = collapse_whitespace(
            str(row.get("baseline_anchor", row.get("baseline_reference", "")))
        )
        if not dimension_id or not description or not baseline_anchor:
            continue
        try:
            weight = float(row.get("w", 0.0))
        except (TypeError, ValueError):
            continue
        if weight > 0:
            cleaned.append(
                {
                    "id": dimension_id,
                    "description": description,
                    "baseline_anchor": baseline_anchor,
                    "w": weight,
                }
            )

    if not cleaned:
        return []

    total = sum(row["w"] for row in cleaned)
    if total <= 0:
        return []
    return [{**row, "w": row["w"] / total} for row in cleaned]


def merge_static_dimension_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    anchor_overrides: Dict[str, str] = {}
    for row in rows:
        dimension_id = normalize_identifier(row.get("id", ""))
        if dimension_id not in STATIC_DIMENSION_IDS:
            continue
        baseline_anchor = collapse_whitespace(
            str(row.get("baseline_anchor", row.get("baseline_reference", "")))
        )
        if baseline_anchor:
            anchor_overrides[dimension_id] = baseline_anchor

    merged: List[Dict[str, Any]] = []
    for spec in STATIC_DIMENSION_SPECS:
        merged.append(
            {
                "id": spec["id"],
                "description": spec["description"],
                "baseline_anchor": anchor_overrides.get(spec["id"], spec["default_anchor"]),
                "w": spec["w"],
            }
        )
    return normalize_dimension_rows(merged)


def sanitize_baseline(payload: Dict[str, Any]) -> Dict[str, int]:
    defaults = {"k": 24, "p": 10, "h": 2, "d": 0, "m": 8, "r": 4}
    cleaned: Dict[str, int] = {}
    for key, default_value in defaults.items():
        try:
            value = int(payload.get(key, default_value))
        except (TypeError, ValueError):
            value = default_value
        cleaned[key] = max(0, value)
    if "response_seconds_total" in payload and payload.get("response_seconds_total") is not None:
        try:
            cleaned["response_seconds_total"] = max(0.0, float(payload["response_seconds_total"]))  # type: ignore[assignment]
        except (TypeError, ValueError):
            pass
    return cleaned


def normalize_task_rows(rows: Sequence[Dict[str, Any]], persona_ids: Sequence[str]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    valid_personas = set(persona_ids)
    for row in rows:
        persona_id = normalize_identifier(row.get("persona_id", ""))
        task_id = normalize_identifier(row.get("task_id", ""))
        task_description = collapse_whitespace(str(row.get("task_description", "")))
        if persona_id not in valid_personas or not task_id:
            continue
        try:
            q_value = float(row.get("q", 0.0))
        except (TypeError, ValueError):
            q_value = 0.0
        cleaned.append(
            {
                "persona_id": persona_id,
                "task_id": task_id,
                "q": max(0.0, q_value),
                "task_description": task_description or f"Complete task `{task_id}`.",
                "baseline": sanitize_baseline(row.get("baseline") or {}),
            }
        )

    if not cleaned:
        return [
            {
                "persona_id": persona_ids[0],
                "task_id": "complete-core-flow",
                "q": 1.0,
                "task_description": "Complete the app's primary end-to-end user-visible workflow.",
                "baseline": {"k": 24, "p": 10, "h": 2, "d": 0, "m": 8, "r": 4},
            }
        ]

    by_persona: Dict[str, List[Dict[str, Any]]] = {persona_id: [] for persona_id in persona_ids}
    for row in cleaned:
        by_persona[row["persona_id"]].append(row)

    normalized: List[Dict[str, Any]] = []
    for persona_id in persona_ids:
        persona_rows = by_persona.get(persona_id, [])
        if not persona_rows:
            normalized.append(
                {
                    "persona_id": persona_id,
                    "task_id": "complete-core-flow",
                    "q": 1.0,
                    "task_description": "Complete the app's primary end-to-end user-visible workflow.",
                    "baseline": {"k": 24, "p": 10, "h": 2, "d": 0, "m": 8, "r": 4},
                }
            )
            continue
        total_q = sum(float(row["q"]) for row in persona_rows)
        if total_q <= 0:
            total_q = float(len(persona_rows))
            for row in persona_rows:
                row["q"] = 1.0
        for row in persona_rows:
            normalized.append({**row, "q": float(row["q"]) / total_q})
    return normalized


def sanitize_synthesized_benchmark_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    defaults = default_benchmark_payload()
    benchmark_id = normalize_identifier(payload.get("benchmark_id", "")) or defaults["benchmark_id"]
    app_name = collapse_whitespace(str(payload.get("app_name", ""))) or defaults["app_name"]
    feature_name = collapse_whitespace(str(payload.get("feature_name", ""))) or defaults["feature_name"]
    app_summary = collapse_whitespace(str(payload.get("app_summary", ""))) or defaults["app_summary"]

    paths_of_interest = [
        str(item).strip() for item in payload.get("paths_of_interest", []) if str(item).strip()
    ] or defaults["paths_of_interest"]
    notes = [str(item).strip() for item in payload.get("notes", []) if str(item).strip()] or defaults["notes"]
    build_commands = [
        str(item).strip() for item in payload.get("build_commands", []) if str(item).strip()
    ] or defaults["build_commands"]

    mandatory_criteria = list(defaults["mandatory_criteria"])

    taus = dict(defaults["taus"])
    raw_taus = payload.get("taus") or {}
    for key in taus:
        try:
            value = float(raw_taus.get(key, taus[key]))
        except (TypeError, ValueError):
            value = float(taus[key])
        taus[key] = value if value > 0 else float(defaults["taus"][key])

    dimensions = merge_static_dimension_rows(payload.get("dimensions") or [])
    raw_tasks = payload.get("tasks") or []
    persona_weights: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    if raw_tasks:
        persona_weights = normalize_weight_rows(payload.get("persona_weights") or defaults["persona_weights"])
        persona_ids = [row["persona_id"] for row in persona_weights]
        tasks = normalize_task_rows(raw_tasks, persona_ids)
    if not dimensions and not tasks:
        raise ValueError("Synthesized benchmark must include at least one weighted dimension or task")

    return {
        "benchmark_id": benchmark_id,
        "app_name": app_name,
        "feature_name": feature_name,
        "app_summary": app_summary,
        "paths_of_interest": paths_of_interest,
        "notes": notes,
        "build_commands": build_commands,
        "mandatory_criteria": mandatory_criteria,
        "dimensions": dimensions,
        "taus": taus,
        "persona_weights": persona_weights,
        "tasks": tasks,
    }


def build_benchmark_synthesis_prompt() -> str:
    static_dimensions_json = json.dumps(
        [
            {
                "id": row["id"],
                "description": row["description"],
                "baseline_anchor": row["default_anchor"],
                "w": row["w"],
            }
            for row in STATIC_DIMENSION_SPECS
        ],
        indent=2,
    )
    return (
        "You are defining a frozen evaluation benchmark for an autonomous app-improvement loop.\n\n"
        "Inspect this repository and fill in a fixed 20-parameter benchmark that can be used to score future app improvements.\n"
        "This benchmark is for the evaluator only. It is not a product roadmap and it is not a backlog.\n"
        "Freeze the repository's current baseline state across the fixed dimensions below. Do not invent new dimensions.\n"
        "If BRAND.md, ABOUT.md, or FEATURES.md exist, treat them as high-signal evidence about the app's current identity.\n"
        "If those files exist, they are usually enough to anchor the benchmark; only read a few targeted code or test files to confirm shipped behavior.\n"
        "Do not narrate your progress or emit interim updates. Inspect minimally, then output the final JSON object only.\n"
        "If the repo contains progress.txt, treat it only as shipped-work history so you avoid benchmarking already-saturated side paths.\n\n"
        "Rules:\n"
        "- Output JSON only. No markdown fences.\n"
        "- Emit compact JSON on a single line if possible.\n"
        "- Be application-agnostic, but use the fixed scorecard below.\n"
        "- Ignore the loop harness itself: prepare.py, train.py, program.md, support_docs/, support_scripts/, and karpathy-files/ are not the product.\n"
        "- Use all 20 fixed dimensions exactly as given. Do not add, remove, rename, or reweight them.\n"
        "- For each fixed dimension, write a concise baseline_anchor describing the current baseline state at benchmark freeze time.\n"
        "- Keep baseline_anchor factual and broad enough that future real improvements can register as better or worse.\n"
        "- Preserve the fixed mandatory criteria.\n"
        "- You are freezing a baseline, not defining a roadmap or desired future features.\n"
        "- Build commands must be real commands appropriate for this repo.\n"
        "- If this is a Rust app, prefer cargo-based checks when supported by the repo.\n"
        "- Use forward slashes in JSON paths.\n"
        "- Stop as soon as you have enough evidence to write a stable benchmark.\n"
        "- Use reasonable defaults when something is ambiguous, but keep notes short and factual.\n\n"
        "Fixed dimensions:\n"
        f"{static_dimensions_json}\n\n"
        "Required JSON shape:\n"
        "{\n"
        '  "benchmark_id": "slug-v1",\n'
        '  "app_name": "App Name",\n'
        '  "feature_name": "Primary improvement benchmark",\n'
        '  "app_summary": "Short summary.",\n'
        '  "paths_of_interest": ["src/"],\n'
        '  "notes": ["Short note."],\n'
        '  "build_commands": ["cargo test"],\n'
        '  "mandatory_criteria": [\n'
        '    {"id": "criterion-id", "description": "Short description."}\n'
        "  ],\n"
        '  "dimensions": [\n'
        "    {\n"
        '      "id": "dimension-id",\n'
        '      "description": "Use the fixed description exactly.",\n'
        '      "baseline_anchor": "Concise factual description of the current baseline state for this dimension.",\n'
        '      "w": 0.10\n'
        "    }\n"
        "  ],\n"
        '  "taus": {"k": 0.28, "p": 1.1, "h": 0.4, "d": 1.2, "m": 1.35, "r": 0.5},\n'
        '  "persona_weights": [],\n'
        '  "tasks": []\n'
        "}\n"
    )


def synthesize_benchmark(
    *,
    tool: str = DEFAULT_TOOL,
    model: Optional[str] = None,
    timeout_seconds: int = BENCHMARK_SYNTHESIS_TIMEOUT_SECONDS,
) -> Benchmark:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw_message, _ = run_agent_prompt(
        build_benchmark_synthesis_prompt(),
        cwd=ROOT,
        output_path=LAST_BENCHMARK_MESSAGE,
        model=model,
        tool=tool,
        timeout_seconds=timeout_seconds,
        available_tools=("rg", "view", "glob"),
        silent=True,
        stream_mode="off",
        reasoning_effort="low",
    )
    payload = sanitize_synthesized_benchmark_payload(extract_json_object(raw_message))
    CACHED_BENCHMARK_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return load_benchmark(CACHED_BENCHMARK_PATH)


def benchmark_uses_static_scorecard(benchmark: Benchmark) -> bool:
    if len(benchmark.dimensions) != len(STATIC_DIMENSION_SPECS):
        return False
    return {dimension.id for dimension in benchmark.dimensions} == STATIC_DIMENSION_IDS


def get_active_benchmark(
    *,
    tool: str = DEFAULT_TOOL,
    model: Optional[str] = None,
    allow_synthesis: bool = True,
    deadline_monotonic: Optional[float] = None,
) -> tuple[Benchmark, str]:
    if BENCHMARK_PATH.exists():
        return load_benchmark(BENCHMARK_PATH), "override"
    if CACHED_BENCHMARK_PATH.exists():
        try:
            cached_benchmark = load_benchmark(CACHED_BENCHMARK_PATH)
            if cached_benchmark.dimensions and benchmark_uses_static_scorecard(cached_benchmark):
                return cached_benchmark, "cache"
            if not allow_synthesis:
                return cached_benchmark, "cache"
        except Exception:
            if not allow_synthesis:
                raise
        CACHED_BENCHMARK_PATH.unlink(missing_ok=True)
    if not allow_synthesis:
        raise FileNotFoundError("No benchmark override or cached benchmark exists")
    benchmark = synthesize_benchmark(
        tool=tool,
        model=model,
        timeout_seconds=bounded_timeout(
            BENCHMARK_SYNTHESIS_TIMEOUT_SECONDS,
            deadline_monotonic=deadline_monotonic,
            label="benchmark synthesis",
        ),
    )
    return benchmark, "synthesized"


def choose_backend(requested: str) -> str:
    if requested not in {"auto", "python", "rust"}:
        raise ValueError(f"Unknown backend `{requested}`")
    if requested == "python":
        return "python"
    rust_available = bool(command_path("cargo")) and RUST_ASSESSOR_MANIFEST.exists()
    if requested == "rust":
        if not rust_available:
            raise RuntimeError("Rust scorer requested but cargo or support_scripts/feature_assessor is unavailable")
        return "rust"
    return "rust" if rust_available else "python"


def is_gh_authenticated() -> bool:
    if not command_path("gh"):
        return False
    try:
        completed = run_noninteractive_command(["gh", "auth", "status"], cwd=ROOT)
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0


def get_copilot_cli_version() -> str:
    copilot_command = resolve_copilot_cli_command()
    if not copilot_command:
        return ""
    try:
        completed = run_noninteractive_command([copilot_command, "--version"], cwd=ROOT)
    except subprocess.TimeoutExpired:
        return ""
    output = (completed.stdout or completed.stderr).strip()
    if not output:
        return ""
    first_line = output.splitlines()[0].strip()
    lowered = first_line.lower()
    if completed.returncode != 0 or "cannot find github copilot cli" in lowered:
        return ""
    return first_line


def check_environment(backend: str = DEFAULT_BACKEND) -> EnvironmentReport:
    benchmark_override_present = BENCHMARK_PATH.exists()
    cached_benchmark_present = CACHED_BENCHMARK_PATH.exists()
    active_benchmark_source = "none"
    active_benchmark_valid = False
    benchmark_id = ""
    app_name = ""

    if benchmark_override_present:
        try:
            benchmark = load_benchmark(BENCHMARK_PATH)
        except Exception:
            active_benchmark_source = "override-invalid"
        else:
            active_benchmark_source = "override"
            active_benchmark_valid = True
            benchmark_id = benchmark.benchmark_id
            app_name = benchmark.app_name
    elif cached_benchmark_present:
        try:
            benchmark = load_benchmark(CACHED_BENCHMARK_PATH)
        except Exception:
            active_benchmark_source = "cache-invalid"
        else:
            active_benchmark_source = "cache"
            active_benchmark_valid = True
            benchmark_id = benchmark.benchmark_id
            app_name = benchmark.app_name

    copilot_cli_version = get_copilot_cli_version()

    return EnvironmentReport(
        harness_root=str(HARNESS_ROOT),
        repo_root=str(ROOT),
        benchmark_override_path=str(BENCHMARK_PATH),
        benchmark_override_present=benchmark_override_present,
        cached_benchmark_path=str(CACHED_BENCHMARK_PATH),
        cached_benchmark_present=cached_benchmark_present,
        active_benchmark_source=active_benchmark_source,
        active_benchmark_valid=active_benchmark_valid,
        benchmark_id=benchmark_id,
        app_name=app_name,
        gh_path=command_path("gh"),
        gh_authenticated=is_gh_authenticated(),
        copilot_cli_available=bool(copilot_cli_version),
        copilot_cli_version=copilot_cli_version,
        cargo_path=command_path("cargo"),
        scorer_backend=choose_backend(backend),
        default_tool=DEFAULT_TOOL,
        default_model=DEFAULT_COPILOT_MODEL,
        protected_paths=sorted([*PROTECTED_FILES, *PROTECTED_PREFIXES]),
    )


def copy_file_into_snapshot(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        target = source.readlink()
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(target)
        return
    shutil.copy2(source, destination)


def build_snapshot(source_root: Path, snapshot_root: Path) -> None:
    tracked_paths = run_git_paths(["ls-files", "-z"], cwd=source_root)
    untracked_paths = run_git_paths(["ls-files", "--others", "--exclude-standard", "-z"], cwd=source_root)
    for relative_path in tracked_paths + untracked_paths:
        source_path = source_root / relative_path
        if not source_path.exists() and not source_path.is_symlink():
            continue
        if source_path.is_dir():
            continue
        copy_file_into_snapshot(source_path, snapshot_root / relative_path)
    run_command(["git", "init", "-q"], cwd=snapshot_root)


def run_build_commands(
    commands: Sequence[str],
    *,
    cwd: Path,
    deadline_monotonic: Optional[float] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for command in commands:
        t0 = time.time()
        timeout_seconds = bounded_timeout(
            BUILD_COMMAND_TIMEOUT_SECONDS,
            deadline_monotonic=deadline_monotonic,
            label="build/test command",
        )
        process = subprocess.Popen(
            shell_command(command),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout, stderr, timed_out = communicate_with_timeout(process, timeout_seconds=timeout_seconds)
        combined = "\n".join(
            [part for part in (stdout.strip(), stderr.strip()) if part]
        ).strip()
        output_tail = "\n".join(combined.splitlines()[-40:])
        result = {
            "command": command,
            "ok": (process.returncode == 0) and not timed_out,
            "returncode": process.returncode if process.returncode is not None else -1,
            "duration_seconds": round(time.time() - t0, 3),
            "timed_out": timed_out,
            "output_tail": output_tail,
        }
        results.append(result)
        if timed_out:
            raise RuntimeError(
                f"Build/test command timed out after {timeout_seconds}s: {command}\n"
                f"{output_tail or '(no output captured)'}"
            )
        if process.returncode != 0:
            raise RuntimeError(
                f"Build/test command failed: {command}\n{output_tail or '(no output captured)'}"
            )
    return results


def build_evaluator_prompt(benchmark: Benchmark, build_results: Sequence[Dict[str, Any]]) -> str:
    benchmark_json = json.dumps(benchmark.to_prompt_payload(), indent=2)
    build_json = json.dumps(list(build_results), indent=2)
    if benchmark.dimensions:
        return (
            "You are the fixed app benchmark evaluator.\n\n"
            "Your job is to inspect this repository snapshot and compare it against the frozen baseline anchors embedded in the benchmark.\n"
            "Do not edit any files. Do not act like a general coding agent. Do not explore broadly. Do not write plans.\n"
            "Do not reinterpret the benchmark or invent new dimensions. Use only build-time, test-derived, and repository-derived evidence from this repo.\n"
            "If you cannot assess enough of the frozen benchmark with confidence, return status unresolved and list the missing pieces.\n\n"
            "Mandatory rules:\n"
            "- Return exactly one JSON object as the entire response. No prose before or after it.\n"
            "- No markdown fences. No explanations. No bullet points.\n"
            "- Emit compact JSON on a single line if possible.\n"
            "- g must be 0 or 1.\n"
            "- cosmic counts must be non-negative integers.\n"
            "- dimension_rows must contain exactly one row for each frozen dimension id.\n"
            "- delta must be in [-1, 1], where positive means better than the frozen baseline anchor, 0 means no meaningful change, and negative means worse.\n"
            "- confidence and observability must be in [0, 1].\n"
            "- Treat each baseline_anchor as the frozen baseline level for that lens, not as a checklist requiring every named detail to move.\n"
            "- A strong focused improvement in one dimension can deserve a large positive delta even if most other dimensions stay flat.\n"
            "- A coherent multi-area improvement can earn positive deltas across multiple touched dimensions and also improve holistic-improvement.\n"
            "- Do not reward scattered micro-tweaks as a strong holistic win. If the change feels fragmented, keep holistic-improvement at 0 or below.\n"
            "- If a candidate makes a real but limited improvement on a dimension, use a small positive delta instead of 0.\n"
            "- If a candidate adds complexity, noise, or churn without a clear upside, use a small negative delta instead of 0.\n"
            "- Reserve delta = 0 only for a true no-op on that dimension.\n"
            "- Use the repo's own identity and evidence. Do not substitute generic UX categories or external taste.\n"
            "- Use only read/search tools if you need them. Never attempt to patch files.\n\n"
            "Frozen benchmark:\n"
            f"{benchmark_json}\n\n"
            "Build/test command results for this snapshot:\n"
            f"{build_json}\n\n"
            "Resolved output shape:\n"
            "{\n"
            '  "status": "resolved",\n'
            '  "g": 1,\n'
            '  "cosmic": {"entries": 0, "exits": 0, "reads": 0, "writes": 0},\n'
            '  "criteria": [{"id": "...", "result": "pass", "reason": "..."}],\n'
            '  "dimension_rows": [\n'
            "    {\n"
            '      "id": "dimension-id",\n'
            '      "delta": 0.15,\n'
            '      "confidence": 0.8,\n'
            '      "observability": 0.9,\n'
            '      "reason": "Short factual justification."\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Unresolved output shape:\n"
            '{\n  "status": "unresolved",\n  "missing": ["..."]\n}\n'
        )
    return (
        "You are the fixed app benchmark evaluator.\n\n"
        "Your job is to inspect this repository snapshot and emit only the measured candidate-side values.\n"
        "Do not edit any files. Do not act like a general coding agent. Do not explore broadly. Do not write plans.\n"
        "Do not change the inferred benchmark layer. Do not reinterpret persona weights,\n"
        "task weights, baselines, or tau constants. Use only build-time, test-derived, and repository-derived evidence from this repo.\n"
        "If any mandatory criterion or task assessment is unknown, return status unresolved and list the missing pieces.\n\n"
        "Mandatory rules:\n"
        "- Return exactly one JSON object as the entire response. No prose before or after it.\n"
        "- No markdown fences. No explanations. No bullet points.\n"
        "- Emit compact JSON on a single line if possible.\n"
        "- g must be 0 or 1.\n"
        "- cosmic counts must be non-negative integers.\n"
        "- feature_rows must contain exactly one row for each frozen persona/task pair.\n"
        "- Each feature row must contain only k, p, h, d, m, r, and optional response_seconds_total.\n"
        "- If the changed candidate weakly helps, reflect that with a small real improvement in the feature rows.\n"
        "- If the changed candidate adds friction or churn, reflect that with a small real regression in the feature rows.\n"
        "- Do not return a baseline copy for a changed passing candidate.\n"
        "- Use only read/search tools if you need them. Never attempt to patch files.\n\n"
        "Frozen benchmark:\n"
        f"{benchmark_json}\n\n"
        "Build/test command results for this snapshot:\n"
        f"{build_json}\n\n"
        "Resolved output shape:\n"
        "{\n"
        '  "status": "resolved",\n'
        '  "g": 1,\n'
        '  "cosmic": {"entries": 0, "exits": 0, "reads": 0, "writes": 0},\n'
        '  "criteria": [{"id": "...", "result": "pass", "reason": "..."}],\n'
        '  "feature_rows": [\n'
        '    {\n'
        '      "persona_id": "...",\n'
        '      "task_id": "...",\n'
        '      "feature": {"k": 0, "p": 0, "h": 0, "d": 0, "m": 0, "r": 0}\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Unresolved output shape:\n"
        '{\n  "status": "unresolved",\n  "missing": ["..."]\n}\n'
    )


def build_discriminative_retry_prompt(benchmark: Benchmark, build_results: Sequence[Dict[str, Any]]) -> str:
    base_prompt = build_evaluator_prompt(benchmark, build_results)
    if benchmark.dimensions:
        extra = (
            "\nAdditional instruction for this retry:\n"
            "- This snapshot is a changed candidate that already passed the gate.\n"
            "- A fully all-zero dimension result is invalid for this retry.\n"
            "- At least one dimension row must carry a non-zero signed delta.\n"
            "- If the effect is subtle, use tiny signed deltas such as +/-0.01 to +/-0.05.\n"
            "- If one dimension clearly moved a lot, say so with a real positive or negative delta instead of spreading tiny values everywhere.\n"
            "- Reward holistic-improvement only when the change hangs together as one coherent stronger whole.\n"
            "- If the change feels busier, noisier, or less useful, prefer a slight negative over zero.\n"
        )
    else:
        extra = (
            "\nAdditional instruction for this retry:\n"
            "- This snapshot is a changed candidate that already passed the gate.\n"
            "- Returning a baseline copy is invalid for this retry.\n"
            "- At least one feature row must differ from the frozen baseline.\n"
            "- If the effect is subtle, make a minimal but real change in the measured feature rows.\n"
            "- If the change adds friction or pointless churn, reflect that as a slight regression instead of a baseline copy.\n"
        )
    return base_prompt + extra


def build_discriminative_rescue_prompt(benchmark: Benchmark, build_results: Sequence[Dict[str, Any]]) -> str:
    benchmark_json = json.dumps(benchmark.to_prompt_payload(), indent=2)
    build_json = json.dumps(list(build_results), indent=2)
    if benchmark.dimensions:
        return (
            "Return one JSON object only.\n"
            "This is the final discriminative rescue pass for a changed candidate that already passed the gate.\n"
            "An all-zero dimension result is invalid.\n"
            "At least one dimension delta must be non-zero.\n"
            "If the effect is weak, use tiny signed deltas such as +/-0.01.\n"
            "A strong focused win may carry one dimension substantially even if other dimensions stay flat.\n"
            "Reward holistic-improvement only for a coherent integrated improvement, not a bundle of disconnected tweaks.\n"
            "If the effect is mostly noise or churn, use slight negative values instead of zero.\n\n"
            "Frozen benchmark:\n"
            f"{benchmark_json}\n\n"
            "Build/test results:\n"
            f"{build_json}\n"
        )
    return (
        "Return one JSON object only.\n"
        "This is the final discriminative rescue pass for a changed candidate that already passed the gate.\n"
        "Returning a baseline copy is invalid.\n"
        "At least one feature row must differ from the frozen baseline.\n"
        "If the effect is weak, make a minimal real difference.\n"
        "If the effect is mostly churn, reflect a slight regression rather than a baseline copy.\n\n"
        "Frozen benchmark:\n"
        f"{benchmark_json}\n\n"
        "Build/test results:\n"
        f"{build_json}\n"
    )


def build_json_repair_prompt(
    benchmark: Benchmark,
    build_results: Sequence[Dict[str, Any]],
    *,
    prior_error: str,
    final_pass: bool = False,
) -> str:
    base_prompt = build_evaluator_prompt(benchmark, build_results)
    error_text = collapse_whitespace(prior_error or "Previous evaluator output was not valid JSON.")
    if final_pass:
        return (
            "Return exactly one JSON object and nothing else.\n"
            "The previous evaluator output was invalid and could not be parsed.\n"
            f"Failure reason: {error_text}\n"
            "You must return either a valid resolved measurement JSON object or a valid unresolved JSON object.\n"
            "No prose, no bullets, no markdown fences, no prefatory text.\n\n"
            f"{base_prompt}"
        )
    return (
        f"{base_prompt}\n\n"
        "Additional instruction for this retry:\n"
        f"- The previous evaluator output was invalid and could not be parsed. Failure reason: {error_text}\n"
        "- Return exactly one JSON object and nothing else.\n"
        "- Do not narrate your inspection.\n"
        "- If you are unsure, return unresolved JSON instead of prose.\n"
    )


def default_model_for_tool(tool: str) -> str:
    if tool == "copilot":
        return DEFAULT_COPILOT_MODEL
    return ""


def build_prompt_file_launcher(prompt_path: Path) -> str:
    return (
        "Read and follow the full task instructions from this UTF-8 file:\n"
        f"{prompt_path}\n\n"
        "Before doing any other work:\n"
        "1. Open THAT EXACT FILE PATH and read it completely.\n"
        "2. Treat it as the authoritative prompt for this run.\n"
        "3. Do not search for a different task, instruction, or spec file anywhere else.\n"
        "4. Stay inside the current repo unless the prompt explicitly requires another path.\n"
        "5. Follow the prompt's required output contract exactly.\n"
        "6. Do not ask the user questions.\n\n"
        "Do not print the file contents back. Execute the task and exit when done.\n"
    )


def coerce_process_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def build_unavailable_usage(tool: str, model: str, prompt: str, duration_seconds: float, timed_out: bool, returncode: int, raw_output: str, output_path: Path, notes: Sequence[str]) -> CopilotInvocation:
    return CopilotInvocation(
        tool=tool,
        model=model,
        prompt_bytes=len(prompt.encode("utf-8")),
        duration_seconds=round(duration_seconds, 3),
        timed_out=timed_out,
        returncode=returncode,
        output_bytes=len(raw_output.encode("utf-8")) if raw_output else 0,
        output_path=str(output_path),
        usage_available=False,
        usage_source="unavailable",
        usage_notes=list(notes),
    )


def run_copilot_prompt(
    prompt: str,
    *,
    cwd: Path,
    output_path: Path,
    model: Optional[str],
    timeout_seconds: int,
    available_tools: Optional[Sequence[str]] = None,
    silent: bool = False,
    stream_mode: str = "on",
    reasoning_effort: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    copilot_command = resolve_copilot_cli_command()
    if not copilot_command:
        raise RuntimeError("GitHub Copilot CLI was not found on PATH")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_tools = list(available_tools or ("apply_patch", "rg", "view", "glob", "report_intent"))
    command = [
        copilot_command,
        f"--available-tools={','.join(selected_tools)}",
        "--disable-builtin-mcps",
        "--allow-all-tools",
        "--allow-all-paths",
        "--allow-all-urls",
        "--no-ask-user",
        "--stream",
        stream_mode,
        "--no-custom-instructions",
    ]
    if silent:
        command.append("-s")
    selected_model = model or default_model_for_tool("copilot")
    if selected_model:
        command.extend(["--model", selected_model])
    if reasoning_effort:
        command.extend(["--effort", reasoning_effort])

    t0 = time.time()
    raw_output = ""
    timed_out = False
    process: Optional[subprocess.Popen[str]] = None
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = process.communicate(prompt, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        terminate_process_tree(process)
        tail_stdout, tail_stderr = process.communicate()
        stdout = coerce_process_output(exc.stdout) + coerce_process_output(tail_stdout)
        stderr = coerce_process_output(exc.stderr) + coerce_process_output(tail_stderr)
    raw_output = "\n".join([part for part in ((stdout or "").strip(), (stderr or "").strip()) if part]).strip()
    output_path.write_text(raw_output, encoding="utf-8")

    invocation = build_unavailable_usage(
        tool="copilot",
        model=selected_model,
        prompt=prompt,
        duration_seconds=time.time() - t0,
        timed_out=timed_out,
        returncode=process.returncode if process is not None and process.returncode is not None else -1,
        raw_output=raw_output,
        output_path=output_path,
        notes=[
            "GitHub Copilot CLI does not expose usage or quota details through this harness path.",
            "Telemetry includes invocation count, duration, return code, prompt size, and output size only.",
        ],
    )

    if timed_out:
        return raw_output, invocation.to_dict()
    if process is not None and process.returncode != 0:
        error_text = raw_output.splitlines()
        detail = error_text[-1] if error_text else "copilot failed"
        raise RuntimeError(detail)
    return raw_output, invocation.to_dict()


def run_agent_prompt(
    prompt: str,
    *,
    cwd: Path,
    output_path: Path,
    model: Optional[str],
    tool: str = DEFAULT_TOOL,
    timeout_seconds: int = COPILOT_TIMEOUT_SECONDS,
    available_tools: Optional[Sequence[str]] = None,
    silent: bool = False,
    stream_mode: str = "on",
    reasoning_effort: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    if tool != "copilot":
        raise RuntimeError(f"Unsupported agent tool `{tool}`. This harness is wired for GitHub Copilot CLI.")
    return run_copilot_prompt(
        prompt,
        cwd=cwd,
        output_path=output_path,
        model=model,
        timeout_seconds=timeout_seconds,
        available_tools=available_tools,
        silent=silent,
        stream_mode=stream_mode,
        reasoning_effort=reasoning_effort,
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    def normalize_json_keys(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                normalize_identifier(str(key)): normalize_json_keys(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [normalize_json_keys(item) for item in value]
        return value

    def normalize_candidate(candidate: str) -> str:
        normalized: List[str] = []
        in_string = False
        escape = False
        index = 0
        candidate_length = len(candidate)
        while index < candidate_length:
            character = candidate[index]
            if in_string:
                if escape:
                    normalized.append(character)
                    escape = False
                    index += 1
                    continue
                if character == "\\":
                    next_character = candidate[index + 1] if index + 1 < candidate_length else ""
                    if next_character == '"':
                        lookahead = index + 2
                        while lookahead < candidate_length and candidate[lookahead].isspace():
                            lookahead += 1
                        if lookahead >= candidate_length or candidate[lookahead] in ',]}:':
                            normalized.append("\\\\")
                            index += 1
                            continue
                    if next_character == "u":
                        hex_digits = candidate[index + 2 : index + 6]
                        if len(hex_digits) < 4 or any(ch not in "0123456789abcdefABCDEF" for ch in hex_digits):
                            normalized.append("/")
                            index += 1
                            continue
                    elif next_character and next_character not in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
                        normalized.append("/")
                        index += 1
                        continue
                    normalized.append(character)
                    escape = True
                    index += 1
                    continue
                if character in "\r\n\t":
                    lookahead = index + 1
                    while lookahead < candidate_length and candidate[lookahead] in " \t\r\n":
                        lookahead += 1
                    previous = normalized[-1] if normalized else ""
                    next_character = candidate[lookahead] if lookahead < candidate_length else ""
                    if (
                        previous
                        and next_character
                        and previous.isalnum()
                        and next_character.isalnum()
                        and previous not in "/._-"
                        and next_character not in "/._-"
                        and (not normalized or normalized[-1] != " ")
                    ):
                        normalized.append(" ")
                    index = lookahead
                    continue
                normalized.append(character)
                if character == '"':
                    in_string = False
                index += 1
                continue

            if character in " \r\n\t":
                index += 1
                continue
            normalized.append(character)
            if character == '"':
                in_string = True
            index += 1
        return "".join(normalized)

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        broad_candidate = normalize_candidate(text[first_brace : last_brace + 1])
        try:
            broad_payload = json.loads(broad_candidate)
        except json.JSONDecodeError:
            broad_payload = None
        if isinstance(broad_payload, dict):
            return normalize_json_keys(broad_payload)

    best_payload: Optional[Dict[str, Any]] = None
    best_span = -1
    text_length = len(text)
    for start_index, character in enumerate(text):
        if character != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for end_index in range(start_index, text_length):
            current = text[end_index]
            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    in_string = False
                continue

            if current == '"':
                in_string = True
                continue
            if current == "{":
                depth += 1
                continue
            if current != "}":
                continue
            depth -= 1
            if depth != 0:
                continue
            candidate = normalize_candidate(text[start_index : end_index + 1])
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                break
            if isinstance(payload, dict) and (end_index - start_index) > best_span:
                best_payload = normalize_json_keys(payload)
                best_span = end_index - start_index
            break
    if best_payload is not None:
        return best_payload
    raise ValueError("Failed to locate a JSON object in the evaluator output")


def validate_criteria_rows(payload: Dict[str, Any], benchmark: Benchmark) -> List[Dict[str, Any]]:
    expected_ids = {criterion.id for criterion in benchmark.mandatory_criteria}
    criteria: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in payload.get("criteria", []):
        criterion_id = normalize_identifier(row.get("id", ""))
        if not criterion_id or criterion_id not in expected_ids or criterion_id in seen_ids:
            continue
        seen_ids.add(criterion_id)
        criteria.append(
            {
                "id": criterion_id,
                "result": collapse_whitespace(str(row.get("result", ""))),
                "reason": collapse_whitespace(str(row.get("reason", ""))),
            }
        )
    return criteria


def validate_measurement(payload: Dict[str, Any], benchmark: Benchmark) -> Dict[str, Any]:
    if payload.get("status") != "resolved":
        missing = payload.get("missing") or []
        raise ValueError(f"Evaluator returned unresolved measurement: {missing or payload}")

    g = payload.get("g")
    if g not in {0, 1}:
        raise ValueError(f"Measurement g must be 0 or 1, got {g}")

    cosmic = Cosmic.from_dict(payload["cosmic"])
    cosmic.validate()

    criteria = validate_criteria_rows(payload, benchmark)
    if benchmark.dimensions:
        expected_ids = {dimension.id for dimension in benchmark.dimensions}
        dimension_rows: Dict[str, Dict[str, Any]] = {}
        for row in payload.get("dimension_rows", []):
            dimension_id = normalize_identifier(row.get("id", ""))
            if dimension_id not in expected_ids:
                raise ValueError(f"Unexpected dimension row for `{dimension_id}`")
            if dimension_id in dimension_rows:
                raise ValueError(f"Duplicate dimension row for `{dimension_id}`")
            try:
                delta = float(row.get("delta"))
                confidence = float(row.get("confidence"))
                observability = float(row.get("observability"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric values for dimension `{dimension_id}`") from exc
            if not (-1.0 <= delta <= 1.0):
                raise ValueError(f"delta for dimension `{dimension_id}` must be between -1.0 and 1.0")
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"confidence for dimension `{dimension_id}` must be between 0.0 and 1.0")
            if not (0.0 <= observability <= 1.0):
                raise ValueError(f"observability for dimension `{dimension_id}` must be between 0.0 and 1.0")
            dimension_rows[dimension_id] = {
                "id": dimension_id,
                "delta": delta,
                "confidence": confidence,
                "observability": observability,
                "reason": collapse_whitespace(str(row.get("reason", ""))),
            }

        if set(dimension_rows) != expected_ids:
            missing = sorted(expected_ids - set(dimension_rows))
            raise ValueError(f"Missing dimension rows for: {missing}")

        adjusted_weights = adjusted_dimension_weights([dimension.w for dimension in benchmark.dimensions])
        effective_weight = 0.0
        for index, dimension in enumerate(benchmark.dimensions):
            row = dimension_rows[dimension.id]
            effective_weight += adjusted_weights[index] * dimension_evidence_strength(
                row["confidence"],
                row["observability"],
            )
        if effective_weight < MIN_DIMENSION_EFFECTIVE_WEIGHT:
            raise ValueError(
                f"Evaluator provided insufficient effective benchmark coverage: {effective_weight:.3f} < {MIN_DIMENSION_EFFECTIVE_WEIGHT:.3f}"
            )

        ordered_rows = [dimension_rows[dimension.id] for dimension in benchmark.dimensions]
        return {
            "status": "resolved",
            "g": g,
            "cosmic": asdict(cosmic),
            "criteria": criteria,
            "dimension_rows": ordered_rows,
        }

    expected_keys = {(task.persona_id, task.task_id) for task in benchmark.tasks}
    feature_map: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in payload.get("feature_rows", []):
        key = (
            normalize_identifier(row.get("persona_id", "")),
            normalize_identifier(row.get("task_id", "")),
        )
        if key not in expected_keys:
            raise ValueError(f"Unexpected feature row for `{key[0]}` / `{key[1]}`")
        if key in feature_map:
            raise ValueError(f"Duplicate feature row for `{key[0]}` / `{key[1]}`")
        scenario = Scenario.from_dict(row["feature"])
        scenario.validate(f"feature_rows[{key[0]}/{key[1]}].feature")
        feature_map[key] = scenario.to_dict()

    if set(feature_map) != expected_keys:
        missing = sorted(expected_keys - set(feature_map))
        raise ValueError(f"Missing feature rows for: {missing}")

    feature_rows = [
        {"persona_id": task.persona_id, "task_id": task.task_id, "feature": feature_map[(task.persona_id, task.task_id)]}
        for task in benchmark.tasks
    ]
    return {
        "status": "resolved",
        "g": g,
        "cosmic": asdict(cosmic),
        "criteria": criteria,
        "feature_rows": feature_rows,
    }


def measurement_has_all_zero_deltas(measurement: Dict[str, Any]) -> bool:
    rows = measurement.get("dimension_rows") or []
    if not rows:
        return False
    return all(abs(float(row.get("delta", 0.0))) <= EPSILON for row in rows)


def measurement_copies_task_baseline(benchmark: Benchmark, measurement: Dict[str, Any]) -> bool:
    rows = measurement.get("feature_rows") or []
    if not benchmark.tasks or not rows:
        return False
    feature_lookup = {
        (str(row.get("persona_id", "")), str(row.get("task_id", ""))): row.get("feature", {})
        for row in rows
        if isinstance(row, dict)
    }
    expected_keys = {(task.persona_id, task.task_id) for task in benchmark.tasks}
    if set(feature_lookup) != expected_keys:
        return False
    for task in benchmark.tasks:
        if feature_lookup[(task.persona_id, task.task_id)] != task.baseline.to_dict():
            return False
    return True


def measurement_needs_discrimination(benchmark: Benchmark, measurement: Dict[str, Any], *, require_nonzero_delta: bool) -> bool:
    if not require_nonzero_delta:
        return False
    if int(measurement.get("g", 0)) != 1:
        return False
    if benchmark.dimensions:
        return measurement_has_all_zero_deltas(measurement)
    return measurement_copies_task_baseline(benchmark, measurement)


def apply_conservative_discriminative_fallback(
    benchmark: Benchmark,
    measurement: Dict[str, Any],
    *,
    reason: str,
) -> Dict[str, Any]:
    note = collapse_whitespace(reason) or "Evaluator failed to discriminate a changed passing candidate."
    updated = dict(measurement)
    updated["criteria"] = [
        {
            **row,
            "reason": collapse_whitespace(
                f"{str(row.get('reason', '')).strip()} {note}".strip()
            ),
        }
        for row in measurement.get("criteria", [])
    ]
    if benchmark.dimensions:
        updated["dimension_rows"] = [
            {
                **row,
                "delta": -RESCUE_SIGNED_DELTA if abs(float(row.get("delta", 0.0))) <= EPSILON else float(row.get("delta", 0.0)),
                "confidence": max(float(row.get("confidence", 0.0)), RESCUE_MIN_CONFIDENCE),
                "observability": max(float(row.get("observability", 0.0)), RESCUE_MIN_OBSERVABILITY),
                "reason": collapse_whitespace(
                    f"{str(row.get('reason', '')).strip()} Conservative slight-negative fallback applied because the evaluator would not return a discriminative score for this changed passing candidate."
                ),
            }
            for row in measurement.get("dimension_rows", [])
        ]
        return updated

    nudged_rows: List[Dict[str, Any]] = []
    nudged = False
    for row in measurement.get("feature_rows", []):
        new_row = {
            "persona_id": str(row.get("persona_id", "")),
            "task_id": str(row.get("task_id", "")),
            "feature": dict(row.get("feature", {})),
        }
        if not nudged:
            feature = dict(new_row["feature"])
            if isinstance(feature.get("response_seconds_total"), (int, float)):
                feature["response_seconds_total"] = float(feature["response_seconds_total"]) + 0.1
            else:
                feature["p"] = int(feature.get("p", 0)) + 1
            new_row["feature"] = feature
            nudged = True
        nudged_rows.append(new_row)
    updated["feature_rows"] = nudged_rows
    return updated


def combine_usage_blocks(*blocks: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    valid = [block for block in blocks if block]
    if not valid:
        return merged
    first = dict(valid[0])
    merged.update(first)
    merged["duration_seconds"] = round(sum(float(block.get("duration_seconds", 0.0)) for block in valid), 3)
    merged["prompt_bytes"] = sum(int(block.get("prompt_bytes", 0)) for block in valid)
    merged["output_bytes"] = sum(int(block.get("output_bytes", 0)) for block in valid)
    merged["usage_notes"] = [
        note
        for block in valid
        for note in block.get("usage_notes", [])
    ]
    return merged


def fallback_measurement(
    benchmark: Benchmark,
    build_results: Sequence[Dict[str, Any]],
    reason: str,
) -> Dict[str, Any]:
    build_ok = all(bool(result.get("ok")) for result in build_results)
    criteria = []
    for criterion in benchmark.mandatory_criteria:
        criteria.append(
            {
                "id": criterion.id,
                "result": "pass" if build_ok else "fail",
                "reason": (
                    "Fallback heuristic measurement used because the evaluator output could not be "
                    f"validated: {reason}"
                ),
            }
        )
    return {
        "status": "resolved",
        "g": 1 if build_ok else 0,
        "cosmic": {"entries": 0, "exits": 0, "reads": 0, "writes": 0},
        "criteria": criteria,
        "dimension_rows": [
            {
                "id": dimension.id,
                "delta": 0.0,
                "confidence": 0.0,
                "observability": 0.0,
                "reason": (
                    "Fallback heuristic measurement used because the evaluator output could not be "
                    f"validated: {reason}"
                ),
            }
            for dimension in benchmark.dimensions
        ],
        "feature_rows": [
            {
                "persona_id": task.persona_id,
                "task_id": task.task_id,
                "feature": task.baseline.to_dict(),
            }
            for task in benchmark.tasks
        ],
    }


def assemble_variables(benchmark: Benchmark, measurement: Dict[str, Any]) -> Dict[str, Any]:
    if benchmark.dimensions:
        dimension_lookup = {row["id"]: row for row in measurement["dimension_rows"]}
        return {
            "feature_id": benchmark.benchmark_id,
            "feature_name": benchmark.feature_name,
            "g": measurement["g"],
            "cosmic": measurement["cosmic"],
            "dimensions": [
                {
                    "id": dimension.id,
                    "description": dimension.description,
                    "baseline_anchor": dimension.baseline_anchor,
                    "w": dimension.w,
                    "delta": dimension_lookup[dimension.id]["delta"],
                    "confidence": dimension_lookup[dimension.id]["confidence"],
                    "observability": dimension_lookup[dimension.id]["observability"],
                    "reason": dimension_lookup[dimension.id]["reason"],
                }
                for dimension in benchmark.dimensions
            ],
        }
    feature_lookup = {
        (row["persona_id"], row["task_id"]): row["feature"] for row in measurement["feature_rows"]
    }
    return {
        "feature_id": benchmark.benchmark_id,
        "feature_name": benchmark.feature_name,
        "g": measurement["g"],
        "cosmic": measurement["cosmic"],
        "taus": asdict(benchmark.taus),
        "persona_weights": [weight.to_dict() for weight in benchmark.persona_weights],
        "tasks": [
            {
                "persona_id": task.persona_id,
                "task_id": task.task_id,
                "q": task.q,
                "baseline": task.baseline.to_dict(),
                "feature": feature_lookup[(task.persona_id, task.task_id)],
            }
            for task in benchmark.tasks
        ],
    }


def validate_variables(variables: Dict[str, Any]) -> None:
    if not str(variables.get("feature_name", "")).strip():
        raise ValueError("feature_name must not be empty")

    g = variables.get("g")
    if g not in {0, 1}:
        raise ValueError(f"g must be 0 or 1, got {g}")

    cosmic = Cosmic.from_dict(variables["cosmic"])
    cosmic.validate()

    if variables.get("dimensions"):
        dimensions = variables["dimensions"]
        if not dimensions:
            raise ValueError("at least one dimension is required")
        seen_ids: set[str] = set()
        total_weight = 0.0
        for row in dimensions:
            dimension_id = normalize_identifier(row.get("id", ""))
            description = collapse_whitespace(str(row.get("description", "")))
            baseline_anchor = collapse_whitespace(str(row.get("baseline_anchor", "")))
            if not dimension_id:
                raise ValueError("dimension id must not be empty")
            if dimension_id in seen_ids:
                raise ValueError(f"duplicate dimension `{dimension_id}`")
            seen_ids.add(dimension_id)
            if not description:
                raise ValueError(f"dimension `{dimension_id}` must have a description")
            if not baseline_anchor:
                raise ValueError(f"dimension `{dimension_id}` must have a baseline_anchor")
            w = float(row["w"])
            delta = float(row["delta"])
            confidence = float(row["confidence"])
            observability = float(row["observability"])
            if not (0.0 <= w <= 1.0):
                raise ValueError(f"dimension weight for `{dimension_id}` must be between 0.0 and 1.0")
            if not (-1.0 <= delta <= 1.0):
                raise ValueError(f"dimension delta for `{dimension_id}` must be between -1.0 and 1.0")
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"dimension confidence for `{dimension_id}` must be between 0.0 and 1.0")
            if not (0.0 <= observability <= 1.0):
                raise ValueError(f"dimension observability for `{dimension_id}` must be between 0.0 and 1.0")
            total_weight += w
        if abs(total_weight - 1.0) > EPSILON:
            raise ValueError(f"dimension weights must sum to 1.0, got {total_weight:.12f}")
        return

    taus = Taus.from_dict(variables["taus"])
    taus.validate()
    persona_weights = [PersonaWeight.from_dict(item) for item in variables["persona_weights"]]
    if not persona_weights:
        raise ValueError("at least one persona weight is required")

    persona_map: Dict[str, float] = {}
    for weight in persona_weights:
        weight.validate()
        if weight.persona_id in persona_map:
            raise ValueError(f"duplicate persona weight for `{weight.persona_id}`")
        persona_map[weight.persona_id] = weight.w

    persona_sum = sum(weight.w for weight in persona_weights)
    if abs(persona_sum - 1.0) > EPSILON:
        raise ValueError(f"persona weights must sum to 1.0, got {persona_sum:.12f}")

    tasks = variables["tasks"]
    if not tasks:
        raise ValueError("at least one task is required")

    q_sums: Dict[str, float] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for task in tasks:
        persona_id = str(task["persona_id"])
        task_id = str(task["task_id"])
        q = float(task["q"])
        if persona_id not in persona_map:
            raise ValueError(f"task `{task_id}` references persona `{persona_id}` with no persona weight")
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"q for persona `{persona_id}` task `{task_id}` must be between 0.0 and 1.0")
        key = (persona_id, task_id)
        if key in seen_pairs:
            raise ValueError(f"duplicate persona/task pair: `{persona_id}` / `{task_id}`")
        seen_pairs.add(key)
        Scenario.from_dict(task["baseline"]).validate(f"tasks[{persona_id}/{task_id}].baseline")
        Scenario.from_dict(task["feature"]).validate(f"tasks[{persona_id}/{task_id}].feature")
        q_sums[persona_id] = q_sums.get(persona_id, 0.0) + q

    for weight in persona_weights:
        total = q_sums.get(weight.persona_id, 0.0)
        if abs(total - 1.0) > EPSILON:
            raise ValueError(
                f"task weights q for persona `{weight.persona_id}` must sum to 1.0, got {total:.12f}"
            )


def score_variables_python(variables: Dict[str, Any]) -> Dict[str, Any]:
    validate_variables(variables)

    cosmic = Cosmic.from_dict(variables["cosmic"])
    if variables.get("dimensions"):
        dimension_results: List[Dict[str, Any]] = []
        raw_weights = [float(row["w"]) for row in variables["dimensions"]]
        scoring_weights = adjusted_dimension_weights(raw_weights)
        observable_weight = 0.0
        effective_weight = 0.0
        dimension_signal = 0.0

        for row, scoring_w in zip(variables["dimensions"], scoring_weights):
            w = float(row["w"])
            delta = float(row["delta"])
            confidence = float(row["confidence"])
            observability = float(row["observability"])
            evidence_strength = dimension_evidence_strength(confidence, observability)
            observable_contribution = scoring_w * observability
            effective_contribution = scoring_w * evidence_strength
            signal_contribution = effective_contribution * delta
            observable_weight += observable_contribution
            effective_weight += effective_contribution
            dimension_signal += signal_contribution
            dimension_results.append(
                {
                    "id": str(row["id"]),
                    "w": w,
                    "scoring_w": scoring_w,
                    "delta": delta,
                    "confidence": confidence,
                    "observability": observability,
                    "evidence_strength": evidence_strength,
                    "observable_contribution": observable_contribution,
                    "effective_contribution": effective_contribution,
                    "signal_contribution": signal_contribution,
                    "baseline_anchor": str(row.get("baseline_anchor", "")),
                    "reason": str(row.get("reason", "")),
                }
            )

        if effective_weight < MIN_DIMENSION_EFFECTIVE_WEIGHT:
            raise ValueError(
                f"effective dimension weight must be >= {MIN_DIMENSION_EFFECTIVE_WEIGHT:.3f}, got {effective_weight:.3f}"
            )

        cfp = cosmic.cfp()
        ungated_score_pct = 100.0 * dimension_signal
        score_pct = 100.0 * float(variables["g"]) * dimension_signal
        value_density = dimension_signal / cfp if cfp > 0 else None

        return {
            "feature_id": variables.get("feature_id"),
            "feature_name": variables["feature_name"],
            "formulas": {
                "dimension_weighting": f"w'_i = normalize(w_i^{DIMENSION_WEIGHT_EXPONENT})",
                "dimension_evidence_strength": f"e_i = (confidence_i * observability_i)^{DIMENSION_EVIDENCE_EXPONENT}",
                "dimension_signal": "S = sum_i (w'_i * e_i * delta_i)",
                "score_pct": "score_pct = 100 * g * S",
                "value_density_signal_per_cfp": "value_density_signal_per_cfp = S / CFP",
                "cosmic_function_points": "CFP = Entries + Exits + Reads + Writes",
            },
            "variables_used": {
                "g": variables["g"],
                "cosmic": variables["cosmic"],
                "dimension_weight_exponent": DIMENSION_WEIGHT_EXPONENT,
                "dimension_evidence_exponent": DIMENSION_EVIDENCE_EXPONENT,
            },
            "dimension_results": dimension_results,
            "score": {
                "cosmic_function_points": cfp,
                "observable_weight": observable_weight,
                "effective_weight": effective_weight,
                "dimension_signal": dimension_signal,
                "ungated_score_pct": ungated_score_pct,
                "score_pct": score_pct,
                "value_density_signal_per_cfp": value_density,
            },
        }

    taus = Taus.from_dict(variables["taus"])
    persona_weights = {item["persona_id"]: float(item["w"]) for item in variables["persona_weights"]}

    weighted_baseline_seconds = 0.0
    weighted_feature_seconds = 0.0
    task_results: List[Dict[str, Any]] = []
    task_weight_sums: Dict[str, float] = {}

    for task in variables["tasks"]:
        persona_id = str(task["persona_id"])
        task_id = str(task["task_id"])
        w = persona_weights[persona_id]
        q = float(task["q"])
        baseline = Scenario.from_dict(task["baseline"])
        feature = Scenario.from_dict(task["feature"])
        baseline_seconds = baseline.predicted_time_seconds(taus)
        feature_seconds = feature.predicted_time_seconds(taus)
        delta_seconds = baseline_seconds - feature_seconds
        weighted_baseline = w * q * baseline_seconds
        weighted_feature = w * q * feature_seconds
        weighted_delta = w * q * delta_seconds

        weighted_baseline_seconds += weighted_baseline
        weighted_feature_seconds += weighted_feature
        task_weight_sums[persona_id] = task_weight_sums.get(persona_id, 0.0) + q

        task_results.append(
            {
                "persona_id": persona_id,
                "task_id": task_id,
                "w": w,
                "q": q,
                "baseline_seconds": baseline_seconds,
                "feature_seconds": feature_seconds,
                "delta_seconds": delta_seconds,
                "weighted_baseline_seconds": weighted_baseline,
                "weighted_feature_seconds": weighted_feature,
                "weighted_delta_seconds": weighted_delta,
            }
        )

    if weighted_baseline_seconds <= 0.0:
        raise ValueError("weighted baseline time must be > 0")

    weighted_delta_seconds = weighted_baseline_seconds - weighted_feature_seconds
    ungated_time_gain_ratio = weighted_delta_seconds / weighted_baseline_seconds
    ungated_score_pct = 100.0 * ungated_time_gain_ratio
    score_pct = 100.0 * float(variables["g"]) * weighted_delta_seconds / weighted_baseline_seconds
    cfp = cosmic.cfp()
    value_density = weighted_delta_seconds / cfp if cfp > 0 else None

    return {
        "feature_id": variables.get("feature_id"),
        "feature_name": variables["feature_name"],
        "formulas": {
            "baseline_seconds": "B = sum_p w_p * sum_t q_(p,t) * T_baseline(p,t)",
            "feature_seconds": "F = sum_p w_p * sum_t q_(p,t) * T_feature(p,t)",
            "delta_seconds": "Delta = B - F",
            "score_pct": "score_pct = 100 * g * Delta / B",
            "task_time_seconds": (
                "T = K*tau_k + P*tau_p + H*tau_h + D*tau_d + M*tau_m + rho, "
                "where rho = response_seconds_total if provided, else R*tau_r"
            ),
            "cosmic_function_points": "CFP = Entries + Exits + Reads + Writes",
            "value_density_seconds_saved_per_cfp": "value_density_seconds_saved_per_cfp = Delta / CFP",
        },
        "variables_used": {
            "g": variables["g"],
            "cosmic": variables["cosmic"],
            "taus": variables["taus"],
            "persona_weights": variables["persona_weights"],
            "task_weight_sums_by_persona": task_weight_sums,
        },
        "task_results": task_results,
        "score": {
            "cosmic_function_points": cfp,
            "weighted_baseline_seconds": weighted_baseline_seconds,
            "weighted_feature_seconds": weighted_feature_seconds,
            "weighted_delta_seconds": weighted_delta_seconds,
            "ungated_time_gain_ratio": ungated_time_gain_ratio,
            "ungated_score_pct": ungated_score_pct,
            "score_pct": score_pct,
            "value_density_seconds_saved_per_cfp": value_density,
        },
    }


def score_variables_rust(
    variables: Dict[str, Any],
    *,
    deadline_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    process = subprocess.Popen(
        [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--manifest-path",
            str(RUST_ASSESSOR_MANIFEST),
            "--",
            "score",
            "-",
        ],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    payload = json.dumps(variables)
    try:
        stdout, stderr = process.communicate(
            input=payload,
            timeout=bounded_timeout(
                SCORER_TIMEOUT_SECONDS,
                deadline_monotonic=deadline_monotonic,
                label="rust scorer",
            ),
        )
    except subprocess.TimeoutExpired:
        terminate_process_tree(process)
        stdout, stderr = process.communicate()
        raise RuntimeError("Rust scorer timed out")
    if process.returncode != 0:
        error_text = (stderr or stdout).strip().splitlines()
        detail = error_text[-1] if error_text else "cargo run failed"
        raise RuntimeError(detail)
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Rust scorer produced invalid JSON: {exc}") from exc


def score_variables(
    variables: Dict[str, Any],
    backend: str = DEFAULT_BACKEND,
    *,
    deadline_monotonic: Optional[float] = None,
) -> tuple[Dict[str, Any], str]:
    if variables.get("dimensions"):
        return score_variables_python(variables), "python"
    selected = choose_backend(backend)
    if selected == "python":
        return score_variables_python(variables), "python"
    try:
        return score_variables_rust(variables, deadline_monotonic=deadline_monotonic), "rust"
    except Exception:
        if backend != "auto":
            raise
        return score_variables_python(variables), "python"


def run_validated_evaluator_prompt(
    prompt: str,
    *,
    benchmark: Benchmark,
    cwd: Path,
    output_path: Path,
    model: Optional[str],
    tool: str,
    timeout_seconds: int,
) -> tuple[Optional[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    raw_message, usage = run_agent_prompt(
        prompt,
        cwd=cwd,
        output_path=output_path,
        model=model,
        tool=tool,
        timeout_seconds=timeout_seconds,
        available_tools=("rg", "view", "glob"),
        silent=True,
        stream_mode="off",
        reasoning_effort="low",
    )
    try:
        measurement = validate_measurement(extract_json_object(raw_message), benchmark)
    except Exception as exc:
        return None, usage, str(exc)
    return measurement, usage, None


def write_debug_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def evaluate_worktree(
    *,
    backend: str = DEFAULT_BACKEND,
    tool: str = DEFAULT_TOOL,
    model: Optional[str] = None,
    deadline_monotonic: Optional[float] = None,
    require_nonzero_delta: bool = False,
) -> Dict[str, Any]:
    benchmark, benchmark_source = get_active_benchmark(
        tool=tool,
        model=model,
        allow_synthesis=True,
        deadline_monotonic=deadline_monotonic,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_root = Path(tempfile.mkdtemp(prefix="autoresearch-eval-", dir=CACHE_DIR))
    try:
        build_snapshot(ROOT, snapshot_root)
        build_results = run_build_commands(
            benchmark.build_commands,
            cwd=snapshot_root,
            deadline_monotonic=deadline_monotonic,
        )
        evaluator_prompt = build_evaluator_prompt(benchmark, build_results)
        measurement, copilot_usage, measurement_error = run_validated_evaluator_prompt(
            evaluator_prompt,
            benchmark=benchmark,
            cwd=snapshot_root,
            output_path=LAST_EVALUATOR_MESSAGE,
            model=model,
            tool=tool,
            timeout_seconds=bounded_timeout(
                EVALUATOR_TIMEOUT_SECONDS,
                deadline_monotonic=deadline_monotonic,
                label="evaluator",
            ),
        )
        if measurement is None:
            retry_measurement, retry_usage, retry_error = run_validated_evaluator_prompt(
                build_json_repair_prompt(
                    benchmark,
                    build_results,
                    prior_error=measurement_error or "Failed to parse evaluator output.",
                ),
                benchmark=benchmark,
                cwd=snapshot_root,
                output_path=LAST_EVALUATOR_MESSAGE,
                model=model,
                tool=tool,
                timeout_seconds=bounded_timeout(
                    ALL_ZERO_RETRY_TIMEOUT_SECONDS,
                    deadline_monotonic=deadline_monotonic,
                    label="evaluator JSON repair retry",
                ),
            )
            copilot_usage = combine_usage_blocks(copilot_usage, retry_usage)
            if retry_measurement is not None:
                measurement = retry_measurement
            else:
                rescue_measurement, rescue_usage, rescue_error = run_validated_evaluator_prompt(
                    build_json_repair_prompt(
                        benchmark,
                        build_results,
                        prior_error="; ".join(part for part in (measurement_error, retry_error) if part),
                        final_pass=True,
                    ),
                    benchmark=benchmark,
                    cwd=snapshot_root,
                    output_path=LAST_EVALUATOR_MESSAGE,
                    model=model,
                    tool=tool,
                    timeout_seconds=bounded_timeout(
                        ALL_ZERO_RETRY_TIMEOUT_SECONDS,
                        deadline_monotonic=deadline_monotonic,
                        label="evaluator JSON repair rescue",
                    ),
                )
                copilot_usage = combine_usage_blocks(copilot_usage, rescue_usage)
                if rescue_measurement is not None:
                    measurement = rescue_measurement
                else:
                    failure_reason = "; ".join(
                        part for part in (measurement_error, retry_error, rescue_error) if part
                    )
                    raise RuntimeError(f"Evaluator measurement failed: {failure_reason}")

        if measurement_needs_discrimination(benchmark, measurement, require_nonzero_delta=require_nonzero_delta):
            retry_measurement, retry_usage, retry_error = run_validated_evaluator_prompt(
                build_discriminative_retry_prompt(benchmark, build_results),
                benchmark=benchmark,
                cwd=snapshot_root,
                output_path=LAST_EVALUATOR_MESSAGE,
                model=model,
                tool=tool,
                timeout_seconds=bounded_timeout(
                    ALL_ZERO_RETRY_TIMEOUT_SECONDS,
                    deadline_monotonic=deadline_monotonic,
                    label="all-zero evaluator retry",
                ),
            )
            copilot_usage = combine_usage_blocks(copilot_usage, retry_usage)
            if retry_measurement is not None and not measurement_needs_discrimination(
                benchmark,
                retry_measurement,
                require_nonzero_delta=require_nonzero_delta,
            ):
                measurement = retry_measurement
            else:
                rescue_measurement, rescue_usage, rescue_error = run_validated_evaluator_prompt(
                    build_discriminative_rescue_prompt(benchmark, build_results),
                    benchmark=benchmark,
                    cwd=snapshot_root,
                    output_path=LAST_EVALUATOR_MESSAGE,
                    model=model,
                    tool=tool,
                    timeout_seconds=bounded_timeout(
                        ALL_ZERO_RETRY_TIMEOUT_SECONDS,
                        deadline_monotonic=deadline_monotonic,
                        label="discriminative evaluator rescue",
                    ),
                )
                copilot_usage = combine_usage_blocks(copilot_usage, rescue_usage)
                if rescue_measurement is not None and not measurement_needs_discrimination(
                    benchmark,
                    rescue_measurement,
                    require_nonzero_delta=require_nonzero_delta,
                ):
                    measurement = rescue_measurement
                else:
                    failure_reason = "; ".join(
                        part
                        for part in (
                            retry_error,
                            rescue_error,
                            "Conservative discriminative fallback applied after nondiscriminative evaluator output.",
                        )
                        if part
                    )
                    measurement = apply_conservative_discriminative_fallback(
                        benchmark,
                        measurement,
                        reason=failure_reason,
                    )
    finally:
        shutil.rmtree(snapshot_root, ignore_errors=True)

    variables = assemble_variables(benchmark, measurement)
    assessment, scorer_backend = score_variables(
        variables,
        backend,
        deadline_monotonic=deadline_monotonic,
    )

    write_debug_json(LAST_BUILD_RESULTS_PATH, build_results)
    write_debug_json(LAST_MEASUREMENT_PATH, measurement)
    write_debug_json(LAST_VARIABLES_PATH, variables)

    report = {
        "benchmark_source": benchmark_source,
        "benchmark_id": benchmark.benchmark_id,
        "app_name": benchmark.app_name,
        "feature_name": benchmark.feature_name,
        "benchmark_payload": benchmark.to_prompt_payload(),
        "scorer_backend": scorer_backend,
        "copilot_usage": copilot_usage,
        "build_results": build_results,
        "measurement": measurement,
        "assessment": assessment,
    }
    write_debug_json(LAST_COPILOT_USAGE_PATH, copilot_usage)
    write_debug_json(LAST_SCORE_PATH, report)
    return report


def score_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    score_block = report["assessment"]["score"]
    summary = {
        "benchmark_source": report["benchmark_source"],
        "benchmark_id": report["benchmark_id"],
        "app_name": report["app_name"],
        "feature_name": report["feature_name"],
        "scorer_backend": report["scorer_backend"],
        "gate": report["measurement"]["g"],
        "score_pct": score_block["score_pct"],
        "build_commands_ran": len(report["build_results"]),
    }
    if "dimension_signal" in score_block:
        summary["dimension_signal"] = score_block["dimension_signal"]
        summary["effective_weight"] = score_block["effective_weight"]
        summary["observable_weight"] = score_block["observable_weight"]
    else:
        summary["weighted_delta_seconds"] = score_block["weighted_delta_seconds"]
        summary["weighted_feature_seconds"] = score_block["weighted_feature_seconds"]
    return summary


def emit(payload: Any, as_json: bool) -> None:
    if as_json:
        serializable = asdict(payload) if is_dataclass(payload) else payload
        print(json.dumps(serializable, indent=2, sort_keys=True))
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
        return
    for key, value in asdict(payload).items():
        print(f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and score the app-focused autoresearch benchmark.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=["auto", "python", "rust"],
        help="Scoring backend to use.",
    )
    parser.add_argument(
        "--tool",
        default=DEFAULT_TOOL,
        choices=[DEFAULT_TOOL],
        help="Agent tool to use for benchmark evaluation.",
    )
    parser.add_argument("--model", default=None, help="GitHub Copilot CLI model override for evaluator runs.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("check", help="Verify runtime prerequisites and benchmark discovery.")
    subparsers.add_parser("score", help="Evaluate the current repo snapshot against the active frozen benchmark.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command or "check"
    if command == "check":
        emit(check_environment(args.backend), args.json)
        return
    if command == "score":
        report = evaluate_worktree(backend=args.backend, tool=args.tool, model=args.model)
        emit(report if args.json else score_summary(report), args.json)
        return
    parser.error(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
