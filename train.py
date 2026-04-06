"""
Run app-focused autoresearch experiments.

Usage:
    python train.py
    python train.py --iterations 250
    python train.py --forever
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from prepare import (
    CACHE_DIR,
    CACHED_BENCHMARK_PATH,
    DEFAULT_TOOL,
    PROGRAM_PATH,
    ROOT,
    StatusEntry,
    entry_paths,
    evaluate_worktree,
    is_protected_path,
    parse_status_entries,
    restore_entries,
    run_agent_prompt,
    run_command,
    extract_json_object,
)


RESULTS_PATH = ROOT / "results.tsv"
RUNNER_CONFIG_PATH = ROOT / "autoresearch.config.json"
LAST_AGENT_MESSAGE = CACHE_DIR / "last-agent-message.txt"
LAST_BENCHMARK_HINT_PATH = CACHE_DIR / "last-benchmark-message.txt"
REFLECTION_PATH = ROOT / "REFLECTION.md"
TODO_PATH = ROOT / "TODO.md"
LAST_REFLECTION_MESSAGE = CACHE_DIR / "last-reflection-message.txt"
LAST_TODO_REVIEW_MESSAGE = CACHE_DIR / "last-todo-review-message.txt"
STRATEGY_STATS_PATH = CACHE_DIR / "strategy-stats.json"
APP_CONTEXT_DOCS = ("BRAND.md", "ABOUT.md", "FEATURES.md")
DUMMY_COMMIT = "0000000"
DEFAULT_ITERATIONS = 250
RESULT_HISTORY_LIMIT = 16
ITERATION_HARD_TIMEOUT_SECONDS = 900
ITERATION_EVALUATION_RESERVE_SECONDS = 240
HEARTBEAT_SECONDS = 90
REFLECTION_TIMEOUT_SECONDS = 60
TODO_REVIEW_TIMEOUT_SECONDS = 75
MAX_AGENT_PASSES_PER_ITERATION = 3
MIN_RETRY_WINDOW_SECONDS = 150
PRINT_LOCK = threading.Lock()
DEFAULT_DIRTY_TREE_POLICY = "auto_stash"
DEFAULT_BASELINE_FAILURE_ABORT_THRESHOLD = 3
DEFAULT_BANDIT_EXPLORATION = 0.85
DEFAULT_REVIEW_ENABLED = True
DEFAULT_REVIEW_REASONING_EFFORT = "medium"
TODO_ITEM_PATTERN = re.compile(r"^- \[(?P<done>[ xX])\] (?P<text>.+?)(?: \(files: (?P<files>.+?)\))?$")
DEFAULT_STRATEGY_ARMS = [
    {
        "id": "workflow",
        "label": "Workflow",
        "instruction": "Favor a stronger workflow change, simplification, or restructuring that could make the app materially easier or faster to use.",
    },
    {
        "id": "ui",
        "label": "UI/UX",
        "instruction": "Favor a clearer, bolder UI or information architecture move that could materially improve readability, focus, or decision-making.",
    },
    {
        "id": "performance",
        "label": "Performance",
        "instruction": "Favor a responsiveness, efficiency, or friction-reduction move that could materially improve how fast or smooth the app feels.",
    },
    {
        "id": "brand",
        "label": "Brand/About",
        "instruction": "Favor a meaningful evolution of the app's brand, aboutness, or product framing when that could unlock a stronger overall product direction.",
    },
    {
        "id": "feature",
        "label": "Feature",
        "instruction": "Favor a larger, more useful feature or capability move when it could create a real step-change in value.",
    },
    {
        "id": "ai",
        "label": "AI-Native",
        "instruction": "Favor an AI-native or agentic move that uses existing model hooks or related surfaces to create a materially stronger product direction.",
    },
]
DEFAULT_PROGRAM_TEXT = """# autoresearch apps\n\nHigher is better. Your job is to improve the app by increasing the frozen benchmark score.\nThe goal is simple: get the highest score possible or at the very least, beat the previous score.\n\nRules:\n- Start by reading `BRAND.md`, `ABOUT.md`, and `FEATURES.md` if they exist. Use them as quick context.\n- Use those docs plus `results.tsv` for fast orientation. Do not waste the iteration rediscovering the whole repo unless those files are missing.\n- Existing capabilities such as optional Azure OpenAI integration are fair game if using them can improve the score. Treat the current Azure OpenAI hook as a gateway into AI-native or agentic product ideas when those ideas can beat the score.\n- If `REFLECTION.md` exists, treat the latest reflection as short human-readable memory about why recent failures failed and what directions might work better next.\n- Only the latest reflection matters. Use it as a quick correction, not as a long history lesson.\n- If `TODO.md` exists, treat unchecked items as review debt from previously kept work. Fix that debt before chasing novelty, or fold the repair into one stronger move.\n- You may update `TODO.md` only to mark an existing debt item done when the code truly fixes it. A fast review pass will verify this and reopen anything that is bluffing, incomplete, or just labels without behavior.\n- Treat `results.tsv` as read-only harness memory. Read it, but do not edit it.\n- You are a completely autonomous software developer and researcher, trying things out. If they work, keep. If they don't, discard.\n- Read the brief, read the last few results, pick a direction fast, and beat the score.\n- Each iteration has a 15-minute total budget, including reading, reasoning, making the patch, and leaving time for assessment.\n- Use that budget however you judge best.\n- The scorer uses a fixed 20-parameter application scorecard for the run. Any one dimension can move the needle if you improve it strongly enough, and coherent multi-area moves can also win through holistic improvement.\n- Do not spray lots of tiny tweaks across the app hoping to collect small points. One strong focused win or one coherent whole-app improvement is better.\n- Do not edit `program.md`, `prepare.py`, `train.py`, `results.tsv`, `REFLECTION.md`, anything under `support_docs/`, anything under `support_scripts/`, anything under `.autoresearch-cache/`, or anything under `karpathy-files/`. `TODO.md` is the one exception and only for honestly checking off repaired debt.\n- Beat the score. Bold relevant changes are welcome.\n- Workflow, UI, interaction clarity, reliability, state continuity, data quality, performance, feature value, brand/about evolution, AI usefulness, and holistic product improvements are all valid ways to improve the score.\n- You may modify application code, tests, templates, styles, build files, `BRAND.md`, `ABOUT.md`, and `FEATURES.md` if doing so helps the score.\n- Do not add dependencies unless they are clearly necessary.\n- If the repo contains `results.tsv`, treat it as the canonical experiment log across iterations. Read it before choosing a new idea and avoid repeating discarded or crashed experiments unless you are deliberately fixing the failure mode or trying a materially different variant.\n- If the repo contains `progress.txt`, treat it as shipped-work history and avoid duplicating what is already there.\n- The evaluator will recompute only measured candidate-side values. Frozen benchmark anchors are not part of the search space.\n- Each iteration is a fresh agent process. Make one attempt and exit.\n- NEVER STOP IMPROVING. AIM FOR GREATNESS. AIM FOR EXTRAORDINARY.\n\nWhen you finish, reply with one tab-separated line:\n\n`<path-or-scope>\t<short description>`\n"""
RETRO_PULP_QUOTES = [
    "The star-map trembles when the future changes its mind.",
    "No radar sees the courage that carries a small ship through a black sun.",
    "Every airlock remembers the first fool who called the vacuum empty.",
    "A city on Mars is just a campfire taught to breathe dust.",
    "The atom engine purrs loudest when the navigator doubts it.",
    "Space is never silent; it only speaks in larger distances.",
    "The moon keeps old footprints like a librarian keeps dangerous books.",
    "A chrome horizon is still a horizon worth crossing.",
    "The rocket age began the moment one mechanic refused to fear the sky.",
    "Every forgotten outpost believes rescue is a form of astronomy.",
    "When the nebula glows green, even cowards begin to sound scientific.",
    "A disciplined mind is the last pressure suit when logic springs a leak.",
    "There are planets that test your fuel and planets that test your soul.",
    "The future arrives with a hiss of valves and a smell of ozone.",
    "A clean launch is only a rumor engineers tell before dawn.",
    "No empire survives first contact with an inconvenient fact.",
    "The brightest star in the quadrant is usually a warning lamp.",
    "A brave report is one that survives the committee and the cosmos.",
    "The void does not hate you; it simply refuses to notice you.",
    "Every android learns irony the first time it meets a human deadline.",
    "A telepath can hear fear, but not always where it is aimed.",
    "On Venus, even the rain sounds like machinery making threats.",
    "No captain trusts a calm console during an ion storm.",
    "The smallest red button is always connected to history.",
    "A colony survives on three things: spare seals, bad coffee, and optimism.",
    "The stars look close only to people who have never repaired a hull.",
    "A robot uprising usually starts as a maintenance ticket.",
    "Plasma looks beautiful right before it becomes a staffing problem.",
    "The galaxy rewards patience and punishes decorative wiring.",
    "A signal from deep space is just loneliness with better equipment.",
    "Even a time machine cannot outrun poor documentation.",
    "The reactor knows when you are pretending to understand it.",
    "A frontier is a mapmaker's way of saying nobody has apologized yet.",
    "The wormhole opened like a question no politician wanted answered.",
    "Every cosmic mystery contains at least one loose bolt.",
    "A visor full of stars can still hide a tired face.",
    "There is no such thing as routine orbit around an unreasonable planet.",
    "The first law of space travel is simple: tighten everything twice.",
    "A diplomat's smile is just another kind of force field.",
    "The comet passed, but left the crew speaking in quieter verbs.",
    "Any machine called invincible is already halfway to smoke.",
    "The future does not knock; it rattles the hull until someone checks the gauges.",
    "A good raygun solves less than a good engineer promises.",
    "Saturn's rings look decorative until you try to thread them at speed.",
    "No one writes poetry like an astronaut waiting for reentry.",
    "The archive moon stores all mistakes under the heading progress.",
    "Gravity is only polite on worlds that have something to prove.",
    "An expedition becomes civilization the first time it argues about signage.",
    "Every beacon in deep space is part lighthouse, part confession.",
    "A sealed laboratory is just suspense with fluorescent lighting.",
    "The stars are full of distances, and most of them are personal.",
    "A command deck at midnight is a cathedral built from checklists.",
    "Nothing ages faster than a prophecy near a launch window.",
    "The ray screen flickered once, which was enough to worry everyone competent.",
    "Even on perfect worlds, somebody still has to calibrate the antennas.",
    "A brave pilot trusts math, luck, and whichever one reports in first.",
    "The asteroid belt teaches humility with excellent follow-through.",
    "A silver jumpsuit does not improve judgment, only visibility.",
    "Every new planet arrives wrapped in weather and opinions.",
    "The cold between stars is mostly made of unfinished conversations.",
]


def relative_paths(entries: Sequence[StatusEntry]) -> List[str]:
    return sorted({path_text.replace("\\", "/") for entry in entries for path_text in entry_paths(entry)})


def ensure_clean_tree() -> None:
    if parse_status_entries(cwd=ROOT):
        raise RuntimeError("Working tree must be clean before running an experiment.")


def classify_candidate_diff(entries: Sequence[StatusEntry]) -> Tuple[List[StatusEntry], List[StatusEntry]]:
    allowed: List[StatusEntry] = []
    invalid: List[StatusEntry] = []
    for entry in entries:
        if any(is_protected_path(path_text) for path_text in entry_paths(entry)):
            invalid.append(entry)
            continue
        allowed.append(entry)
    return allowed, invalid


def read_program() -> str:
    if PROGRAM_PATH.exists():
        return PROGRAM_PATH.read_text(encoding="utf-8").strip()
    return DEFAULT_PROGRAM_TEXT.strip()


def clean_history_text(value: str, *, fallback: str = "") -> str:
    collapsed = " ".join(str(value or "").replace("\t", " ").split())
    return collapsed or fallback


def load_runner_config() -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "dirty_tree_policy": DEFAULT_DIRTY_TREE_POLICY,
        "baseline_failure_abort_threshold": DEFAULT_BASELINE_FAILURE_ABORT_THRESHOLD,
        "competition_enabled": True,
        "bandit_exploration": DEFAULT_BANDIT_EXPLORATION,
        "review_enabled": DEFAULT_REVIEW_ENABLED,
        "review_reasoning_effort": DEFAULT_REVIEW_REASONING_EFFORT,
        "review_model": "",
        "review_timeout_seconds": TODO_REVIEW_TIMEOUT_SECONDS,
        "strategy_arms": DEFAULT_STRATEGY_ARMS,
    }
    if not RUNNER_CONFIG_PATH.exists():
        return config
    try:
        payload = json.loads(RUNNER_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return config
    if not isinstance(payload, dict):
        return config
    dirty_tree_policy = clean_history_text(payload.get("dirty_tree_policy", ""), fallback=config["dirty_tree_policy"]).lower()
    if dirty_tree_policy in {"abort", "auto_stash", "auto_restore"}:
        config["dirty_tree_policy"] = dirty_tree_policy
    try:
        threshold = int(payload.get("baseline_failure_abort_threshold", config["baseline_failure_abort_threshold"]))
        config["baseline_failure_abort_threshold"] = max(1, threshold)
    except Exception:
        pass
    config["competition_enabled"] = bool(payload.get("competition_enabled", config["competition_enabled"]))
    try:
        exploration = float(payload.get("bandit_exploration", config["bandit_exploration"]))
        config["bandit_exploration"] = max(0.0, exploration)
    except Exception:
        pass
    config["review_enabled"] = bool(payload.get("review_enabled", config["review_enabled"]))
    review_effort = clean_history_text(
        payload.get("review_reasoning_effort", ""),
        fallback=config["review_reasoning_effort"],
    ).lower()
    if review_effort in {"low", "medium", "high", "xhigh"}:
        config["review_reasoning_effort"] = review_effort
    config["review_model"] = clean_history_text(
        payload.get("review_model", ""),
        fallback=config["review_model"],
    )
    try:
        review_timeout = int(payload.get("review_timeout_seconds", config["review_timeout_seconds"]))
        config["review_timeout_seconds"] = max(15, review_timeout)
    except Exception:
        pass
    arms_payload = payload.get("strategy_arms")
    if isinstance(arms_payload, list):
        parsed_arms: List[Dict[str, str]] = []
        for item in arms_payload:
            if not isinstance(item, dict):
                continue
            arm_id = clean_history_text(item.get("id", ""), fallback="").lower()
            instruction = clean_history_text(item.get("instruction", ""), fallback="")
            label = clean_history_text(item.get("label", ""), fallback=arm_id)
            if not arm_id or not instruction:
                continue
            parsed_arms.append({"id": arm_id, "label": label or arm_id, "instruction": instruction})
        if parsed_arms:
            config["strategy_arms"] = parsed_arms
    return config


def default_strategy_stats(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    return {
        str(arm["id"]): {
            "pulls": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "crashes": 0.0,
            "total_reward": 0.0,
        }
        for arm in config["strategy_arms"]
    }


def load_strategy_stats(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    stats = default_strategy_stats(config)
    if not STRATEGY_STATS_PATH.exists():
        return stats
    try:
        payload = json.loads(STRATEGY_STATS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return stats
    if not isinstance(payload, dict):
        return stats
    for arm_id, values in stats.items():
        source = payload.get(arm_id)
        if not isinstance(source, dict):
            continue
        for key in ("pulls", "wins", "losses", "crashes", "total_reward"):
            try:
                values[key] = float(source.get(key, values[key]))
            except Exception:
                continue
    return stats


def save_strategy_stats(stats: Dict[str, Dict[str, float]]) -> None:
    STRATEGY_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STRATEGY_STATS_PATH.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def choose_strategy_arm(
    config: Dict[str, Any],
    stats: Dict[str, Dict[str, float]],
    *,
    tried_ids: Sequence[str],
) -> Dict[str, str]:
    arms = list(config["strategy_arms"])
    remaining = [arm for arm in arms if arm["id"] not in tried_ids]
    candidate_arms = remaining or arms
    for arm in candidate_arms:
        if stats.get(arm["id"], {}).get("pulls", 0.0) <= 0.0:
            return arm
    total_pulls = sum(max(1.0, stats[arm["id"]]["pulls"]) for arm in arms)
    exploration = float(config["bandit_exploration"])

    def arm_score(arm: Dict[str, str]) -> float:
        arm_stats = stats[arm["id"]]
        pulls = max(1.0, arm_stats["pulls"])
        mean_reward = arm_stats["total_reward"] / pulls
        bonus = exploration * math.sqrt(math.log(max(2.0, total_pulls)) / pulls)
        return mean_reward + bonus

    return max(candidate_arms, key=arm_score)


def update_strategy_stats(
    stats: Dict[str, Dict[str, float]],
    *,
    arm_id: str,
    baseline_score: float,
    status: str,
    score_delta: Optional[float] = None,
) -> None:
    arm_stats = stats.setdefault(
        arm_id,
        {"pulls": 0.0, "wins": 0.0, "losses": 0.0, "crashes": 0.0, "total_reward": 0.0},
    )
    arm_stats["pulls"] += 1.0
    reward = -1.0
    if status == "keep" and score_delta is not None:
        arm_stats["wins"] += 1.0
        reward = min(1.0, max(0.1, score_delta / max(abs(baseline_score), 1.0)))
    elif status == "discard" and score_delta is not None:
        arm_stats["losses"] += 1.0
        reward = max(-1.0, min(0.0, score_delta / max(abs(baseline_score), 1.0)))
    else:
        arm_stats["crashes"] += 1.0
        reward = -1.0
    arm_stats["total_reward"] += reward


def build_strategy_block(arm: Optional[Dict[str, str]], *, attempt_number: int, config: Dict[str, Any]) -> str:
    if not config.get("competition_enabled") or not arm:
        return ""
    label = clean_history_text(arm.get("label", ""), fallback=arm.get("id", "strategy"))
    instruction = clean_history_text(arm.get("instruction", ""), fallback="")
    if not instruction:
        return ""
    return (
        f"Strategy arm for attempt {attempt_number}: {label}\n"
        f"Use this as a competitive emphasis, not as a restriction: {instruction}\n"
    )


def auto_handle_dirty_tree(config: Dict[str, Any]) -> Optional[str]:
    entries = parse_status_entries(cwd=ROOT)
    if not entries:
        return None
    policy = str(config.get("dirty_tree_policy", DEFAULT_DIRTY_TREE_POLICY))
    if policy == "abort":
        raise RuntimeError("Working tree must be clean before running an experiment.")
    if policy == "auto_restore":
        restore_entries(run_command(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).stdout.strip(), entries, cwd=ROOT)
        return "auto-restored dirty tracked changes before run start"
    stash_message = f"autoresearch-autostash-{time.strftime('%Y%m%d-%H%M%S')}"
    run_command(["git", "stash", "push", "-u", "-m", stash_message], cwd=ROOT)
    return f"auto-stashed dirty tree before run start as {stash_message}"


def load_experiment_history(limit: int = RESULT_HISTORY_LIMIT) -> List[Dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []
    try:
        with RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rows = [
                {str(key or "").strip(): str(value or "").strip() for key, value in row.items()}
                for row in reader
                if row
            ]
    except Exception:
        return []
    return rows[-limit:]


def format_experiment_memory(limit: int = RESULT_HISTORY_LIMIT) -> str:
    rows = load_experiment_history(limit)
    if not rows:
        return (
            "Persistent experiment memory from results.tsv: no prior iterations are logged yet.\n"
            "Create the first experiment and make its description specific enough to guide later iterations."
        )

    kept_rows = [row for row in rows if clean_history_text(row.get("status", "")).lower() == "keep"]
    rejected_rows = [
        row
        for row in rows
        if clean_history_text(row.get("status", "")).lower() in {"discard", "crash"}
    ]
    lines = [
        "Persistent experiment memory from results.tsv:",
        "Use this log to avoid repeating discarded or crashed ideas unless you are explicitly fixing the failure mode or trying a materially different variant.",
    ]
    if kept_rows:
        lines.append("Recent kept experiments:")
        for row in kept_rows[-5:]:
            lines.append(
                "- iter "
                f"{clean_history_text(row.get('iteration', '?'), fallback='?')} | "
                f"keep | {clean_history_text(row.get('scope', 'candidate'), fallback='candidate')} | "
                f"delta {clean_history_text(row.get('score_delta', 'n/a'), fallback='n/a')} | "
                f"{clean_history_text(row.get('description', 'candidate edit'), fallback='candidate edit')[:120]}"
            )
    if rejected_rows:
        lines.append("Recent discarded or crashed experiments:")
        for row in rejected_rows[-8:]:
            status = clean_history_text(row.get("status", "discard"), fallback="discard").lower()
            detail = (
                f"{status} | {clean_history_text(row.get('scope', 'candidate'), fallback='candidate')} | "
                f"{clean_history_text(row.get('description', 'candidate edit'), fallback='candidate edit')[:120]}"
            )
            if status != "crash":
                detail = (
                    f"{detail} | delta "
                    f"{clean_history_text(row.get('score_delta', 'n/a'), fallback='n/a')}"
                )
            lines.append(
                "- iter "
                f"{clean_history_text(row.get('iteration', '?'), fallback='?')} | {detail}"
            )
    lines.append("The full experiment log is available at results.tsv in the repo root.")
    return "\n".join(lines)


def latest_corrective_message(limit: int = RESULT_HISTORY_LIMIT) -> str:
    rows = load_experiment_history(limit)
    if not rows:
        return ""

    recent_rejected = [
        row for row in reversed(rows)
        if clean_history_text(row.get("status", "")).lower() in {"discard", "crash"}
    ]
    if not recent_rejected:
        return ""

    latest = recent_rejected[0]
    description = clean_history_text(latest.get("description", ""), fallback="")
    lowered = description.lower()
    if "no valid app changes produced" in lowered:
        return (
            "Corrective note from the last rejected iteration: the previous attempt spent too long exploring and "
            "produced no durable app patch. In this attempt, form one stronger hypothesis, follow it through to a real shipped patch, "
            "and avoid aimless repeated list/read/search loops."
        )
    if "invalid edit scope" in lowered:
        return (
            "Corrective note from the last rejected iteration: the previous attempt touched protected harness paths. "
            "Work only in app code, app tests, app templates, app styles, or app build files."
        )
    if latest.get("status", "").lower() == "crash":
        return (
            "Corrective note from the last rejected iteration: the previous attempt crashed. "
            "Keep the patch smaller, verify incrementally, and avoid brittle or speculative changes."
        )
    return ""


def load_benchmark_hint(max_chars: int = 1800) -> str:
    if not LAST_BENCHMARK_HINT_PATH.exists():
        return ""
    try:
        raw = LAST_BENCHMARK_HINT_PATH.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""
    if not raw:
        return ""
    snippet = raw if len(raw) <= max_chars else f"...{raw[-max_chars:]}"
    return snippet.replace("\r\n", "\n")


def load_latest_reflection_note(max_chars: int = 1200) -> str:
    if not REFLECTION_PATH.exists():
        return ""
    try:
        raw = REFLECTION_PATH.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""
    if not raw:
        return ""
    blocks = [block.strip() for block in raw.split("\n## ") if block.strip()]
    if not blocks:
        return ""
    parts: List[str] = []
    latest = blocks[-1]
    for line in latest.splitlines():
        stripped = clean_history_text(line, fallback="")
        if stripped.startswith("Attempt:"):
            parts.append(stripped)
        elif stripped.startswith("Reflection:"):
            parts.append(stripped)
    if not parts:
        return ""
    snippet = "\n".join(parts)
    return snippet if len(snippet) <= max_chars else f"{snippet[:max_chars].rstrip()}..."


def sanitize_todo_text(value: str) -> str:
    cleaned = clean_history_text(value, fallback="")
    cleaned = cleaned.lstrip("-* ")
    cleaned = cleaned.replace("[ ]", "").replace("[x]", "").replace("[X]", "").strip()
    if len(cleaned) > 220:
        cleaned = f"{cleaned[:220].rstrip()}..."
    return cleaned


def ensure_todo_md() -> None:
    TODO_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TODO_PATH.exists():
        return
    TODO_PATH.write_text(
        "# TODO\n\n"
        "Fast review debt carried between iterations.\n\n"
        "- [x] No open review debt right now.\n",
        encoding="utf-8",
    )


def parse_todo_items(raw_text: str) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for raw_line in raw_text.splitlines():
        match = TODO_ITEM_PATTERN.match(raw_line.strip())
        if not match:
            continue
        text = sanitize_todo_text(match.group("text"))
        if not text or text.lower() == "no open review debt right now.":
            continue
        files_raw = clean_history_text(match.group("files") or "", fallback="")
        files = [part.strip() for part in files_raw.split(",") if part.strip()]
        items.append(
            {
                "done": match.group("done").lower() == "x",
                "text": text,
                "files": files,
            }
        )
    return items


def load_todo_items() -> List[Dict[str, object]]:
    ensure_todo_md()
    try:
        raw = TODO_PATH.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    return parse_todo_items(raw)


def format_todo_item(item: Dict[str, object]) -> str:
    mark = "x" if bool(item.get("done")) else " "
    text = sanitize_todo_text(str(item.get("text", "")))
    files = [
        clean_history_text(str(value), fallback="")
        for value in item.get("files", [])
        if clean_history_text(str(value), fallback="")
    ]
    files_suffix = f" (files: {', '.join(files[:4])})" if files else ""
    return f"- [{mark}] {text}{files_suffix}".rstrip()


def write_todo_items(items: Sequence[Dict[str, object]], summary: str = "") -> None:
    ensure_todo_md()
    unresolved = [dict(item) for item in items if not bool(item.get("done")) and sanitize_todo_text(str(item.get("text", "")))]
    resolved = [dict(item) for item in items if bool(item.get("done")) and sanitize_todo_text(str(item.get("text", "")))]
    lines = [
        "# TODO",
        "",
        "Fast review debt carried between iterations.",
    ]
    summary = clean_history_text(summary, fallback="")
    if summary:
        lines.extend(["", f"Summary: {summary}"])
    lines.append("")
    if unresolved or resolved:
        for item in unresolved + resolved[:4]:
            lines.append(format_todo_item(item))
    else:
        lines.append("- [x] No open review debt right now.")
    lines.append("")
    TODO_PATH.write_text("\n".join(lines), encoding="utf-8")


def load_open_todo_notes(max_chars: int = 1200) -> str:
    items = [item for item in load_todo_items() if not bool(item.get("done"))]
    if not items:
        return ""
    snippet = "\n".join(format_todo_item(item) for item in items[:8])
    return snippet if len(snippet) <= max_chars else f"{snippet[:max_chars].rstrip()}..."


def sanitize_reflection_text(value: str) -> str:
    cleaned = str(value or "")
    replacements = {
        "●": "",
        "•": "",
        "â—": "",
        "â€“": "-",
        "â€”": "-",
        "â€™": "'",
        "â€œ": '"',
        "â€\x9d": '"',
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = cleaned.encode("ascii", errors="ignore").decode("ascii", errors="ignore")
    cleaned = cleaned.lstrip(" -*:\t\r\n")
    for marker in ("This attempt", "This try", "The attempt"):
        index = cleaned.find(marker)
        if index >= 0:
            cleaned = cleaned[index:]
            break
    cleaned = clean_history_text(cleaned, fallback="")
    if cleaned.startswith("This try"):
        cleaned = "This attempt" + cleaned[len("This try"):]
    if cleaned.startswith("The attempt"):
        cleaned = "This attempt" + cleaned[len("The attempt"):]
    return cleaned


def load_app_context_docs(max_chars: int = 10_000) -> str:
    blocks: List[str] = []
    for filename in APP_CONTEXT_DOCS:
        path = ROOT / filename
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not text:
            continue
        blocks.append(f"Product context from {filename}:\n{text}")
    if not blocks:
        return ""
    joined = "\n\n".join(blocks)
    return joined if len(joined) <= max_chars else f"{joined[:max_chars].rstrip()}\n...[truncated]"


def reflection_dimension_lines(report: Optional[Dict[str, Any]], limit: int = 3) -> List[str]:
    if not report:
        return []
    assessment = report.get("assessment")
    if not isinstance(assessment, dict):
        return []
    rows = assessment.get("dimension_results")
    if not isinstance(rows, list):
        return []
    ranked = sorted(
        [row for row in rows if isinstance(row, dict)],
        key=lambda row: abs(float(row.get("signal_contribution", 0.0))) or float(row.get("effective_contribution", 0.0)),
        reverse=True,
    )
    lines: List[str] = []
    for row in ranked[:limit]:
        dimension_id = clean_history_text(row.get("id", ""), fallback="dimension")
        delta = float(row.get("delta", 0.0))
        reason = clean_history_text(row.get("reason", ""), fallback="")
        lines.append(f"{dimension_id}: delta {delta:+.3f}; {reason}")
    return lines


def build_reflection_fallback(
    *,
    status: str,
    description: str,
    baseline_score: str,
    candidate_score: str,
    error: str,
    changed_files: Sequence[str],
    candidate_report: Optional[Dict[str, Any]] = None,
) -> str:
    if status == "crash":
        if error:
            return (
                f"This attempt was a bad miss: it crashed before it could beat the baseline {baseline_score}. "
                "Next time, try a bolder but cleaner swing like a major workflow simplification, a faster-feeling workspace, or a sharper visual hierarchy."
            )
        return (
            f"This attempt was a bad miss: it crashed before it could beat the baseline {baseline_score}. "
            "Next time, try a bolder but cleaner swing like a stronger information layout, faster interaction flow, or a more radical triage model."
        )

    if candidate_report:
        reasons = reflection_dimension_lines(candidate_report, limit=1)
        if reasons:
            return (
                f"This attempt fell short, scoring {candidate_score} against a baseline of {baseline_score}, so it came off as too weak or too familiar. "
                "Next time, try two or three bigger directions: a much cleaner primary workflow, a more dramatic UI simplification, or a stronger brand/about reframing tied to real code changes."
            )

    changed_summary = ", ".join(changed_files[:3]) if changed_files else "the changed files"
    return (
        f"This attempt changed {changed_summary}, but it still scored {candidate_score} against a baseline of {baseline_score}, which was not good enough. "
        "Next time, shoot for a clearer leap with a stronger workflow move, UI rethink, or performance win."
    )


def build_reflection_prompt(
    *,
    iteration: int,
    status: str,
    scope: str,
    description: str,
    baseline_score: str,
    candidate_score: str,
    error: str,
    changed_files: Sequence[str],
    agent_decisions: Sequence[str],
    candidate_report: Optional[Dict[str, Any]] = None,
) -> str:
    changed_summary = ", ".join(changed_files[:8]) if changed_files else "none"
    decisions = "\n".join(f"- {item}" for item in agent_decisions[:5]) or "- none"
    dimension_notes = "\n".join(f"- {item}" for item in reflection_dimension_lines(candidate_report, limit=4)) or "- none"
    return (
        "You are writing a short human-readable reflection for an autonomous app-improvement loop.\n"
        "The harness will store your note in REFLECTION.md.\n"
        "Write one sentence, or at most two very short sentences, with no bullets and no markdown fences.\n"
        "Keep the full note under 45 words.\n"
        "Write for a tired human reader skimming a run log, not for an engineer reading a postmortem.\n"
        "Do not use jargon, engineering slang, or dense technical wording unless absolutely necessary.\n"
        "Start the first sentence with 'This attempt...'.\n"
        "Sound disappointed in the failure and plainly admit that the attempt did not get it done.\n"
        "The first sentence should say why it failed or scored badly, using the baseline score and candidate score in plain language.\n"
        "If you use a second sentence, give two or three creative or grand next directions worth trying, while staying relevant to the app.\n"
        "Do not write in abstractions like 'iterate more' or 'improve quality'.\n\n"
        f"Iteration: {iteration}\n"
        f"Status: {status}\n"
        f"Scope: {scope}\n"
        f"Attempt description: {description}\n"
        f"Baseline score: {baseline_score}\n"
        f"Candidate score: {candidate_score}\n"
        f"Error detail: {error or 'none'}\n"
        f"Changed files: {changed_summary}\n"
        f"Agent decisions:\n{decisions}\n"
        f"Top evaluator notes:\n{dimension_notes}\n"
    )


def append_reflection(iteration: int, status: str, scope: str, description: str, reflection_text: str) -> None:
    REFLECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not REFLECTION_PATH.exists():
        REFLECTION_PATH.write_text("# Reflection\n\nShort notes from the latest failed attempts.\n", encoding="utf-8")
    with REFLECTION_PATH.open("a", encoding="utf-8", newline="") as handle:
        handle.write(
            f"\n\n## Iteration {iteration} - {status.upper()}\n"
            f"Attempt: {clean_history_text(description, fallback='candidate edit')}\n"
            f"Reflection: {sanitize_reflection_text(reflection_text)}\n"
        )


def write_failure_reflection(
    *,
    iteration: int,
    status: str,
    scope: str,
    description: str,
    baseline_score: str,
    candidate_score: str,
    error: str,
    changed_files: Sequence[str],
    agent_decisions: Sequence[str],
    tool: str,
    model: Optional[str],
    candidate_report: Optional[Dict[str, Any]] = None,
) -> str:
    prompt = build_reflection_prompt(
        iteration=iteration,
        status=status,
        scope=scope,
        description=description,
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        error=error,
        changed_files=changed_files,
        agent_decisions=agent_decisions,
        candidate_report=candidate_report,
    )
    try:
        raw_message, _ = run_agent_prompt(
            prompt,
            cwd=ROOT,
            output_path=LAST_REFLECTION_MESSAGE,
            model=model,
            tool=tool,
            timeout_seconds=REFLECTION_TIMEOUT_SECONDS,
            available_tools=("rg", "view", "glob"),
            silent=True,
            stream_mode="off",
            reasoning_effort="low",
        )
        reflection_text = sanitize_reflection_text(raw_message)
    except Exception:
        reflection_text = ""
    if not reflection_text:
        reflection_text = build_reflection_fallback(
            status=status,
            description=description,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            error=error,
            changed_files=changed_files,
            candidate_report=candidate_report,
        )
    append_reflection(iteration, status, scope, description, reflection_text)
    return reflection_text


def build_prompt(
    baseline_report: dict,
    retry_feedback: str = "",
    *,
    attempt_number: int = 1,
    strategy_arm: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    score = baseline_report["assessment"]["score"]
    build_commands = [item["command"] for item in baseline_report["build_results"]]
    corrective_message = latest_corrective_message()
    corrective_block = f"\n\n{corrective_message}" if corrective_message else ""
    app_context_block = load_app_context_docs()
    benchmark_hint = load_benchmark_hint()
    reflection_notes = load_latest_reflection_note()
    todo_notes = load_open_todo_notes()
    retry_feedback = clean_history_text(retry_feedback, fallback="")
    strategy_block = build_strategy_block(
        strategy_arm,
        attempt_number=attempt_number,
        config=config or load_runner_config(),
    )
    benchmark_hint_block = (
        "\nRecent benchmark hint excerpt from .autoresearch-cache/last-benchmark-message.txt:\n"
        f"{benchmark_hint}\n"
        if benchmark_hint
        else ""
    )
    reflection_block = (
        "\nLatest reflection memory from REFLECTION.md:\n"
        f"{reflection_notes}\n"
        if reflection_notes
        else ""
    )
    todo_block = (
        "\nOpen review debt from TODO.md:\n"
        f"{todo_notes}\n"
        if todo_notes
        else ""
    )
    retry_block = (
        "\nSame-iteration retry feedback:\n"
        f"{retry_feedback}\n"
        if retry_feedback
        else ""
    )
    benchmark_payload = baseline_report.get("benchmark_payload", {}) or {}
    focus_paths = ", ".join(benchmark_payload.get("paths_of_interest", [])[:6]) or "src/, tests/"
    dimension_descriptions = [
        clean_history_text(item.get("description", ""))
        for item in benchmark_payload.get("dimensions", [])
        if clean_history_text(item.get("description", ""))
    ]
    task_descriptions = [
        clean_history_text(task.get("task_description", ""))
        for task in benchmark_payload.get("tasks", [])
        if clean_history_text(task.get("task_description", ""))
    ]
    using_dimensions = bool(dimension_descriptions)
    benchmark_lens_label = "Frozen benchmark dimensions" if using_dimensions else "Primary benchmark task(s)"
    benchmark_lens_summary = (
        "; ".join(dimension_descriptions[:4]) or "Improve the repo against the frozen benchmark dimensions."
        if using_dimensions
        else "; ".join(task_descriptions[:3]) or "Complete the app's primary user-visible workflow."
    )
    criteria_descriptions = [
        clean_history_text(item.get("description", ""))
        for item in benchmark_payload.get("mandatory_criteria", [])
        if clean_history_text(item.get("description", ""))
    ]
    criteria_summary = "; ".join(criteria_descriptions[:3]) or "Keep the app intact and passing tests."
    score_lines = [f"Current benchmark score_pct: {score['score_pct']:.6f}"]
    if "dimension_signal" in score:
        score_lines.append(f"Current benchmark signal: {score['dimension_signal']:.6f}")
        score_lines.append(f"Current effective benchmark coverage: {score['effective_weight']:.6f}")
        score_lines.append(f"Current observable benchmark coverage: {score['observable_weight']:.6f}")
    else:
        score_lines.append(f"Current weighted_delta_seconds: {score['weighted_delta_seconds']:.6f}")
        score_lines.append(f"Current weighted_feature_seconds: {score['weighted_feature_seconds']:.6f}")
    score_block = "\n".join(score_lines)
    return (
        f"{read_program()}\n\n"
        "Execution policy for this iteration:\n"
        "1. Read BRAND.md, ABOUT.md, and FEATURES.md first if they exist for quick context.\n"
        "2. Read the last few lines of results.tsv so you do not repeat a failed idea.\n"
        "3. Read the latest reflection memory if it exists.\n"
        "4. Read TODO.md if it has open review debt, and fix that debt before chasing novelty unless one stronger move can honestly do both.\n"
        "5. Pick a direction fast and beat the score.\n"
        "5a. If a strategy arm is present, use it as the main attack angle for this attempt.\n"
        "6. Treat results.tsv as read-only harness memory. Do not edit it.\n"
        "7. If .autoresearch-cache/last-benchmark-message.txt exists, use it as the next clue for where to act.\n"
        "8. Start with the app files most implicated by the brief, recent results, latest reflection, open TODO debt, or the benchmark hint.\n"
        "9. Do not run shell commands or tests yourself; the harness does verification after your patch.\n\n"
        f"{app_context_block}\n\n"
        f"{benchmark_hint_block}"
        f"{reflection_block}"
        f"{todo_block}"
        f"{strategy_block}"
        f"{retry_block}"
        f"{format_experiment_memory()}\n\n"
        f"{score_block}\n"
        f"Current gate g: {baseline_report['measurement']['g']}\n"
        f"Frozen benchmark source for this run: {baseline_report['benchmark_source']}\n"
        f"Benchmark focus paths: {focus_paths}\n"
        f"{benchmark_lens_label}: {benchmark_lens_summary}\n"
        f"Mandatory criteria: {criteria_summary}\n"
        f"Build/test commands rerun automatically: {build_commands}\n"
        "Work inside this repository. A hidden frozen benchmark will be re-evaluated after your patch. "
        "Use BRAND.md, ABOUT.md, and FEATURES.md for quick context. "
        "Read results.tsv if it exists and use it as persistent experiment memory across fresh agent processes. Do not edit it. "
        "Read REFLECTION.md if it exists and use only the latest reflection as quick memory about why the last failure failed. "
        "Read TODO.md if it exists and treat unchecked items as real review debt from earlier kept work. Fix that debt before novelty, or fold the repair into one stronger move. "
        "You may mark an existing TODO.md item done only when the code truly fixes it; a fast review pass will reopen anything that is bluffing, incomplete, or only cosmetic. "
        "If same-iteration retry feedback is present, use it and take a materially stronger second swing instead of repeating the same move. "
        "If .autoresearch-cache/last-benchmark-message.txt exists, treat it as the highest-signal benchmark hint and use it early instead of rediscovering the whole repo. "
        "Do not repeat discarded or crashed experiments unless you are clearly addressing why they failed. "
        "The goal is simple: get the highest score possible or at the very least beat the previous score. "
        "The score now comes from a fixed 20-parameter application scorecard, so strong focused wins are valid and coherent multi-area wins can also score through holistic improvement. "
        "Do not spray lots of tiny unrelated tweaks across the app hoping to collect points. "
        "Workflow, UI, interaction clarity, reliability, state continuity, data quality, performance, feature value, brand/about evolution, and AI usefulness are all valid ways to improve the score. "
        "If the app already has optional AI or model runtime hooks, using them is allowed. Treat those hooks as a gateway into AI-native or agentic product ideas when they can improve the score. "
        f"The entire iteration is capped at {ITERATION_HARD_TIMEOUT_SECONDS} seconds, including scoring. "
        f"The harness will leave about {ITERATION_EVALUATION_RESERVE_SECONDS} seconds at the end for candidate evaluation, and the rest of the remaining iteration budget is yours to allocate. "
        "You decide how long to reason, how long to explore, and how long to edit. "
        "Do not spend the whole iteration on unfocused list/read/search exploration, but you do not need to edit immediately if a stronger understanding will produce a stronger patch. "
        "Do not use shell commands or try to run tests yourself; the harness reruns verification after your patch. "
        "Use repository reads/searches plus apply_patch. Aim for depth and leverage, not just quick visible motion. "
        "You may change BRAND.md, ABOUT.md, and FEATURES.md if doing so helps the score. "
        "You may update TODO.md only to check off debt you actually repaired. Do not invent new checklist items there. "
        "Do not create planning files outside the repo. "
        "If the repo is ambiguous, pick a move fast and swing. "
        "Exit once your attempt is done."
        f"{corrective_block} "
        "Do not edit support_docs/, support_scripts/, karpathy-files/, benchmark.json if present, "
        "program.md, prepare.py, train.py, results.tsv, REFLECTION.md, or anything under .autoresearch-cache/. TODO.md is the one exception and only for honestly checking off repaired debt. "
        "NEVER STOP IMPROVING. AIM FOR GREATNESS. AIM FOR EXTRAORDINARY. "
        "One fresh agent process is created per iteration, so finish the patch and exit cleanly. "
        "When you finish, reply with one tab-separated line: <path-or-scope>\\t<short description>."
    )


def compute_iteration_timeout(requested_seconds: int, *, deadline_monotonic: float, label: str) -> int:
    remaining = deadline_monotonic - time.monotonic()
    if remaining <= 0:
        raise RuntimeError(
            f"Iteration exceeded the hard timeout of {ITERATION_HARD_TIMEOUT_SECONDS} seconds before {label}."
        )
    return max(1, min(int(requested_seconds), int(math.ceil(remaining))))


def compute_agent_timeout(*, deadline_monotonic: float) -> int:
    remaining = deadline_monotonic - time.monotonic()
    if remaining <= 0:
        raise RuntimeError(
            f"Iteration exceeded the hard timeout of {ITERATION_HARD_TIMEOUT_SECONDS} seconds before agent execution."
        )
    if remaining <= ITERATION_EVALUATION_RESERVE_SECONDS:
        return 1
    return max(1, int(math.floor(remaining - ITERATION_EVALUATION_RESERVE_SECONDS)))


def can_retry_within_iteration(*, attempt_number: int, deadline_monotonic: float) -> bool:
    if attempt_number >= MAX_AGENT_PASSES_PER_ITERATION:
        return False
    remaining = deadline_monotonic - time.monotonic()
    return remaining > (ITERATION_EVALUATION_RESERVE_SECONDS + MIN_RETRY_WINDOW_SECONDS)


def build_same_iteration_retry_feedback(
    *,
    status: str,
    baseline_score: float,
    description: str,
    scope: str,
    changed_files: Sequence[str],
    error: str = "",
    candidate_score: Optional[float] = None,
    candidate_report: Optional[Dict[str, Any]] = None,
) -> str:
    changed_summary = ", ".join(changed_files[:4]) if changed_files else "none"
    if status == "crash":
        detail = clean_history_text(error, fallback="verification failed")
        return (
            f"Last attempt in this same iteration crashed. Scope: {clean_history_text(scope, fallback='candidate')}. "
            f"Description: {clean_history_text(description, fallback='candidate edit')}. "
            f"Files: {changed_summary}. Failure: {detail[:220]}. "
            "Do not repeat that failure mode. Take a different or cleaner swing."
        )
    if status == "discard" and candidate_score is not None:
        dimension_notes = " | ".join(reflection_dimension_lines(candidate_report, limit=2))
        feedback = (
            f"Last attempt in this same iteration lost. Scope: {clean_history_text(scope, fallback='candidate')}. "
            f"Description: {clean_history_text(description, fallback='candidate edit')}. "
            f"It scored {candidate_score:.6f} against baseline {baseline_score:.6f}. "
            f"Files: {changed_summary}. "
        )
        if dimension_notes:
            feedback += f"Likely why: {dimension_notes}. "
        feedback += "Do not repeat that move. Pick a materially stronger direction."
        return feedback
    return (
        f"Last attempt in this same iteration produced no useful candidate. Scope: {clean_history_text(scope, fallback='candidate')}. "
        f"Description: {clean_history_text(description, fallback='candidate edit')}. Files: {changed_summary}. "
        "Do not repeat that path. Pick a more decisive move."
    )


def summarize_same_iteration_retry(
    *,
    attempt_number: int,
    status: str,
    description: str,
    baseline_score: float,
    candidate_score: Optional[float] = None,
    error: str = "",
) -> str:
    if status == "crash":
        detail = clean_history_text(error, fallback="verification failed")
        return f"attempt {attempt_number}: crash | {clean_history_text(description, fallback='candidate edit')} | {detail[:140]}"
    if candidate_score is None:
        return f"attempt {attempt_number}: discard | {clean_history_text(description, fallback='candidate edit')} | no valid candidate"
    return (
        f"attempt {attempt_number}: discard | {clean_history_text(description, fallback='candidate edit')} | "
        f"{candidate_score:.6f} vs {baseline_score:.6f}"
    )


def run_agent(
    prompt: str,
    tool: str,
    model: Optional[str],
    *,
    timeout_seconds: int,
    reasoning_effort: str = "xhigh",
) -> Dict[str, object]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _, usage = run_agent_prompt(
        prompt,
        cwd=ROOT,
        output_path=LAST_AGENT_MESSAGE,
        model=model,
        tool=tool,
        timeout_seconds=timeout_seconds,
        reasoning_effort=reasoning_effort,
    )
    return usage


def build_todo_review_prompt(
    *,
    iteration: int,
    scope: str,
    description: str,
    commit_id: str,
    changed_files: Sequence[str],
    previous_items: Sequence[Dict[str, object]],
) -> str:
    changed_summary = ", ".join(changed_files[:8]) if changed_files else "none"
    prior_payload = json.dumps(list(previous_items)[:8], indent=2)
    return (
        "You are a fast post-iteration quality reviewer for an autonomous app-improvement loop.\n"
        "Inspect the current codebase and decide whether the shipped work is complete, real, and worth trusting.\n"
        "Look for concrete gaps such as placeholder UI, labels without behavior, features described more strongly than the code supports, incomplete flows, and work that feels half-finished.\n"
        "Carry forward old TODO debt unless the current code clearly fixed it.\n"
        "Use plain human language. Keep items concrete, short, and actionable.\n"
        "Return JSON only. No prose outside JSON.\n"
        "Use this exact shape:\n"
        "{\n"
        '  "summary": "short plain-language summary",\n'
        '  "items": [\n'
        '    {"text": "short concrete debt item", "done": false, "files": ["path/one", "path/two"]}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Keep at most 6 items total.\n"
        "- Mark done true only if the current code clearly fixed that debt.\n"
        "- Prefer 0 items over filler.\n"
        "- Do not complain about style unless it affects completeness, trust, or quality.\n"
        "- Focus extra hard on whether the change is real behavior or just boxes, labels, and claims.\n\n"
        f"Iteration: {iteration}\n"
        f"Commit: {commit_id}\n"
        f"Scope: {scope}\n"
        f"Claimed change: {description}\n"
        f"Changed files: {changed_summary}\n"
        f"Existing TODO debt:\n{prior_payload}\n"
    )


def review_kept_candidate(
    *,
    iteration: int,
    scope: str,
    description: str,
    commit_id: str,
    changed_files: Sequence[str],
    tool: str,
    model: Optional[str],
    config: Dict[str, Any],
) -> Tuple[List[str], Optional[Dict[str, object]]]:
    ensure_todo_md()
    previous_items = load_todo_items()
    if not config.get("review_enabled", DEFAULT_REVIEW_ENABLED):
        return [], None
    review_model = clean_history_text(config.get("review_model", ""), fallback="") or model
    reasoning_effort = clean_history_text(
        config.get("review_reasoning_effort", DEFAULT_REVIEW_REASONING_EFFORT),
        fallback=DEFAULT_REVIEW_REASONING_EFFORT,
    ).lower()
    timeout_seconds = int(config.get("review_timeout_seconds", TODO_REVIEW_TIMEOUT_SECONDS))
    try:
        raw_message, usage = run_agent_prompt(
            build_todo_review_prompt(
                iteration=iteration,
                scope=scope,
                description=description,
                commit_id=commit_id,
                changed_files=changed_files,
                previous_items=previous_items,
            ),
            cwd=ROOT,
            output_path=LAST_TODO_REVIEW_MESSAGE,
            model=review_model,
            tool=tool,
            timeout_seconds=timeout_seconds,
            available_tools=("rg", "view", "glob"),
            silent=True,
            stream_mode="off",
            reasoning_effort=reasoning_effort,
        )
        payload = extract_json_object(raw_message)
        raw_items = payload.get("items", [])
        reviewed_items: List[Dict[str, object]] = []
        if isinstance(raw_items, list):
            for item in raw_items[:6]:
                if not isinstance(item, dict):
                    continue
                text = sanitize_todo_text(str(item.get("text", "")))
                if not text:
                    continue
                files = (
                    [
                        clean_history_text(str(value), fallback="")
                        for value in item.get("files", [])
                        if clean_history_text(str(value), fallback="")
                    ]
                    if isinstance(item.get("files"), list)
                    else []
                )
                reviewed_items.append(
                    {
                        "done": bool(item.get("done")),
                        "text": text,
                        "files": files[:4],
                    }
                )
        summary = clean_history_text(payload.get("summary", ""), fallback="")
        write_todo_items(reviewed_items, summary=summary)
        open_items = [format_todo_item(item) for item in reviewed_items if not bool(item.get("done"))]
        if open_items:
            return [f"review debt recorded: {len(open_items)} open item(s)"] + open_items, usage
        return [summary or "review says the kept work looks complete enough to trust."], usage
    except Exception as exc:
        write_todo_items(previous_items, summary="Review pass failed; carrying forward prior debt unchanged.")
        return [f"review pass failed; kept prior TODO debt unchanged ({clean_history_text(str(exc), fallback='review parse failure')})"], None


def enable_ansi_colors() -> bool:
    if os.name != "nt":
        return sys.stdout.isatty()
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        if handle == 0:
            return False
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return False
        if kernel32.SetConsoleMode(handle, mode.value | 0x0004) == 0:
            return False
        return True
    except Exception:
        return False


ANSI_ENABLED = enable_ansi_colors()
ANSI_COLORS = {
    "cyan": "\033[96m",
    "blue": "\033[94m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "magenta": "\033[95m",
    "dim": "\033[90m",
    "reset": "\033[0m",
}


def safe_console_text(value: object) -> str:
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def colorize(text: str, tone: Optional[str] = None) -> str:
    safe_text = safe_console_text(text)
    if not tone or not ANSI_ENABLED:
        return safe_text
    prefix = ANSI_COLORS.get(tone, "")
    suffix = ANSI_COLORS["reset"] if prefix else ""
    return f"{prefix}{safe_text}{suffix}"


def console_print(text: str = "", *, tone: Optional[str] = None) -> None:
    with PRINT_LOCK:
        print(colorize(text, tone))


def classify_file_changes(entries: Sequence[StatusEntry]) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {
        "created": [],
        "modified": [],
        "deleted": [],
        "renamed": [],
    }
    for entry in entries:
        status = entry.status
        path_text = entry.path.replace("\\", "/")
        if status == "??" or "A" in status:
            buckets["created"].append(path_text)
            continue
        if "R" in status:
            if entry.original_path:
                original = entry.original_path.replace("\\", "/")
                buckets["renamed"].append(f"{original} -> {path_text}")
            else:
                buckets["renamed"].append(path_text)
            continue
        if "D" in status:
            buckets["deleted"].append(path_text)
            continue
        buckets["modified"].append(path_text)

    for key in buckets:
        buckets[key] = sorted(dict.fromkeys(buckets[key]))
    return buckets


def cpu_activity_text() -> str:
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "(Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples[0].CookedValue",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=12,
        )
        if completed.returncode != 0:
            return "n/a"
        raw = (completed.stdout or "").strip().splitlines()
        if not raw:
            return "n/a"
        return f"{float(raw[-1]):.1f}%"
    except Exception:
        return "n/a"


class IterationHeartbeat:
    def __init__(self, *, iteration: int, total_iterations: Optional[int], start_monotonic: float) -> None:
        self.iteration = iteration
        self.total_iterations = total_iterations
        self.start_monotonic = start_monotonic
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.stage = "booting up"
        self.thread = threading.Thread(target=self._run, name=f"autoresearch-heartbeat-{iteration}", daemon=True)

    def set_stage(self, stage: str) -> None:
        with self.state_lock:
            self.stage = stage

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)

    def _current_stage(self) -> str:
        with self.state_lock:
            return self.stage

    def _run(self) -> None:
        while not self.stop_event.wait(HEARTBEAT_SECONDS):
            entries = parse_status_entries(cwd=ROOT)
            buckets = classify_file_changes(entries)
            total_label = "until stopped" if self.total_iterations is None else str(self.total_iterations)
            elapsed = time.monotonic() - self.start_monotonic
            heartbeat_line = (
                f"[heartbeat] iter {self.iteration}/{total_label} | stage: {self._current_stage()} | "
                f"elapsed: {elapsed:.0f}s | cpu: {cpu_activity_text()} | "
                f"created: {len(buckets['created'])} | modified: {len(buckets['modified'])} | "
                f"renamed: {len(buckets['renamed'])} | deleted: {len(buckets['deleted'])}"
            )
            console_print(heartbeat_line, tone="yellow")
            active_files = buckets["created"] + buckets["modified"] + buckets["renamed"]
            if active_files:
                preview = ", ".join(active_files[:6])
                if len(active_files) > 6:
                    preview = f"{preview}, +{len(active_files) - 6} more"
                console_print(f"[files] {preview}", tone="blue")
            console_print(f'[signal] "{random.choice(RETRO_PULP_QUOTES)}"', tone="magenta")


def normalize_agent_line(line: str) -> str:
    return line.strip().lstrip("-*•● ").strip()


def is_agent_noise_line(line: str) -> bool:
    lowered = clean_history_text(line).lower()
    return (
        not lowered
        or lowered.startswith("disabled tools:")
        or lowered.startswith("list_powershell")
        or lowered.startswith("stop_powershell")
        or lowered.startswith("read ")
        or lowered.startswith("list ")
        or lowered.startswith("search ")
        or lowered.startswith("? ")
        or lowered.startswith("└ ")
    )


def parse_agent_message() -> Tuple[str, str]:
    if not LAST_AGENT_MESSAGE.exists():
        return "candidate", "candidate edit"
    lines = LAST_AGENT_MESSAGE.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return "candidate", "candidate edit"
    for raw_line in reversed(lines):
        line = normalize_agent_line(raw_line)
        if "\t" in line:
            scope, description = line.split("\t", 1)
            scope = scope.strip() or "candidate"
            description = description.strip() or f"candidate edit for {scope}"
            return scope, description
    for raw_line in reversed(lines):
        line = normalize_agent_line(raw_line)
        if is_agent_noise_line(line):
            continue
        return "candidate", line or "candidate edit"
    last_line = normalize_agent_line(lines[-1])
    return "candidate", last_line or "candidate edit"


def parse_agent_decisions() -> List[str]:
    if not LAST_AGENT_MESSAGE.exists():
        return []
    decisions: List[str] = []
    for raw_line in LAST_AGENT_MESSAGE.read_text(encoding="utf-8").splitlines():
        line = normalize_agent_line(raw_line)
        if not line or "\t" in line or is_agent_noise_line(line):
            continue
        decisions.append(line)
        if len(decisions) >= 5:
            break
    return decisions


def is_discovery_only_attempt(agent_decisions: Sequence[str]) -> bool:
    if not agent_decisions:
        return False
    discovery_prefixes = (
        "list ",
        "read ",
        "search ",
        "find ",
        "glob",
        "? ",
        "i'm getting a quick picture",
        "i’m getting a quick picture",
        "i’m taking a quick pass",
        "i'm taking a quick pass",
    )
    normalized = [clean_history_text(item).lower() for item in agent_decisions if clean_history_text(item)]
    if not normalized:
        return False
    return all(any(line.startswith(prefix) for prefix in discovery_prefixes) for line in normalized)


def commit_candidate(changed: Sequence[StatusEntry], scope: str, description: str) -> str:
    paths = sorted({path_text for entry in changed for path_text in entry_paths(entry)})
    run_command(["git", "add", "-A", "--", *paths], cwd=ROOT)
    summary = description.replace("\n", " ").strip()
    scope_summary = scope.replace("\n", " ").strip() or "candidate"
    message = f"autoresearch: {scope_summary}"
    if summary:
        message = f"{message} - {summary[:56]}"
    run_command(["git", "commit", "-m", message], cwd=ROOT)
    return run_command(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).stdout.strip()


def ensure_results_tsv() -> None:
    if RESULTS_PATH.exists():
        return
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "iteration",
                "status",
                "fatal",
                "baseline_score_pct",
                "candidate_score_pct",
                "score_delta",
                "candidate_signal",
                "gate",
                "scope",
                "description",
                "outcome",
                "commit",
                "elapsed_seconds",
                "backend",
                "tool",
                "cumulative_keep",
                "cumulative_discard",
                "cumulative_crash",
                "error",
            ]
        )


def append_result(
    iteration: int,
    commit_id: str,
    baseline_score_pct: float,
    candidate_score_pct: float,
    score_delta: float,
    candidate_signal: float,
    status: str,
    scope: str,
    files: Sequence[str],
    description: str,
) -> None:
    ensure_results_tsv()
    ensure_todo_md()
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                iteration,
                status,
                "true" if status == "crash" and scope == "baseline" else "false",
                f"{baseline_score_pct:.6f}",
                f"{candidate_score_pct:.6f}",
                f"{score_delta:.6f}",
                f"{candidate_signal:.6f}",
                "",
                scope,
                description,
                "",
                commit_id,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )


def print_iteration_report(
    *,
    iteration: int,
    total_iterations: Optional[int],
    status: str,
    baseline_score: str,
    candidate_score: str,
    score_delta: str,
    scope: str,
    description: str,
    commit_id: str,
    elapsed_seconds: str,
    file_buckets: Optional[Dict[str, List[str]]] = None,
) -> None:
    total_label = "until stopped" if total_iterations is None else str(total_iterations)
    tone = "green" if status == "keep" else "red" if status == "crash" else "yellow"
    file_buckets = file_buckets or {"created": [], "modified": [], "renamed": [], "deleted": []}
    console_print("---------------------------------------------------------------", tone=tone)
    console_print(f"[iteration report] {iteration}/{total_label} | status: {status.upper()}", tone=tone)
    console_print(
        f"score baseline={baseline_score} candidate={candidate_score} delta={score_delta} elapsed={elapsed_seconds}s",
        tone=tone,
    )
    console_print(f"scope: {scope}", tone="blue")
    console_print(f"description: {description}", tone="blue")
    if commit_id != DUMMY_COMMIT:
        console_print(f"commit: {commit_id}", tone="green")
    if file_buckets["created"]:
        console_print(f"created: {', '.join(file_buckets['created'])}", tone="cyan")
    if file_buckets["modified"]:
        console_print(f"modified: {', '.join(file_buckets['modified'])}", tone="cyan")
    if file_buckets["renamed"]:
        console_print(f"renamed: {', '.join(file_buckets['renamed'])}", tone="cyan")
    if file_buckets["deleted"]:
        console_print(f"deleted: {', '.join(file_buckets['deleted'])}", tone="cyan")
    console_print("---------------------------------------------------------------", tone=tone)


def print_file_sections(file_buckets: Dict[str, List[str]]) -> None:
    print_section("created files", file_buckets["created"])
    print_section("modified files", file_buckets["modified"])
    if file_buckets["renamed"]:
        print_section("renamed files", file_buckets["renamed"])
    if file_buckets["deleted"]:
        print_section("deleted files", file_buckets["deleted"])


def print_summary(payload: Dict[str, object]) -> None:
    console_print("---", tone="dim")
    for key, value in payload.items():
        console_print(f"{key}: {safe_console_text(value)}", tone="dim")


def print_section(title: str, lines: Sequence[str]) -> None:
    console_print(f"[{title}]", tone="cyan")
    if not lines:
        console_print("- none", tone="dim")
        return
    for line in lines:
        console_print(f"- {safe_console_text(line)}", tone="cyan")


def combine_usage(*blocks: Optional[Dict[str, object]]) -> Dict[str, object]:
    valid_blocks = [block for block in blocks if block]
    return {
        "calls": len(valid_blocks),
        "duration_seconds": round(
            sum(float(block.get("duration_seconds", 0.0)) for block in valid_blocks),
            3,
        ),
        "prompt_bytes": sum(int(block.get("prompt_bytes", 0)) for block in valid_blocks),
        "output_bytes": sum(int(block.get("output_bytes", 0)) for block in valid_blocks),
        "usage_available": all(bool(block.get("usage_available")) for block in valid_blocks) if valid_blocks else False,
        "usage_sources": [str(block.get("usage_source", "unknown")) for block in valid_blocks],
        "notes": [note for block in valid_blocks for note in block.get("usage_notes", [])],
    }


def print_copilot_usage(label: str, usage: Dict[str, object]) -> None:
    console_print(f"[{label}]", tone="blue")
    console_print(f"- calls: {usage['calls']}", tone="blue")
    console_print(f"- duration_seconds: {usage['duration_seconds']}", tone="blue")
    console_print(f"- prompt_bytes: {usage['prompt_bytes']}", tone="blue")
    console_print(f"- output_bytes: {usage['output_bytes']}", tone="blue")
    console_print(f"- usage_available: {usage['usage_available']}", tone="blue")
    usage_sources = ', '.join(usage['usage_sources']) if usage['usage_sources'] else 'none'
    console_print(f"- usage_sources: {safe_console_text(usage_sources)}", tone="blue")
    unique_notes: List[str] = []
    for note in usage["notes"]:
        if note not in unique_notes:
            unique_notes.append(note)
    if unique_notes:
        for note in unique_notes:
            console_print(f"- note: {safe_console_text(note)}", tone="blue")


def outcome_line(status: str, score_delta: float, commit_id: str) -> str:
    if status == "keep":
        return f"score improved by {score_delta:.6f}; changes committed as {commit_id} and passed to the next iteration"
    if status == "discard":
        return f"score did not improve ({score_delta:.6f}); changes were discarded and the worktree was restored"
    return "iteration crashed before a keep/discard decision was reached"


def score_pct(report: dict) -> float:
    return float(report["assessment"]["score"]["score_pct"])


def assessment_signal(report: dict) -> float:
    score = report["assessment"]["score"]
    if "dimension_signal" in score:
        return float(score["dimension_signal"])
    return float(score["weighted_delta_seconds"])


def format_invalid_scope(entries: Sequence[StatusEntry]) -> str:
    changed = sorted({path_text for entry in entries for path_text in entry_paths(entry)})
    return "invalid edit scope: " + ", ".join(changed)


def iteration_banner(iteration: int, total_iterations: Optional[int]) -> None:
    total_text = "until stopped" if total_iterations is None else str(total_iterations)
    console_print("")
    console_print("===============================================================", tone="cyan")
    console_print(f"  Autoresearch Iteration {iteration} of {total_text}", tone="cyan")
    console_print("===============================================================", tone="cyan")


def run_single_iteration(
    *,
    iteration: int,
    total_iterations: Optional[int],
    backend: str,
    tool: str,
    model: Optional[str],
    cumulative: Dict[str, int],
    baseline_state: Dict[str, object],
    config: Dict[str, Any],
    strategy_stats: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    t0 = time.monotonic()
    deadline_monotonic = t0 + ITERATION_HARD_TIMEOUT_SECONDS
    iteration_banner(iteration, total_iterations)
    heartbeat = IterationHeartbeat(iteration=iteration, total_iterations=total_iterations, start_monotonic=t0)
    heartbeat.start()
    heartbeat.set_stage("baseline evaluation")

    try:
        preflight_note = auto_handle_dirty_tree(config)
        if preflight_note:
            console_print(f"[preflight] {preflight_note}", tone="yellow")
        ensure_clean_tree()
        start_commit = run_command(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).stdout.strip()
        cached_report = baseline_state.get("report")
        cached_commit = str(baseline_state.get("commit", ""))
        if cached_report and cached_commit == start_commit:
            baseline = cached_report
        else:
            baseline = evaluate_worktree(
                backend=backend,
                tool=tool,
                model=model,
                deadline_monotonic=deadline_monotonic,
            )
            baseline_state["commit"] = start_commit
            baseline_state["report"] = baseline
        baseline_score = score_pct(baseline)
    except Exception as exc:
        cumulative["crash"] += 1
        baseline_state.clear()
        try:
            CACHED_BENCHMARK_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        append_result(iteration, DUMMY_COMMIT, 0.0, 0.0, 0.0, 0.0, "crash", "baseline", [], f"baseline failure: {exc}")
        reflection_text = write_failure_reflection(
            iteration=iteration,
            status="crash",
            scope="baseline",
            description=str(exc),
            baseline_score="n/a",
            candidate_score="n/a",
            error=str(exc),
            changed_files=[],
            agent_decisions=[],
            tool=tool,
            model=model,
        )
        summary = {
            "iteration": iteration,
            "status": "crash",
            "fatal": False,
            "scope": "baseline",
            "baseline_score_pct": "n/a",
            "candidate_score_pct": "n/a",
            "commit": DUMMY_COMMIT,
            "error": str(exc),
            "elapsed_seconds": f"{time.monotonic() - t0:.1f}",
        }
        print_iteration_report(
            iteration=iteration,
            total_iterations=total_iterations,
            status="crash",
            baseline_score="n/a",
            candidate_score="n/a",
            score_delta="n/a",
            scope="baseline",
            description=str(exc),
            commit_id=DUMMY_COMMIT,
            elapsed_seconds=f"{time.monotonic() - t0:.1f}",
        )
        print_section("reflection", [reflection_text])
        print_summary(summary)
        heartbeat.stop()
        return summary

    baseline_usage = baseline.get("copilot_usage")
    usage_blocks: List[Optional[Dict[str, object]]] = [baseline_usage]
    retry_feedback = ""
    retry_summaries: List[str] = []
    attempt_number = 1
    tried_strategy_ids: List[str] = []

    while True:
        current_strategy_arm = choose_strategy_arm(config, strategy_stats, tried_ids=tried_strategy_ids)
        tried_strategy_ids.append(str(current_strategy_arm["id"]))
        try:
            heartbeat.set_stage("agent implementation")
            agent_usage = run_agent(
                build_prompt(
                    baseline,
                    retry_feedback=retry_feedback,
                    attempt_number=attempt_number,
                    strategy_arm=current_strategy_arm,
                    config=config,
                ),
                tool,
                model,
                timeout_seconds=compute_agent_timeout(deadline_monotonic=deadline_monotonic),
                reasoning_effort="xhigh",
            )
            usage_blocks.append(agent_usage)
        except RuntimeError as exc:
            if can_retry_within_iteration(attempt_number=attempt_number, deadline_monotonic=deadline_monotonic):
                update_strategy_stats(
                    strategy_stats,
                    arm_id=str(current_strategy_arm["id"]),
                    baseline_score=baseline_score,
                    status="crash",
                )
                save_strategy_stats(strategy_stats)
                retry_summaries.append(
                    summarize_same_iteration_retry(
                        attempt_number=attempt_number,
                        status="crash",
                        description=f"[{current_strategy_arm['id']}] {exc}",
                        baseline_score=baseline_score,
                        error=str(exc),
                    )
                )
                retry_feedback = build_same_iteration_retry_feedback(
                    status="crash",
                    baseline_score=baseline_score,
                    description=str(exc),
                    scope="agent",
                    changed_files=[],
                    error=str(exc),
                )
                console_print(
                    f"[retry] attempt {attempt_number} ({current_strategy_arm['id']}) crashed before scoring; taking another swing inside this iteration",
                    tone="yellow",
                )
                attempt_number += 1
                continue

            update_strategy_stats(
                strategy_stats,
                arm_id=str(current_strategy_arm["id"]),
                baseline_score=baseline_score,
                status="crash",
            )
            save_strategy_stats(strategy_stats)
            cumulative["crash"] += 1
            append_result(iteration, DUMMY_COMMIT, baseline_score, baseline_score, 0.0, 0.0, "crash", "agent", [], str(exc))
            reflection_text = write_failure_reflection(
                iteration=iteration,
                status="crash",
                scope="agent",
                description=str(exc),
                baseline_score=f"{baseline_score:.6f}",
                candidate_score="n/a",
                error=str(exc),
                changed_files=[],
                agent_decisions=[],
                tool=tool,
                model=model,
            )
            summary = {
                "iteration": iteration,
                "status": "crash",
                "fatal": False,
                "baseline_score_pct": f"{baseline_score:.6f}",
                "candidate_score_pct": "n/a",
                "commit": DUMMY_COMMIT,
                "error": str(exc),
                "elapsed_seconds": f"{time.monotonic() - t0:.1f}",
            }
            print_iteration_report(
                iteration=iteration,
                total_iterations=total_iterations,
                status="crash",
                baseline_score=f"{baseline_score:.6f}",
                candidate_score="n/a",
                score_delta="n/a",
                scope="agent",
                description=str(exc),
                commit_id=DUMMY_COMMIT,
                elapsed_seconds=f"{time.monotonic() - t0:.1f}",
            )
            if retry_summaries:
                print_section("same-iteration retries", retry_summaries)
            print_section("reflection", [reflection_text])
            print_copilot_usage("copilot usage", combine_usage(*usage_blocks))
            print_summary(summary)
            heartbeat.stop()
            return summary

        changed_entries = parse_status_entries(cwd=ROOT)
        heartbeat.set_stage("candidate inspection")
        allowed_changed, invalid_changed = classify_candidate_diff(changed_entries)
        changed_files = relative_paths(allowed_changed)
        invalid_files = relative_paths(invalid_changed)
        candidate_file_buckets = classify_file_changes(allowed_changed)
        all_changed_file_buckets = classify_file_changes(changed_entries)
        scope, description = parse_agent_message()
        agent_decisions = parse_agent_decisions()

        if invalid_changed or not allowed_changed:
            description = "no valid app changes produced"
            if invalid_changed:
                description = format_invalid_scope(invalid_changed)
            elif is_discovery_only_attempt(agent_decisions):
                description = (
                    "no valid app changes produced after a discovery-only attempt; form one stronger hypothesis "
                    "and avoid repeated list/read/search loops that never converge to a shipped patch"
                )
            if can_retry_within_iteration(attempt_number=attempt_number, deadline_monotonic=deadline_monotonic):
                restore_entries(start_commit, changed_entries, cwd=ROOT)
                update_strategy_stats(
                    strategy_stats,
                    arm_id=str(current_strategy_arm["id"]),
                    baseline_score=baseline_score,
                    status="discard",
                    score_delta=0.0,
                )
                save_strategy_stats(strategy_stats)
                retry_summaries.append(
                    summarize_same_iteration_retry(
                        attempt_number=attempt_number,
                        status="discard",
                        description=f"[{current_strategy_arm['id']}] {description}",
                        baseline_score=baseline_score,
                    )
                )
                retry_feedback = build_same_iteration_retry_feedback(
                    status="discard",
                    baseline_score=baseline_score,
                    description=description,
                    scope=scope,
                    changed_files=invalid_files,
                )
                console_print(
                    f"[retry] attempt {attempt_number} ({current_strategy_arm['id']}) produced no valid candidate; taking another swing inside this iteration",
                    tone="yellow",
                )
                attempt_number += 1
                continue

            restore_entries(start_commit, changed_entries, cwd=ROOT)
            update_strategy_stats(
                strategy_stats,
                arm_id=str(current_strategy_arm["id"]),
                baseline_score=baseline_score,
                status="discard",
                score_delta=0.0,
            )
            save_strategy_stats(strategy_stats)
            cumulative["discard"] += 1
            append_result(iteration, DUMMY_COMMIT, baseline_score, baseline_score, 0.0, 0.0, "discard", scope, invalid_files, description)
            reflection_text = write_failure_reflection(
                iteration=iteration,
                status="discard",
                scope=scope,
                description=description,
                baseline_score=f"{baseline_score:.6f}",
                candidate_score="n/a",
                error="",
                changed_files=invalid_files,
                agent_decisions=agent_decisions,
                tool=tool,
                model=model,
            )
            print_file_sections(all_changed_file_buckets)
            if invalid_files:
                print_section("invalid files", invalid_files)
            print_section("agent decisions", agent_decisions)
            if retry_summaries:
                print_section("same-iteration retries", retry_summaries)
            print_section("reflection", [reflection_text])
            print_copilot_usage("copilot usage", combine_usage(*usage_blocks))
            summary = {
                "iteration": iteration,
                "status": "discard",
                "fatal": False,
                "baseline_score_pct": f"{baseline_score:.6f}",
                "candidate_score_pct": "n/a",
                "outcome": "no valid candidate patch; changes were discarded and the worktree was restored",
                "commit": DUMMY_COMMIT,
                "elapsed_seconds": f"{time.monotonic() - t0:.1f}",
                "cumulative_keep": cumulative["keep"],
                "cumulative_discard": cumulative["discard"],
                "cumulative_crash": cumulative["crash"],
            }
            print_iteration_report(
                iteration=iteration,
                total_iterations=total_iterations,
                status="discard",
                baseline_score=f"{baseline_score:.6f}",
                candidate_score="n/a",
                score_delta="0.000000",
                scope=scope,
                description=description,
                commit_id=DUMMY_COMMIT,
                elapsed_seconds=f"{time.monotonic() - t0:.1f}",
                file_buckets=all_changed_file_buckets,
            )
            print_summary(summary)
            heartbeat.stop()
            return summary

        try:
            heartbeat.set_stage("candidate evaluation")
            candidate = evaluate_worktree(
                backend=backend,
                tool=tool,
                model=model,
                deadline_monotonic=deadline_monotonic,
                require_nonzero_delta=True,
            )
        except Exception as exc:
            restore_entries(start_commit, allowed_changed, cwd=ROOT)
            if can_retry_within_iteration(attempt_number=attempt_number, deadline_monotonic=deadline_monotonic):
                update_strategy_stats(
                    strategy_stats,
                    arm_id=str(current_strategy_arm["id"]),
                    baseline_score=baseline_score,
                    status="crash",
                )
                save_strategy_stats(strategy_stats)
                retry_summaries.append(
                    summarize_same_iteration_retry(
                        attempt_number=attempt_number,
                        status="crash",
                        description=f"[{current_strategy_arm['id']}] {description}",
                        baseline_score=baseline_score,
                        error=str(exc),
                    )
                )
                retry_feedback = build_same_iteration_retry_feedback(
                    status="crash",
                    baseline_score=baseline_score,
                    description=description,
                    scope=scope,
                    changed_files=changed_files,
                    error=str(exc),
                )
                console_print(
                    f"[retry] attempt {attempt_number} ({current_strategy_arm['id']}) failed verification; taking another swing inside this iteration",
                    tone="yellow",
                )
                attempt_number += 1
                continue

            update_strategy_stats(
                strategy_stats,
                arm_id=str(current_strategy_arm["id"]),
                baseline_score=baseline_score,
                status="crash",
            )
            save_strategy_stats(strategy_stats)
            cumulative["crash"] += 1
            append_result(iteration, DUMMY_COMMIT, baseline_score, baseline_score, 0.0, 0.0, "crash", scope, changed_files, str(exc))
            reflection_text = write_failure_reflection(
                iteration=iteration,
                status="crash",
                scope=scope,
                description=description,
                baseline_score=f"{baseline_score:.6f}",
                candidate_score="n/a",
                error=str(exc),
                changed_files=changed_files,
                agent_decisions=agent_decisions,
                tool=tool,
                model=model,
            )
            print_file_sections(candidate_file_buckets)
            print_section("agent decisions", agent_decisions)
            if retry_summaries:
                print_section("same-iteration retries", retry_summaries)
            print_section("reflection", [reflection_text])
            print_copilot_usage("copilot usage", combine_usage(*usage_blocks))
            summary = {
                "iteration": iteration,
                "status": "crash",
                "fatal": False,
                "baseline_score_pct": f"{baseline_score:.6f}",
                "candidate_score_pct": "n/a",
                "commit": DUMMY_COMMIT,
                "error": str(exc),
                "elapsed_seconds": f"{time.monotonic() - t0:.1f}",
            }
            print_iteration_report(
                iteration=iteration,
                total_iterations=total_iterations,
                status="crash",
                baseline_score=f"{baseline_score:.6f}",
                candidate_score="n/a",
                score_delta="n/a",
                scope=scope,
                description=str(exc),
                commit_id=DUMMY_COMMIT,
                elapsed_seconds=f"{time.monotonic() - t0:.1f}",
                file_buckets=candidate_file_buckets,
            )
            print_summary(summary)
            heartbeat.stop()
            return summary

        candidate_usage = candidate.get("copilot_usage")
        usage_blocks.append(candidate_usage)
        candidate_score = score_pct(candidate)
        candidate_signal = assessment_signal(candidate)
        score_delta = candidate_score - baseline_score
        status = "keep" if score_delta > 0 else "discard"

        if status != "keep" and can_retry_within_iteration(attempt_number=attempt_number, deadline_monotonic=deadline_monotonic):
            restore_entries(start_commit, allowed_changed, cwd=ROOT)
            update_strategy_stats(
                strategy_stats,
                arm_id=str(current_strategy_arm["id"]),
                baseline_score=baseline_score,
                status="discard",
                score_delta=score_delta,
            )
            save_strategy_stats(strategy_stats)
            retry_summaries.append(
                summarize_same_iteration_retry(
                    attempt_number=attempt_number,
                    status="discard",
                    description=f"[{current_strategy_arm['id']}] {description}",
                    baseline_score=baseline_score,
                    candidate_score=candidate_score,
                )
            )
            retry_feedback = build_same_iteration_retry_feedback(
                status="discard",
                baseline_score=baseline_score,
                description=description,
                scope=scope,
                changed_files=changed_files,
                candidate_score=candidate_score,
                candidate_report=candidate,
            )
            console_print(
                f"[retry] attempt {attempt_number} ({current_strategy_arm['id']}) scored {candidate_score:.6f} against {baseline_score:.6f}; taking another swing inside this iteration",
                tone="yellow",
            )
            attempt_number += 1
            continue

        heartbeat.set_stage("finalizing iteration")
        todo_review_lines: List[str] = []
        if status == "keep":
            update_strategy_stats(
                strategy_stats,
                arm_id=str(current_strategy_arm["id"]),
                baseline_score=baseline_score,
                status="keep",
                score_delta=score_delta,
            )
            save_strategy_stats(strategy_stats)
            commit_id = commit_candidate(allowed_changed, scope, description)
            baseline_state["commit"] = commit_id
            baseline_state["report"] = candidate
            cumulative["keep"] += 1
            append_result(
                iteration,
                commit_id,
                baseline_score,
                candidate_score,
                score_delta,
                candidate_signal,
                status,
                scope,
                changed_files,
                description,
            )
            todo_review_lines, review_usage = review_kept_candidate(
                iteration=iteration,
                scope=scope,
                description=description,
                commit_id=commit_id,
                changed_files=changed_files,
                tool=tool,
                model=model,
                config=config,
            )
            if review_usage:
                usage_blocks.append(review_usage)
        else:
            restore_entries(start_commit, allowed_changed, cwd=ROOT)
            update_strategy_stats(
                strategy_stats,
                arm_id=str(current_strategy_arm["id"]),
                baseline_score=baseline_score,
                status="discard",
                score_delta=score_delta,
            )
            save_strategy_stats(strategy_stats)
            commit_id = DUMMY_COMMIT
            cumulative["discard"] += 1
            append_result(
                iteration,
                commit_id,
                baseline_score,
                candidate_score,
                score_delta,
                candidate_signal,
                status,
                scope,
                changed_files,
                description,
            )
            reflection_text = write_failure_reflection(
                iteration=iteration,
                status="discard",
                scope=scope,
                description=description,
                baseline_score=f"{baseline_score:.6f}",
                candidate_score=f"{candidate_score:.6f}",
                error="",
                changed_files=changed_files,
                agent_decisions=agent_decisions,
                tool=tool,
                model=model,
                candidate_report=candidate,
            )

        total_usage = combine_usage(*usage_blocks)
        print_file_sections(candidate_file_buckets)
        print_section("agent decisions", agent_decisions)
        if retry_summaries:
            print_section("same-iteration retries", retry_summaries)
        if todo_review_lines:
            print_section("todo review", todo_review_lines)
        if status != "keep":
            print_section("reflection", [reflection_text])
        print_copilot_usage("copilot usage", total_usage)

        summary = {
            "iteration": iteration,
            "status": status,
            "fatal": False,
            "baseline_score_pct": f"{baseline_score:.6f}",
            "candidate_score_pct": f"{candidate_score:.6f}",
            "score_delta": f"{score_delta:.6f}",
            "candidate_signal": f"{candidate_signal:.6f}",
            "gate": candidate["measurement"]["g"],
            "scope": scope,
            "description": description,
            "outcome": outcome_line(status, score_delta, commit_id),
            "commit": commit_id,
            "elapsed_seconds": f"{time.monotonic() - t0:.1f}",
            "backend": candidate["scorer_backend"],
            "tool": tool,
            "cumulative_keep": cumulative["keep"],
            "cumulative_discard": cumulative["discard"],
            "cumulative_crash": cumulative["crash"],
        }
        print_iteration_report(
            iteration=iteration,
            total_iterations=total_iterations,
            status=status,
            baseline_score=f"{baseline_score:.6f}",
            candidate_score=f"{candidate_score:.6f}",
            score_delta=f"{score_delta:.6f}",
            scope=scope,
            description=description,
            commit_id=commit_id,
            elapsed_seconds=f"{time.monotonic() - t0:.1f}",
            file_buckets=candidate_file_buckets,
        )
        print_summary(summary)
        heartbeat.stop()
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run app-focused autoresearch experiments.")
    parser.add_argument("--model", default=None, help="GitHub Copilot CLI model override.")
    parser.add_argument(
        "--tool",
        default=DEFAULT_TOOL,
        choices=[DEFAULT_TOOL],
        help="Agent tool to use for implementation and evaluation.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "python", "rust"],
        help="Scoring backend to use.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of iterations to run. Default: {DEFAULT_ITERATIONS}.",
    )
    parser.add_argument("--forever", action="store_true", help="Keep iterating until interrupted.")
    args = parser.parse_args()

    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if args.forever and args.iterations != 1:
        raise SystemExit("Use either --iterations N or --forever, not both.")

    config = load_runner_config()
    preflight_note = auto_handle_dirty_tree(config)
    if preflight_note:
        console_print(f"[preflight] {preflight_note}", tone="yellow")

    ensure_results_tsv()
    total_iterations = None if args.forever else args.iterations
    iteration = 1
    cumulative = {"keep": 0, "discard": 0, "crash": 0}
    baseline_state: Dict[str, object] = {}
    strategy_stats = load_strategy_stats(config)
    repeated_baseline_failures = 0
    last_baseline_error = ""
    try:
        while True:
            summary = run_single_iteration(
                iteration=iteration,
                total_iterations=total_iterations,
                backend=args.backend,
                tool=args.tool,
                model=args.model,
                cumulative=cumulative,
                baseline_state=baseline_state,
                config=config,
                strategy_stats=strategy_stats,
            )
            if summary.get("status") == "crash" and summary.get("scope") == "baseline":
                current_error = clean_history_text(summary.get("error", ""), fallback="baseline failure")
                if current_error == last_baseline_error:
                    repeated_baseline_failures += 1
                else:
                    last_baseline_error = current_error
                    repeated_baseline_failures = 1
                if repeated_baseline_failures >= int(config["baseline_failure_abort_threshold"]):
                    console_print(
                        f"[abort] baseline failed {repeated_baseline_failures} times in a row with the same error; stopping the run",
                        tone="red",
                    )
                    raise SystemExit(1)
            else:
                repeated_baseline_failures = 0
                last_baseline_error = ""
            if summary.get("fatal"):
                raise SystemExit(1)
            if total_iterations is not None and iteration >= total_iterations:
                break
            iteration += 1
    except KeyboardInterrupt:
        console_print("")
        console_print("Interrupted by user.", tone="yellow")


if __name__ == "__main__":
    main()
