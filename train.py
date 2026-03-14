"""
Run a single markdown-focused autoresearch experiment.

Usage:
    python train.py
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from prepare import CACHE_DIR, ROOT, is_eligible_markdown, list_markdown_files, score_corpus


RESULTS_PATH = ROOT / "results.tsv"
LAST_AGENT_MESSAGE = CACHE_DIR / "last-agent-message.txt"


def run_command(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(args),
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def ensure_clean_tree() -> None:
    status = run_command(["git", "status", "--porcelain=v1", "--untracked-files=all"]).stdout.strip()
    if status:
        raise RuntimeError("Working tree must be clean before running an experiment.")


def parse_status_lines() -> List[Tuple[str, str]]:
    raw = run_command(["git", "status", "--porcelain=v1", "--untracked-files=all"]).stdout.splitlines()
    parsed: List[Tuple[str, str]] = []
    for line in raw:
        status = line[:2]
        path_text = line[3:]
        if " -> " in path_text:
            _, path_text = path_text.split(" -> ", 1)
        parsed.append((status, path_text))
    return parsed


def classify_candidate_diff() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    eligible: List[Tuple[str, str]] = []
    invalid: List[Tuple[str, str]] = []
    for status, path_text in parse_status_lines():
        if is_eligible_markdown(path_text):
            eligible.append((status, path_text))
            continue
        invalid.append((status, path_text))
    return eligible, invalid


def remove_untracked(path_text: str) -> None:
    path = ROOT / path_text
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def reset_candidate(start_commit: str, changed: Sequence[Tuple[str, str]]) -> None:
    tracked_paths = [path_text for status, path_text in changed if status != "??"]
    if tracked_paths:
        run_command(
            ["git", "restore", "--source", start_commit, "--staged", "--worktree", "--", *tracked_paths]
        )
    for status, path_text in changed:
        if status == "??":
            remove_untracked(path_text)


def read_program() -> str:
    return (ROOT / "program.md").read_text(encoding="utf-8").strip()


def build_prompt(baseline_score: float) -> str:
    corpus_files = len(list_markdown_files())
    return (
        f"{read_program()}\n\n"
        f"Current corpus score: {baseline_score:.6f}\n"
        f"Eligible markdown files: {corpus_files}\n"
        "Work inside this repository. Inspect the markdown corpus, choose exactly one eligible "
        "tracked .md file, and make a small prose-only edit that you expect will reduce the score. "
        "Do not edit program.md, anything under karpathy-files/, or any non-markdown file. "
        "Do not create, delete, or rename files. When you finish, reply with one tab-separated line: "
        "<path>\\t<short description>."
    )


def run_codex(prompt: str, codex_model: Optional[str]) -> subprocess.CompletedProcess:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        "codex",
        "exec",
        "--full-auto",
        "--color",
        "never",
        "--output-last-message",
        str(LAST_AGENT_MESSAGE),
        "--cd",
        str(ROOT),
    ]
    if codex_model:
        command.extend(["--model", codex_model])
    command.append(prompt)
    return run_command(command, check=False)


def parse_agent_message(target_file: str) -> str:
    if not LAST_AGENT_MESSAGE.exists():
        return f"candidate edit for {target_file}"
    message = LAST_AGENT_MESSAGE.read_text(encoding="utf-8").strip().splitlines()
    if not message:
        return f"candidate edit for {target_file}"
    first_line = message[0]
    if "\t" in first_line:
        _, description = first_line.split("\t", 1)
        description = description.strip()
        if description:
            return description
    return first_line.strip() or f"candidate edit for {target_file}"


def commit_candidate(target_file: str, description: str) -> str:
    run_command(["git", "add", "--", target_file])
    summary = description.replace("\n", " ").strip()
    message = f"autoresearch: {target_file}"
    if summary:
        message = f"{message} - {summary[:56]}"
    run_command(["git", "commit", "-m", message])
    return run_command(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()


def ensure_results_tsv() -> None:
    if RESULTS_PATH.exists():
        return
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["commit", "score", "status", "target", "description"])


def append_result(commit_id: str, score: float, status: str, target: str, description: str) -> None:
    ensure_results_tsv()
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([commit_id, f"{score:.6f}", status, target, description])


def print_summary(payload) -> None:
    print("---")
    for key, value in payload.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one markdown autoresearch experiment.")
    parser.add_argument("--model", default=None, help="Codex model override.")
    parser.add_argument("--scorer-model", default="gpt2", help="Frozen scoring model.")
    args = parser.parse_args()

    t0 = time.time()
    ensure_clean_tree()
    start_commit = run_command(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
    baseline = score_corpus(args.scorer_model)
    codex_result = run_codex(build_prompt(baseline.score), args.model)
    eligible_changed, invalid_changed = classify_candidate_diff()
    all_changed = eligible_changed + invalid_changed

    if codex_result.returncode != 0:
        error_text = (codex_result.stderr or codex_result.stdout).strip().splitlines()
        error_detail = error_text[-1] if error_text else "codex exec failed"
        reset_candidate(start_commit, all_changed)
        append_result("0000000", baseline.score, "crash", "-", error_detail)
        print_summary(
            {
                "status": "crash",
                "baseline_score": f"{baseline.score:.6f}",
                "candidate_score": "n/a",
                "target_file": "-",
                "commit": "0000000",
                "error": error_detail,
                "elapsed_seconds": f"{time.time() - t0:.1f}",
            }
        )
        raise SystemExit(codex_result.returncode)

    if invalid_changed or len(eligible_changed) != 1:
        description = "invalid edit scope"
        if all_changed:
            description = f"{description}: " + ", ".join(path for _, path in all_changed)
        reset_candidate(start_commit, all_changed)
        append_result("0000000", baseline.score, "discard", "-", description)
        print_summary(
            {
                "status": "discard",
                "baseline_score": f"{baseline.score:.6f}",
                "candidate_score": "n/a",
                "target_file": "-",
                "commit": "0000000",
                "elapsed_seconds": f"{time.time() - t0:.1f}",
            }
        )
        return

    _, target_file = eligible_changed[0]
    description = parse_agent_message(target_file)
    candidate_commit = commit_candidate(target_file, description)
    candidate = score_corpus(args.scorer_model)
    delta = candidate.score - baseline.score
    status = "keep" if delta < 0 else "discard"
    append_result(candidate_commit, candidate.score, status, target_file, description)
    if status == "discard":
        run_command(["git", "reset", "--hard", start_commit])

    print_summary(
        {
            "status": status,
            "baseline_score": f"{baseline.score:.6f}",
            "candidate_score": f"{candidate.score:.6f}",
            "delta": f"{delta:.6f}",
            "target_file": target_file,
            "commit": candidate_commit,
            "elapsed_seconds": f"{time.time() - t0:.1f}",
            "scorer": args.scorer_model,
        }
    )


if __name__ == "__main__":
    main()
