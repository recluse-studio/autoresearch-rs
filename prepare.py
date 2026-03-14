"""
Preparation and scoring utilities for markdown-focused autoresearch.

Usage:
    python prepare.py check
    python prepare.py score
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".autoresearch-cache"
CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"
DEFAULT_SCORER_MODEL = "gpt2"
EXCLUDED_MARKDOWN = {
    "program.md",
}
EXCLUDED_PREFIXES = (
    "karpathy-files/",
    ".git/",
    ".autoresearch-cache/",
)
_SCORER_CACHE = {}


@dataclass
class EnvironmentReport:
    repo_root: str
    codex_path: str
    codex_authenticated: bool
    corpus_files: int
    scorer_model: str
    scorer_device: str


@dataclass
class CorpusScore:
    score: float
    file_count: int
    segment_count: int
    token_count: int
    model_name: str
    device: str


def run_git(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def is_eligible_markdown(path: str) -> bool:
    posix_path = path.replace("\\", "/")
    if not posix_path.endswith(".md"):
        return False
    if posix_path in EXCLUDED_MARKDOWN:
        return False
    return not any(posix_path.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def list_markdown_files() -> List[Path]:
    tracked = run_git(["ls-files"]).splitlines()
    return [ROOT / path for path in tracked if is_eligible_markdown(path)]


def strip_yaml_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    marker = "\n---\n"
    end = text.find(marker, 4)
    if end == -1:
        return text
    return text[end + len(marker) :]


def extract_visible_prose(markdown: str) -> List[str]:
    text = strip_yaml_frontmatter(markdown)
    text = re.sub(r"(?ms)^```.*?^```[ \t]*\n?", "\n", text)
    text = re.sub(r"(?ms)^~~~.*?^~~~[ \t]*\n?", "\n", text)
    text = re.sub(r"(?m)^\[[^\]]+\]:\s+\S+.*$", "", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = re.sub(r"^\s{0,3}>\s?", "", line)
        line = re.sub(r"^\s{0,3}(?:[-+*]|\d+\.)\s+", "", line)
        line = line.replace("|", " ")
        line = re.sub(r"\s+", " ", line).strip()
        cleaned_lines.append(line)
    normalized = "\n".join(cleaned_lines)
    blocks = re.split(r"\n\s*\n+", normalized)
    segments = []
    for block in blocks:
        block = re.sub(r"\s+", " ", block).strip()
        if len(block) < 3:
            continue
        if not re.search(r"[A-Za-z]", block):
            continue
        segments.append(block)
    return segments


def iter_corpus_segments(paths: Sequence[Path]) -> Iterable[str]:
    for path in paths:
        markdown = path.read_text(encoding="utf-8")
        for segment in extract_visible_prose(markdown):
            yield segment


def load_runtime():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch, AutoTokenizer, AutoModelForCausalLM, device


def load_scorer(model_name: str):
    if model_name in _SCORER_CACHE:
        return _SCORER_CACHE[model_name]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch, AutoTokenizer, AutoModelForCausalLM, device = load_runtime()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    bundle = (torch, tokenizer, model, device)
    _SCORER_CACHE[model_name] = bundle
    return bundle


def build_unigram_log_probs(tokenized_segments: Sequence[Sequence[int]], vocab_size: int) -> Dict[int, float]:
    counts: Counter[int] = Counter()
    total = 0
    for token_ids in tokenized_segments:
        counts.update(token_ids)
        total += len(token_ids)
    if total == 0:
        raise ValueError("No prose tokens found in the markdown corpus.")
    denominator = total + vocab_size
    return {token_id: math.log((count + 1) / denominator) for token_id, count in counts.items()}


def sequence_log_probability(torch_module, model, prefix_ids: Sequence[int], device: str, max_positions: int) -> float:
    token_tensor = torch_module.tensor(prefix_ids, dtype=torch_module.long, device=device)
    total = 0.0
    start = 0
    with torch_module.no_grad():
        while start < token_tensor.numel() - 1:
            end = min(start + max_positions, token_tensor.numel())
            window = token_tensor[start:end]
            logits = model(window.unsqueeze(0)).logits[0]
            log_probs = torch_module.log_softmax(logits[:-1], dim=-1)
            targets = window[1:]
            total += log_probs.gather(-1, targets.unsqueeze(-1)).sum().item()
            if end == token_tensor.numel():
                break
            start = end - 1
    return total


def score_corpus(model_name: str = DEFAULT_SCORER_MODEL) -> CorpusScore:
    paths = list_markdown_files()
    if not paths:
        raise ValueError("No eligible tracked markdown files were found.")
    segments = list(iter_corpus_segments(paths))
    if not segments:
        raise ValueError("Eligible markdown files exist, but no visible prose was extracted.")
    torch_module, tokenizer, model, device = load_scorer(model_name)
    tokenized_segments = [
        tokenizer.encode(segment, add_special_tokens=False)
        for segment in segments
    ]
    tokenized_segments = [token_ids for token_ids in tokenized_segments if token_ids]
    if not tokenized_segments:
        raise ValueError("Visible prose was extracted, but tokenization produced zero tokens.")
    unigram_log_probs = build_unigram_log_probs(tokenized_segments, tokenizer.vocab_size)
    fallback_log_prob = math.log(1 / (sum(len(token_ids) for token_ids in tokenized_segments) + tokenizer.vocab_size))
    bos_token_id = tokenizer.eos_token_id
    if bos_token_id is None:
        raise ValueError(f"Tokenizer for {model_name} does not expose an EOS token.")
    max_positions = int(getattr(model.config, "n_positions", 1024))
    total_model_log_prob = 0.0
    total_unigram_log_prob = 0.0
    total_tokens = 0
    for token_ids in tokenized_segments:
        total_tokens += len(token_ids)
        total_unigram_log_prob += sum(unigram_log_probs.get(token_id, fallback_log_prob) for token_id in token_ids)
        prefixed = [bos_token_id, *token_ids]
        total_model_log_prob += sequence_log_probability(
            torch_module,
            model,
            prefixed,
            device,
            max_positions,
        )
    score = -((total_model_log_prob - total_unigram_log_prob) / total_tokens)
    return CorpusScore(
        score=score,
        file_count=len(paths),
        segment_count=len(tokenized_segments),
        token_count=total_tokens,
        model_name=model_name,
        device=device,
    )


def check_environment(model_name: str = DEFAULT_SCORER_MODEL) -> EnvironmentReport:
    codex_path = subprocess.run(
        ["python3", "-c", "import shutil; print(shutil.which('codex') or '')"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    codex_authenticated = False
    if CODEX_AUTH_PATH.exists():
        try:
            auth_data = json.loads(CODEX_AUTH_PATH.read_text(encoding="utf-8"))
            codex_authenticated = bool(auth_data.get("auth_mode"))
        except json.JSONDecodeError:
            codex_authenticated = False
    torch_module, _, _, device = load_scorer(model_name)
    del torch_module
    return EnvironmentReport(
        repo_root=str(ROOT),
        codex_path=codex_path,
        codex_authenticated=codex_authenticated,
        corpus_files=len(list_markdown_files()),
        scorer_model=model_name,
        scorer_device=device,
    )


def emit(payload, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
        return
    for key, value in asdict(payload).items():
        print(f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and score the markdown autoresearch corpus.")
    parser.add_argument("--model", default=DEFAULT_SCORER_MODEL, help="Frozen scoring model to use.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("check", help="Verify runtime prerequisites and corpus discovery.")
    subparsers.add_parser("score", help="Compute the corpus score.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command or "check"
    if command == "check":
        emit(check_environment(args.model), args.json)
        return
    if command == "score":
        emit(score_corpus(args.model), args.json)
        return
    parser.error(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
