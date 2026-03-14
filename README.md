# autoresearch-rs

This fork keeps the original `autoresearch` loop shape but swaps model training for markdown prose revision.

## What stayed the same

- `program.md` is still the operational instruction file for the agent.
- `prepare.py` is still the fixed harness.
- `train.py` is still the one-command experiment runner.
- `results.tsv` is still the untracked experiment log.
- The loop is still score, edit, keep/discard, repeat.

## What changed

- The corpus is tracked `.md` files in the repository.
- The runner uses `codex exec` to make one prose edit at a time.
- The evaluator uses a frozen language model and scores the corpus with negative corpus-average WPSLOR.
- Lower score is better.

## Quick start

Requirements:

- Python 3.10+
- `uv`
- Codex CLI authenticated with your ChatGPT account

```bash
# 1. Install dependencies
uv sync

# 2. Verify the environment and corpus
uv run prepare.py check

# 3. Inspect the current corpus score
uv run prepare.py score

# 4. Run one prose experiment
uv run train.py
```

The runner will:

1. score the current markdown corpus
2. ask Codex to choose one markdown file and edit it
3. reject the run if the edit scope is larger than one eligible markdown file
4. rescore the corpus
5. keep the commit if the score improved, otherwise reset back
6. append a row to `results.tsv`

## Repo notes

- `karpathy-files/` preserves upstream originals and moved upstream artifacts.
- `program.md` stays intentionally minimal for now.
- The current scorer is a prototype metric for normalized fluency and acceptability, not a full prose-quality judge.
