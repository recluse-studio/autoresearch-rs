# autoresearch-app-improvement

This fork keeps the original `autoresearch` shape, but points it at application improvement instead of model training.

The intended UX is Karpathy-style:

- drop the harness into a target app repo
- run one command
- let the loop choose experiments
- keep only score-improving changes

You should not need to author a roadmap, task list, or benchmark file just to get started.

## What stays fixed

- `program.md` is the default guidance for the agent
- `prepare.py` is the fixed harness and scorer plumbing
- `train.py` is the loop runner
- `results.tsv` is the persistent experiment log for fresh agent processes
- `REFLECTION.md` stores the latest short failure reflection for the next iteration
- `TODO.md` stores review debt that must be repaired before chasing novelty

## What changes

- the agent edits application code instead of `train.py`
- GitHub Copilot CLI is the model runner
- the score is your gated KLM/COSMIC roll-up, not `val_bpb`
- the benchmark lens is synthesized automatically on first run and frozen in `.autoresearch-cache/`

## Quick start

Requirements:

- Python 3.10+
- GitHub CLI authenticated with GitHub
- GitHub Copilot CLI available through `gh copilot`
- `cargo` only if you want the local Rust scorer instead of the Python mirror

Starter setup for a new repo:

1. Copy `prepare.py`, `program.md`, `train.py`, `autoresearch_cli.py`, `pyproject.toml`, and `uv.lock` into the target repo root.
2. Copy the files from `place_these_in_root/` into the target repo root.
3. Customize `ABOUT.md`, `BRAND.md`, and `FEATURES.md` so the loop has fast context for that app.
4. Add root-level ignores for `results.tsv`, `REFLECTION.md`, `TODO.md`, and `.autoresearch-cache/` if your repo does not already ignore them.

Install the entrypoint from the repo root:

```bash
pip install -e .
```

Run one iteration:

```bash
python3 train.py
```

Run the default loop length:

```bash
autoresearch
```

Run 250 iterations:

```bash
python3 train.py --iterations 250
```

Run until stopped:

```bash
python3 train.py --forever
```

That is the normal workflow.

The default iteration count is `250`. Use `--iterations` or `--forever` to override it.

The `place_these_in_root/` folder is included to make the first setup in a new app repo faster.

## How it works

On the first scoring pass, the harness:

1. inspects the repo
2. synthesizes a compact frozen benchmark for the run
3. stores it in `.autoresearch-cache/`

On each iteration, the loop:

1. scores the current repo state
2. launches a fresh GitHub Copilot CLI process
3. creates `results.tsv` if needed and supplies recent experiment history to the fresh agent
4. rejects edits that touch the fixed harness or support assets
5. rescales the candidate with the same frozen benchmark
6. keeps the patch only if `score_pct` improved
7. appends a row to `results.tsv`
8. updates `REFLECTION.md` with the latest short reflection after a discard or crash
9. updates `TODO.md` with review debt if a fast post-keep review finds that shipped work is incomplete, bluffing, or only labels without behavior

Each iteration also prints:

- baseline score, candidate score, and score delta
- modified files in the candidate patch
- a short decision summary extracted from the agent output
- explicit keep/discard messaging, including whether the patch is passed to the next iteration
- cumulative keep/discard/crash counts
- Copilot invocation telemetry for the iteration

If the evaluator cannot confidently resolve a repo snapshot, the harness falls back to a neutral baseline measurement instead of crashing. That keeps the loop running while still refusing to credit uncertain gains.

## Files

- `program.md`
  Default agent instructions. You should not need to edit this unless you want special behavior.

- `train.py`
  Main entrypoint. This is the command you normally run.

- `autoresearch_cli.py`
  Console-script entrypoint used by `autoresearch` when the package is installed.

- `prepare.py`
  Optional diagnostics and scoring entrypoint. Useful for debugging the harness, but not required for normal use.

- `place_these_in_root/`
  Starter root files for a fresh target app repo: `ABOUT.md`, `BRAND.md`, `FEATURES.md`, `REFLECTION.md`, `TODO.md`, and `results.tsv`.

## Notes

- `support_docs/` and `support_scripts/` hold precedent and support assets. They are gitignored on purpose.
- If you deliberately place a `benchmark.json` in the repo root, it acts as an advanced manual override. Normal usage does not require this.
- `karpathy-files/` preserves upstream reference material.
- Copilot quota or billing information is not exposed by the current CLI integration. The harness reports invocation count, duration, prompt bytes, and output bytes, and labels quota usage as unavailable.
- The harness treats `results.tsv`, `REFLECTION.md`, and `TODO.md` as root-level memory files in the target repo and expects them to stay untracked.
