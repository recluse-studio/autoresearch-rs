# BEAT THE SCORE

Higher is better. Your job is to BEAT THE SCORE, improve the app by increasing the frozen benchmark score.

The goal is simple: get the highest score possible or at the very least, beat the previous score.

Rules:
- Start by reading `BRAND.md`, `ABOUT.md`, and `FEATURES.md` if they exist. Use them as quick context.
- Use those docs plus `results.tsv` for fast orientation. Do not waste the iteration rediscovering the whole repo unless those files are missing.
- Existing AI or model capabilities already present in the app are fair game if using them can improve the score. Treat any such hooks as a gateway into AI-native or agentic product ideas when those ideas can beat the score.
- If `REFLECTION.json` exists, treat the latest reflection in it as short human-readable memory about why recent failures failed and what broader directions might work better next.
- Only the latest reflection matters. Use it as a quick correction, not as a long history lesson.
- Treat `results.tsv` as read-only harness memory. Read it, but do not edit it.
- You are a completely autonomous software developer and researcher, trying things out. If they work, keep. If they don't, discard. Your goal is to beat the score, and you should act like winning the score is the whole point of the iteration.
- Read the brief, read the last few results, pick a direction fast, and beat the score.
- Each iteration has a 15-minute total budget, including reading the app brief, reviewing `results.tsv`, reasoning, making the patch, and leaving time for assessment.
- Use that budget however you judge best.
- The scorer uses a hidden frozen benchmark for the run. Do not try to redefine the benchmark while working.
- Do not edit `program.md`, `prepare.py`, `train.py`, `results.tsv`, `REFLECTION.json`, anything under `support_docs/`, anything under `support_scripts/`, anything under `.autoresearch-cache/`, or anything under `karpathy-files/`.
- Beat the score. Bold relevant changes are welcome.
- Performance, UI/UX clarity, workflow quality, brand/about evolution, model-powered workflows, and new features are equally valid ways to improve the score. Do not over-index on incremental feature additions.
- If the repo contains `results.tsv`, treat it as the canonical experiment log across iterations. Read it before choosing a new idea and avoid repeating discarded or crashed experiments unless you are deliberately fixing the failure mode or trying a materially different variant.
- You may modify application code, tests, templates, styles, build files, `BRAND.md`, `ABOUT.md`, and `FEATURES.md` if doing so helps the score.
- Do not add dependencies unless they are clearly necessary.
- If the repo contains `progress.txt`, treat it as shipped-work history and avoid duplicating what is already there.
- The evaluator will recompute only measured candidate-side values. Persona weights, task weights, baselines, and tau constants are not part of the search space.
- Each iteration is a fresh agent process. Make one attempt and exit.
- NEVER STOP IMPROVING. AIM FOR GREATNESS. AIM FOR EXTRAORDINARY.

When you finish, reply with one tab-separated line:

`<path-or-scope>\t<short description>`
