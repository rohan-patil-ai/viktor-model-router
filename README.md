# Viktor Model Router

Starter kit for the **Viktor Challenge** at the TUM.ai hackathon (Munich, 22–23 Aug 2026).
From real LLM-request logs, build a router that picks the right model for every call —
then prove it works, even though the log shows only the model that ran, and no outputs or token counts.

## Quick start

```bash
# 1. No dataset yet? Generate a synthetic sample with the same shape:
python3 scripts/make_synthetic_sample.py            # writes ./export/

# 2. Got the real dataset links (shipped at kickoff)? Then instead: the export ships
#    as trajectories_v1_<index>.jsonl.tar.gz archives — download, verify the posted
#    SHA-256, then:  mkdir -p export && tar xzf trajectories_v1_01.jsonl.tar.gz -C export/

# 3. Sanity-check the export, reconstruct trajectories, print stats:
python3 scripts/load_trajectories.py export/

# 4. Run the baseline heuristic router + cache-aware cost report:
python3 scripts/baseline_router.py export/

# 5. Turn results into a cost–quality frontier CSV (+ PNG if matplotlib is installed):
python3 scripts/plot_frontier.py results/routes.jsonl
```

## Run this model

The trained router UI uses the RouteLLM-style classifier and nearest historical
trajectory outcomes:

```bash
python3 scripts/router_ui.py 8080
```

Open <http://localhost:8080>.

Retrain the classifier on local challenge exports:

```bash
python3 scripts/routellm_classifier.py export/
```

The trained artifact is `results/routellm_ui_model.pkl`.

Python 3.10+, standard library only (matplotlib optional for the PNG).

## Using a coding agent

Point Claude Code / Codex / Cursor / opencode at this repo — `AGENTS.md` briefs your agent.
In Claude Code you also get slash commands:

- `/setup` — set up everything needed to participate
- `/make-presentation` — build a Viktor-branded presentation of your solution
- `/prepare-submission` — package your solution into a formal submission

## What's here

| Path | What |
|---|---|
| `AGENTS.md` | Agent briefing: dataset shape, the cache trap, judging, starter ideas |
| `skills/` | The three guided workflows above (plain Markdown, readable by humans too) |
| `scripts/` | Baseline router, trained RouteLLM-style router, loader, outcome signal, cache-aware cost model, frontier plot, and synthetic sample |
| `templates/presentation.html` | Self-contained branded slide template |

## Rules that matter

- **License:** challenge use only — no redistribution of the dataset. Full terms ship with the download.
- No GPU or API keys needed. Judge-model rescoring is allowed (credits announced at kickoff).
- Questions → the challenge Discord; the Viktor team answers there all weekend.
