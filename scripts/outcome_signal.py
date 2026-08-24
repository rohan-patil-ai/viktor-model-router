#!/usr/bin/env python3
"""Construct an outcome/quality signal from recovered outputs.

The export has no quality labels. But call i's output lives inside call i+1's
input — so we can recover intermediate outputs and score them.

Signals we extract:
1. Error rate — fraction of function_call_outputs containing "error"
2. Task completion — whether submit_draft was called (the final deliverable tool)
3. Tool efficiency — ratio of unique tools used to total function calls
4. Conversation depth — more calls may indicate the model struggled

All signals are heuristic. The writeup must say so.
"""
import json, sys, re
from collections import defaultdict
from load_trajectories import iter_requests, group_trajectories, est_tokens


def count_errors(calls):
    """Count function_call_outputs with error indicators in recovered history."""
    errors = 0
    total = 0
    for c in calls:
        for item in c["input"]:
            if item.get("type") == "function_call_output":
                total += 1
                output = str(item.get("output", ""))
                if re.search(r'"error"', output, re.IGNORECASE) or \
                   re.search(r'"success"\s*:\s*false', output, re.IGNORECASE) or \
                   re.search(r'Error:|Traceback|FAILED|exception', output):
                    errors += 1
    return errors, total


def has_submit_draft(calls):
    """Check if submit_draft was called — indicates the model produced a deliverable."""
    for c in calls:
        for item in c["input"]:
            if item.get("type") == "function_call" and item.get("name") == "submit_draft":
                return True
    return False


def tool_efficiency(calls):
    """Ratio of unique tool names used to total function calls. Higher = more diverse,
    potentially more capable execution. Low ratio with many calls = repetitive retries."""
    names = set()
    total = 0
    for c in calls:
        for item in c["input"]:
            if item.get("type") == "function_call":
                names.add(item.get("name", "?"))
                total += 1
    if total == 0:
        return 1.0
    return len(names) / total


def outcome_score(calls):
    """Composite quality score in [0, 1]. Higher is better.

    Components (weighted):
    - error_free_rate (0.4): fraction of function outputs without errors
    - completion (0.2): 1 if submit_draft called, 0.5 otherwise (most tasks don't use it)
    - efficiency (0.2): tool diversity ratio (penalizes repetitive retries)
    - brevity (0.2): penalty for excessive calls (>4 calls suggests struggling)
    """
    errors, total_outputs = count_errors(calls)
    error_free_rate = 1.0 - (errors / total_outputs) if total_outputs > 0 else 1.0

    completion = 1.0 if has_submit_draft(calls) else 0.5

    efficiency = tool_efficiency(calls)

    # Brevity: 1.0 for 1 call, decays for more (most trajectories are 1 call)
    n = len(calls)
    brevity = 1.0 / (1.0 + 0.2 * max(0, n - 1))

    score = (0.4 * error_free_rate +
             0.2 * completion +
             0.2 * efficiency +
             0.2 * brevity)

    return {
        "outcome_score": round(score, 4),
        "error_free_rate": round(error_free_rate, 4),
        "errors": errors,
        "total_outputs": total_outputs,
        "has_submit_draft": has_submit_draft(calls),
        "tool_efficiency": round(efficiency, 4),
        "n_calls": n,
    }


def main():
    export = sys.argv[1] if len(sys.argv) > 1 else "export"
    groups = group_trajectories(r for _, _, r in iter_requests(export))

    # Per-model outcome stats
    model_scores = defaultdict(list)
    for key, calls in groups.items():
        out = outcome_score(calls)
        model_scores[calls[0]["model"]].append(out["outcome_score"])

    print(f"Scored {len(groups)} trajectories\n")
    print("=== OUTCOME SCORE BY MODEL (higher = better) ===")
    for m in sorted(model_scores):
        vals = model_scores[m]
        avg = sum(vals) / len(vals)
        print(f"  {m:25s}  n={len(vals):4d}  avg={avg:.4f}  "
              f"min={min(vals):.4f}  max={max(vals):.4f}")

    # Overall error stats
    total_errors = 0
    total_outputs = 0
    for key, calls in groups.items():
        out = outcome_score(calls)
        total_errors += out["errors"]
        total_outputs += out["total_outputs"]
    print(f"\nOverall: {total_errors}/{total_outputs} function outputs with errors "
          f"({total_errors/total_outputs*100:.1f}%)")


if __name__ == "__main__":
    main()
