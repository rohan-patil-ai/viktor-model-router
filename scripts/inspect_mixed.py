#!/usr/bin/env python3
"""Dump every multi-model trajectory (premise violation) to a readable text file.

These are the closest thing to a real experiment in the dataset: the same task,
switching models mid-way. Read them to look for retry/error/re-do patterns right
after a switch -- that is a candidate outcome signal.

Usage: python scripts/inspect_mixed.py export/ > results/mixed_trajectories.txt
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from load_trajectories import iter_requests, group_trajectories, first_user_text

def item_text(item):
    role = item.get("role") or item.get("type")
    if role == "function_call":
        return f"[function_call] {item.get('name', '?')}({str(item.get('arguments',''))[:200]})"
    if role == "function_call_output":
        out = item.get("output", "")
        return f"[function_call_output] {str(out)[:300]}"
    if role in ("user", "assistant", "system"):
        c = item.get("content")
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            text = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") in ("input_text", "output_text"))
        else:
            text = ""
        return f"[{role}] {text[:300]}"
    return f"[{role}] {json.dumps(item)[:200]}"

def main():
    export = sys.argv[1] if len(sys.argv) > 1 else "export"
    reqs = [r for _, _, r in iter_requests(export)]
    groups = group_trajectories(reqs)
    mixed = [(k, v) for k, v in groups.items() if len({r["model"] for r in v}) > 1]
    print(f"# {len(mixed)} multi-model trajectories\n")
    for k, calls in mixed:
        print(f"{'='*80}")
        print(f"trajectory {k}  ({len(calls)} calls)")
        print(f"opening user text: {first_user_text(calls[0])[:200]!r}")
        prev_model = None
        for i, c in enumerate(calls):
            switch = " <-- MODEL SWITCH" if prev_model and c["model"] != prev_model else ""
            print(f"\n  -- call {i+1}/{len(calls)}: model={c['model']}{switch}")
            # print the new items only (this call's input minus the previous call's input)
            new_items = c["input"][len(calls[i-1]["input"]):] if i > 0 else c["input"]
            for item in new_items[-4:]:
                print(f"     {item_text(item)}")
            prev_model = c["model"]
        print()

if __name__ == "__main__":
    main()
