#!/usr/bin/env python
"""
Generate per-episode instruction sidecars for every (scene, task, split, episode)
present locally. Each sidecar randomly picks ONE phrasing from
{original} ∪ {20 synonyms} and writes:

  data/bench_data/{task}/{split}/instructions/episode{N}.json
    -> {"seen": [<chosen_text>], "unseen": []}

The original instruction is included as candidate index 0; the 20 synonyms are
indices 1..20. The chosen index is recorded in `_meta` for reproducibility.
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LANG_DIR = REPO_ROOT / "benchmark" / "bench_description" / "task_language"
DATA_ROOT = REPO_ROOT / "customized_robotwin" / "data" / "bench_data"

# Map task name -> scene by scanning the language files.
def load_task_language():
    by_task = {}
    for scene in ("kitchenl", "kitchens", "office", "study"):
        f = LANG_DIR / f"{scene}.json"
        if not f.exists():
            print(f"WARN: missing {f}")
            continue
        with open(f) as fh:
            data = json.load(fh)
        for task, payload in data.items():
            by_task[task] = (scene, payload)
    return by_task


def list_episodes(task_dir: Path):
    """Yield (split_name, episode_int) for every hdf5 under task_dir/<split>/data."""
    if not task_dir.is_dir():
        return
    for split_dir in sorted(task_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        data_dir = split_dir / "data"
        if not data_dir.is_dir():
            continue
        for f in sorted(data_dir.iterdir()):
            m = re.match(r"^episode(\d+)\.hdf5$", f.name)
            if m:
                yield split_dir.name, int(m.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-task", type=str, default=None,
                        help="If set, only process this task name.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    by_task = load_task_language()
    print(f"Loaded language for {len(by_task)} tasks")

    written = 0
    skipped_no_data = 0
    per_task = defaultdict(int)

    for task, (scene, payload) in sorted(by_task.items()):
        if args.only_task and task != args.only_task:
            continue
        original = payload["original"]
        synonyms = payload["synonyms"]
        candidates = [original] + list(synonyms)  # length 21 (or 1 + len(synonyms))
        task_dir = DATA_ROOT / task
        if not task_dir.is_dir():
            skipped_no_data += 1
            continue

        for split_name, ep_idx in list_episodes(task_dir):
            instr_dir = task_dir / split_name / "instructions"
            instr_dir.mkdir(exist_ok=True)
            out = instr_dir / f"episode{ep_idx}.json"
            chosen_idx = rng.randrange(len(candidates))
            chosen = candidates[chosen_idx]
            obj = {
                "seen": [chosen],
                "unseen": [],
                "_meta": {
                    "task": task,
                    "scene": scene,
                    "candidate_index": chosen_idx,
                    "is_original": chosen_idx == 0,
                    "n_candidates": len(candidates),
                },
            }
            if not args.dry_run:
                with open(out, "w") as fh:
                    json.dump(obj, fh, indent=2)
            written += 1
            per_task[task] += 1

    print()
    print(f"Wrote {written} sidecar files (dry-run={args.dry_run}).")
    print(f"Tasks with no local data dir: {skipped_no_data}")
    print()
    for t, n in sorted(per_task.items()):
        print(f"  {t}: {n}")


if __name__ == "__main__":
    main()
