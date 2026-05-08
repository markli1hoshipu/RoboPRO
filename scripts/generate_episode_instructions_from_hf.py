#!/usr/bin/env python
"""
Generate per-episode instruction sidecars for every (scene, task, split, episode)
that exists on HuggingFace. Writes files to a local staging directory matching
the HF repo layout:

  <staging>/{scene}/{task}/{split_alias}/instructions/episode{N}.json
    -> {"seen": [<chosen_text>], "unseen": [], "_meta": {...}}

Split alias is the HF-side name (e.g. "clean", "d10"), not the local
`bench_demo_kitchens_d10` dir name.

Each sidecar picks one phrasing uniformly at random from {original} ∪ {20 synonyms}.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi

REPO = "Hoshipu/roboreal_data"
REPO_ROOT = Path(__file__).resolve().parent.parent
LANG_DIR = REPO_ROOT / "benchmark/bench_description/task_language"


def load_task_language():
    by_task = {}
    for scene in ("kitchenl", "kitchens", "office", "study"):
        f = LANG_DIR / f"{scene}.json"
        with open(f) as fh:
            data = json.load(fh)
        for task, payload in data.items():
            by_task[task] = (scene, payload)
    return by_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--staging", type=str,
                        default=str(REPO_ROOT / "customized_robotwin/data/bench_instructions"))
    parser.add_argument("--cache", type=str, default="/tmp/hf_episode_inventory.json")
    parser.add_argument("--refresh", action="store_true",
                        help="Force refresh of HF file listing.")
    args = parser.parse_args()

    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    by_task = load_task_language()

    # Fetch HF file listing and extract (scene, task, split, episode).
    cache = Path(args.cache)
    if cache.exists() and not args.refresh:
        inventory = json.loads(cache.read_text())
    else:
        print("Fetching HF file listing...")
        api = HfApi()
        files = api.list_repo_files(REPO, repo_type="dataset")
        pat = re.compile(r"^([^/]+)/([^/]+)/([^/]+)/data/episode(\d+)\.hdf5$")
        inventory = []
        for f in files:
            m = pat.match(f)
            if m:
                inventory.append(list(m.groups()))
        cache.write_text(json.dumps(inventory))
        print(f"Cached {len(inventory)} hdf5 references to {cache}")

    # Group by (scene, task, split).
    eps_by_slot = defaultdict(set)
    for scene, task, split, ep in inventory:
        eps_by_slot[(scene, task, split)].add(int(ep))

    written = 0
    per_task = defaultdict(int)

    for (scene, task, split), eps in sorted(eps_by_slot.items()):
        if task not in by_task:
            print(f"WARN: no language for task {task}")
            continue
        scene_tag, payload = by_task[task]
        original = payload["original"]
        synonyms = list(payload["synonyms"])
        candidates = [original] + synonyms  # len = 21

        out_dir = staging / scene / task / split / "instructions"
        out_dir.mkdir(parents=True, exist_ok=True)
        for ep in sorted(eps):
            idx = rng.randrange(len(candidates))
            obj = {
                "seen": [candidates[idx]],
                "unseen": [],
                "_meta": {
                    "task": task,
                    "scene": scene,
                    "candidate_index": idx,
                    "is_original": idx == 0,
                    "n_candidates": len(candidates),
                },
            }
            (out_dir / f"episode{ep}.json").write_text(json.dumps(obj, indent=2))
            written += 1
            per_task[f"{scene}/{task}"] += 1

    print()
    print(f"Wrote {written} sidecar files under {staging}/")
    print()
    # Report incomplete tasks
    for k in sorted(per_task):
        if per_task[k] != 200:
            print(f"  {k}: {per_task[k]}/200")


if __name__ == "__main__":
    main()
