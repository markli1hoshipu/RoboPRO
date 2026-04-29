"""Compact partial put_can_next_to_basket configs so collect_data.py will
resume first-pass with fresh seeds to backfill missing episodes.

For each config:
  1. Read seed.txt (list of 10 seeds used in first pass)
  2. Find which episode_idx's have hdf5 (successful 2nd-pass)
  3. Rename pkl/hdf5/mp4 files so successful episodes sit at 0..K-1 contiguous
  4. Rewrite seed.txt with just those K seeds
  5. Delete pkls whose episode_idx > K-1 (they were for failed 2nd-pass seeds)

After this, re-running the job: first pass sees suc_num=K, episode_num=10,
tries new seeds (starting from max(seed_list)+1) until it has 10 pkls; second
pass renders the fresh K..9 episodes.
"""
import os, glob, re, shutil, sys

BASE = "/shared_work/robotwin_bench/customized_robotwin/data/bench_data/put_can_next_to_basket"
CONFIGS = [f"bench_demo_kitchens_d{i}" for i in range(6, 16)]

def ep_idx(path, ext):
    m = re.search(rf"episode(\d+)\.{ext}$", path)
    return int(m.group(1)) if m else None

for cfg in CONFIGS:
    d = f"{BASE}/{cfg}"
    if not os.path.isdir(d):
        print(f"SKIP {cfg}: no dir"); continue

    seed_file = f"{d}/seed.txt"
    with open(seed_file) as f:
        seeds = [int(x) for x in f.read().split()]

    hdf5_ids = sorted([ep_idx(p, "hdf5") for p in glob.glob(f"{d}/data/episode*.hdf5")])
    pkl_ids = sorted([ep_idx(p, "pkl") for p in glob.glob(f"{d}/_traj_data/episode*.pkl")])

    print(f"\n== {cfg} ==  seeds={len(seeds)} pkls={len(pkl_ids)} hdf5={len(hdf5_ids)}")

    # Keep only episodes where hdf5 exists. Remap sequentially to 0..K-1.
    kept_seeds = [seeds[i] for i in hdf5_ids]
    K = len(hdf5_ids)

    # Rename hdf5/mp4/pkl for kept episodes to 0..K-1 contiguous
    # Use two-phase rename via temp suffix to avoid clobber
    for new_idx, old_idx in enumerate(hdf5_ids):
        if new_idx == old_idx:
            continue
        for subdir, ext in [("data", "hdf5"), ("video", "mp4"), ("_traj_data", "pkl")]:
            src = f"{d}/{subdir}/episode{old_idx}.{ext}"
            tmp = f"{d}/{subdir}/episode{new_idx}.{ext}.tmp"
            if os.path.exists(src):
                shutil.move(src, tmp)

    # Delete any remaining pkls with idx > K-1 (they're for failed 2nd-pass seeds,
    # or for seeds that were never rendered because collect stopped at 10)
    for p in glob.glob(f"{d}/_traj_data/episode*.pkl"):
        i = ep_idx(p, "pkl")
        if i is not None:
            os.remove(p)

    # Move .tmp files back to final names
    for subdir, ext in [("data", "hdf5"), ("video", "mp4"), ("_traj_data", "pkl")]:
        for tmp in glob.glob(f"{d}/{subdir}/episode*.{ext}.tmp"):
            final = tmp[:-4]  # strip .tmp
            shutil.move(tmp, final)

    # Rewrite seed.txt with only kept seeds
    with open(seed_file, "w") as f:
        for s in kept_seeds:
            f.write(f"{s} ")

    new_hdf5 = sorted([ep_idx(p, "hdf5") for p in glob.glob(f"{d}/data/episode*.hdf5")])
    new_pkl = sorted([ep_idx(p, "pkl") for p in glob.glob(f"{d}/_traj_data/episode*.pkl")])
    print(f"  -> compacted: seeds={K} pkls={new_pkl} hdf5={new_hdf5}")
