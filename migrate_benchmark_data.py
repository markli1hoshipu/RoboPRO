"""Migrate complete task-configs from data/bench_data/{task}/{cfg} to benchmark_data/{group}/{task}/{sub}.

Skips task-configs currently being written by active SLURM jobs.
"""
import os, re, sys, shutil, subprocess
from pathlib import Path

DATA_ROOT = Path("/shared_work/robotwin_bench/customized_robotwin/data/bench_data")
STAGE_ROOT = Path("/shared_work/robotwin_bench/benchmark_data")
BENCH_ENVS = Path("/shared_work/robotwin_bench/benchmark/bench_envs")

def task_group(task):
    for g in ("office", "study", "kitchenl", "kitchens"):
        if (BENCH_ENVS / g / f"{task}.py").exists():
            return g
    return None

def cfg_to_sub(cfg):
    if cfg == "bench_demo_clean" or cfg.endswith("_clean"):
        return "clean"
    m = re.search(r"(d\d+)$", cfg)
    return m.group(1) if m else None

def is_complete(d, sub):
    target = 100 if sub == "clean" else 10
    h = len(list((d / "data").glob("*.hdf5"))) if (d / "data").is_dir() else 0
    v = len(list((d / "video").glob("*.mp4"))) if (d / "video").is_dir() else 0
    return h >= target and v >= target

def running_task_cfgs():
    """Parse out 'task=X config=Y' from slurm logs of RUNNING jobs."""
    out = subprocess.check_output(
        ["squeue", "-u", os.getlogin(), "-t", "RUNNING",
         "-h", "-o", "%i %j"]
    ).decode()
    pairs = set()
    for line in out.strip().splitlines():
        parts = line.split()
        if not parts: continue
        jid, jname = parts[0], parts[1]
        # array job ids look like 1226_33
        if "_" not in jid: continue
        arr = jid.split("_")[-1]
        prefix = "kl" if jname.startswith("rtwkl") else "slurm"
        for logf in (f"/shared_work/robotwin_bench/customized_robotwin/logs/{prefix}-{jid.replace('_','_')}.out",
                     f"/shared_work/robotwin_bench/customized_robotwin/logs/{prefix}-{jid}.out"):
            p = Path(logf)
            if not p.exists(): continue
            text = p.read_text(errors="ignore").splitlines()
            for ln in text[:20]:
                m = re.search(r"task=(\S+)\s+config=(\S+)", ln)
                if m:
                    pairs.add((m.group(1), m.group(2)))
                    break
            break
    return pairs

def main():
    running = running_task_cfgs()
    print(f"skipping {len(running)} running task-cfgs: {sorted(running)}")
    moved = kept = skipped = 0
    if not DATA_ROOT.exists():
        print(f"no data root: {DATA_ROOT}")
        return
    for task_dir in sorted(DATA_ROOT.iterdir()):
        if not task_dir.is_dir(): continue
        task = task_dir.name
        group = task_group(task)
        if not group:
            continue
        for cfg_dir in sorted(task_dir.iterdir()):
            if not cfg_dir.is_dir(): continue
            cfg = cfg_dir.name
            sub = cfg_to_sub(cfg)
            if not sub:
                continue
            if (task, cfg) in running:
                print(f"  SKIP running: {task}/{cfg}")
                skipped += 1
                continue
            if not is_complete(cfg_dir, sub):
                print(f"  SKIP incomplete: {task}/{cfg}")
                kept += 1
                continue
            dst = STAGE_ROOT / group / task / sub
            if dst.exists():
                print(f"  SKIP dst exists: {dst}")
                kept += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"  MV {task}/{cfg}  ->  {group}/{task}/{sub}")
            shutil.move(str(cfg_dir), str(dst))
            moved += 1
    # clean up empty task dirs
    for task_dir in list(DATA_ROOT.iterdir()):
        if task_dir.is_dir() and not any(task_dir.iterdir()):
            task_dir.rmdir()
    print(f"\nDONE: moved={moved} kept={kept} skipped_running={skipped}")

if __name__ == "__main__":
    main()
