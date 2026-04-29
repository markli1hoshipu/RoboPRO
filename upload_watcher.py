"""Wait for in-flight collection jobs to finish, then upload each folder to HF.

Target folders (local -> HF path):
  - chain_heat_hamburger_ks/bench_demo_kitchens_clean        -> kitchens/chain_heat_hamburger_ks/clean
  - chain_heat_hamburger_ks/bench_demo_kitchens_d6..d15      -> kitchens/chain_heat_hamburger_ks/d6..d15
  - chain_serve_hamburger_ks/...                              -> kitchens/chain_serve_hamburger_ks/...
  - move_seal_cup_next_to_box/bench_demo_study_clean         -> study/move_seal_cup_next_to_box/clean
  - move_cups_into_box/bench_demo_study_{clean,d8}           -> study/move_cups_into_box/{clean,d8}
  - move_seal_onto_book/bench_demo_study_clean               -> study/move_seal_onto_book/clean
  - put_can_next_to_basket/bench_demo_kitchens_d9..d15       -> kitchenl/put_can_next_to_basket/d9..d15
  - move_can_from_cabinet_to_basket/bench_demo_kitchens_d7..d10 -> kitchenl/move_can_from_cabinet_to_basket/d7..d10
  - put_can_in_cabinet/bench_demo_kitchens_d6..d9            -> kitchenl/put_can_in_cabinet/d6..d9

Ready = hdf5_count == target (100 for clean, 10 for clutter).
"""
import os, time, glob
from huggingface_hub import HfApi

TOKEN = ""
REPO = "Hoshipu/roboreal_data"
BASE = "/shared_work/robotwin_bench/customized_robotwin/data/bench_data"

# (task, config_dir_name, hf_group, hf_short, target_hdf5)
TARGETS = []

# Chain heat (kitchens): clean=100 + d6..d15=10
TARGETS.append(("chain_heat_hamburger_ks", "bench_demo_kitchens_clean", "kitchens", "clean", 100))
# Chain serve (kitchens): only clean rerun was needed (clutter already uploaded)
TARGETS.append(("chain_serve_hamburger_ks", "bench_demo_kitchens_clean", "kitchens", "clean", 100))
# Wait, clutter for heat/serve was already collected (first pass) but never uploaded.
# Include all configs to be safe.
for cfg in [f"d{i}" for i in range(6, 16)]:
    TARGETS.append(("chain_heat_hamburger_ks", f"bench_demo_kitchens_{cfg}", "kitchens", cfg, 10))
    TARGETS.append(("chain_serve_hamburger_ks", f"bench_demo_kitchens_{cfg}", "kitchens", cfg, 10))

# Study reruns (clean only — clutter already on HF)
TARGETS.append(("move_seal_cup_next_to_box", "bench_demo_study_clean", "study", "clean", 100))
TARGETS.append(("move_cups_into_box",        "bench_demo_study_clean", "study", "clean", 100))
TARGETS.append(("move_cups_into_box",        "bench_demo_study_d8",    "study", "d8",    10))
TARGETS.append(("move_seal_onto_book",       "bench_demo_study_clean", "study", "clean", 100))

# Can tasks (kitchenl)
for cfg in ["d9", "d10", "d11", "d12", "d13", "d14", "d15"]:
    TARGETS.append(("put_can_next_to_basket", f"bench_demo_kitchens_{cfg}", "kitchenl", cfg, 10))
for cfg in ["d7", "d8", "d9", "d10"]:
    TARGETS.append(("move_can_from_cabinet_to_basket", f"bench_demo_kitchens_{cfg}", "kitchenl", cfg, 10))
for cfg in ["d6", "d7", "d8", "d9"]:
    TARGETS.append(("put_can_in_cabinet", f"bench_demo_kitchens_{cfg}", "kitchenl", cfg, 10))

api = HfApi(token=TOKEN)
uploaded = set()

def count_hdf5(task, cfg):
    return len(glob.glob(f"{BASE}/{task}/{cfg}/data/*.hdf5"))

POLL_SEC = 300  # 5 min
MAX_ITERS = 48  # 4h max wait

for it in range(MAX_ITERS):
    remaining = [t for t in TARGETS if (t[0], t[1]) not in uploaded]
    if not remaining:
        print("[ALL DONE] all targets uploaded", flush=True)
        break

    for task, cfg, grp, short, target in list(remaining):
        local = f"{BASE}/{task}/{cfg}"
        if not os.path.isdir(local):
            print(f"[SKIP-nodir] {task}/{cfg}", flush=True)
            uploaded.add((task, cfg))
            continue
        cur = count_hdf5(task, cfg)
        if cur < target:
            print(f"[WAIT] {task}/{cfg}: {cur}/{target}", flush=True)
            continue
        # Ready — upload
        repo_path = f"{grp}/{task}/{short}"
        print(f"[UPLOAD] {local} -> {repo_path} ({cur} hdf5)", flush=True)
        try:
            api.upload_folder(
                folder_path=local, repo_id=REPO, repo_type="dataset",
                path_in_repo=repo_path,
                ignore_patterns=[".cache/*", "*.log"],
                commit_message=f"upload {task}/{short}",
            )
            print(f"[DONE] {task}/{cfg}", flush=True)
            uploaded.add((task, cfg))
        except Exception as e:
            print(f"[ERROR] {task}/{cfg}: {e}", flush=True)

    if len(uploaded) == len(TARGETS):
        break
    print(f"--- iter {it}: {len(uploaded)}/{len(TARGETS)} uploaded, sleeping {POLL_SEC}s ---", flush=True)
    time.sleep(POLL_SEC)

print(f"FINAL: {len(uploaded)}/{len(TARGETS)} uploaded", flush=True)
