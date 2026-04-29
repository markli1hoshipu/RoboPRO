import os
import sys
from huggingface_hub import HfApi

TOKEN = ""
REPO = "Hoshipu/roboreal_data"
BASE = "/shared_work/robotwin_bench/customized_robotwin/data/bench_data"

TASKS = [
    "close_microwave_ks",
    "put_hamburger_in_microwave_ks",
    "pick_hamburger_from_microwave_ks",
    "chain_apple_bin_bowl_rack_spoon_sink_ks",
    "chain_apple_sink_plate_bread_board_ks",
    "chain_bowl_rack_apple_sink_ks",
]

CONFIGS = ["clean"] + [f"d{i}" for i in range(6, 16)]

api = HfApi(token=TOKEN)

for task in TASKS:
    for cfg in CONFIGS:
        local_dir = f"{BASE}/{task}/bench_demo_kitchens_{cfg}"
        if not os.path.isdir(local_dir):
            print(f"[SKIP] missing: {local_dir}", flush=True)
            continue
        repo_path = f"kitchens/{task}/{cfg}"
        print(f"[UPLOAD] {local_dir} -> {repo_path}", flush=True)
        try:
            api.upload_folder(
                folder_path=local_dir,
                repo_id=REPO,
                repo_type="dataset",
                path_in_repo=repo_path,
                ignore_patterns=[".cache/*", "*.log"],
                commit_message=f"upload {task}/{cfg}",
            )
            print(f"[DONE] {task}/{cfg}", flush=True)
        except Exception as e:
            print(f"[ERROR] {task}/{cfg}: {e}", flush=True)

print("ALL UPLOADS COMPLETE", flush=True)
