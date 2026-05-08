from huggingface_hub import HfApi
import sys, os

api = HfApi()
print(f"[upload] user={api.whoami()['name']}", flush=True)

api.upload_large_folder(
    folder_path="/shared_work/robotwin_bench/benchmark_data",
    repo_id="Hoshipu/roboreal_data",
    repo_type="dataset",
    print_report=True,
    print_report_every=30,
)
print("[upload] DONE", flush=True)
