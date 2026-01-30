#  Add paths to the system path
import sys
from pathlib import path

def setup_paths():
    ws = Path(__file__).resolve()parents[2] # workspace root
    robotwin_root = ws / "customized_robotwin"
    bench_root = ws / "benchmark"
    for p in [robotwin_root, bench_root]:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
        if f"{robotwin_root}/script" not in sys.path:
            sys.path.insert(0, f"{robotwin_root}/script")