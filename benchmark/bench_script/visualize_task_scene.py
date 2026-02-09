"""
Simple visualization script for benchmark task scenes.
Uses environments from bench_envs and the same config layout as collect_data.

USAGE:
    Run this script from the benchmark folder:
    
    cd benchmark
    source set_env.sh  # or set ROBOTWIN_ROOT and BENCH_ROOT manually
    python bench_script/visualize_task_scene.py <task_name> <task_config> [options]

EXAMPLES:
    # Basic usage with default seed
    python bench_script/visualize_task_scene.py grab_roller_thing bench_demo_clean
    
    # With custom seed
    python bench_script/visualize_task_scene.py grab_roller_thing bench_demo_clean --seed 42
    
    # With custom render frequency
    python bench_script/visualize_task_scene.py grab_roller_thing bench_demo_clean --render-freq 5

    # Roll out the task (run play_once) then view
    python bench_script/visualize_task_scene.py put_away_stapler bench_demo_clean --rollout

ARGUMENTS:
    task_name      Task module name from bench_envs (e.g. grab_roller_thing)
    task_config    Task config name without .yml extension (e.g. bench_demo_clean)

OPTIONS:
    --seed N           Random seed for scene initialization (default: 0)
    --render-freq N    Render every N simulation steps (default: 1)
    --rollout          Run play_once() to roll out the task; if not set, only view initial setup

NOTES:
    - The script automatically changes directory to customized_robotwin for proper path resolution
    - Requires ROBOTWIN_ROOT and BENCH_ROOT environment variables (set via set_env.sh)
    - Close the viewer window to exit the visualization
"""
import sys
import os
import argparse
import importlib
import yaml
from pathlib import Path

from setup_paths import setup_paths
setup_paths()
# Paths: script is run from benchmark folder, but changes to customized_robotwin
current_file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file_path)
# bench_root is the parent of bench_script directory
bench_root = Path(os.environ["BENCH_ROOT"])
robotwin_root = Path(os.environ["ROBOTWIN_ROOT"])

os.chdir(robotwin_root)  # Change to customized_robotwin for proper path resolution

from envs import CONFIGS_PATH  # from customized_robotwin


def get_env_class(task_name):
    """Load task env class from bench_envs, or envs if not in bench_envs. Handles class name matching module or not."""
    try:
        envs_module = importlib.import_module(f"bench_envs.{task_name}")
    except ModuleNotFoundError:
        envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        return getattr(envs_module, task_name)
    except AttributeError:
        # Class name may differ from module name (e.g. module grab_roller_thing, class grab_roller)
        from envs._base_task import Base_Task
        for name in dir(envs_module):
            obj = getattr(envs_module, name)
            if isinstance(obj, type) and issubclass(obj, Base_Task) and obj is not Base_Task:
                return obj
        raise SystemExit(f"No task class found in bench_envs.{task_name}")


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def main():
    parser = argparse.ArgumentParser(description="Visualize a benchmark task scene")
    parser.add_argument("task_name", type=str, help="Task module name (e.g. grab_roller_thing)")
    parser.add_argument("task_config", type=str, help="Task config name (e.g. bench_demo_clean)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for scene")
    parser.add_argument("--render-freq", type=int, default=3, help="Render every N steps (default 1)")
    parser.add_argument("--rollout", action="store_true", help="Run play_once() to roll out the task")
    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    seed = args.seed
    render_freq = args.render_freq
    rollout = args.rollout

    # Load env class from bench_envs
    env_class = get_env_class(task_name)
    config_path = bench_root / "bench_task_config" / f"{task_config}.yml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    cfg["task_name"] = task_name
    cfg["render_freq"] = render_freq
    cfg["now_ep_num"] = 0
    cfg["seed"] = seed
    cfg["need_plan"] = True
    cfg["save_data"] = False

    # Embodiment setup (same as collect_data)
    embodiment_type = cfg.get("embodiment", ["aloha-agilex"])
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type_name):
        robot_file = _embodiment_types[embodiment_type_name]["file_path"]
        if robot_file is None:
            raise SystemExit("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        cfg["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        cfg["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        cfg["embodiment_dis"] = embodiment_type[2]
        cfg["dual_arm_embodied"] = False
    else:
        raise SystemExit("embodiment config should have 1 or 3 entries")

    cfg["left_embodiment_config"] = get_embodiment_config(cfg["left_robot_file"])
    cfg["right_embodiment_config"] = get_embodiment_config(cfg["right_robot_file"])

    # Build env and setup scene with viewer
    # print(f"Loading task: {task_name} with config: {task_config} (seed={seed})")
    env = env_class()
    try:
        env.setup_demo(**cfg)
    except Exception as e:
        print("Setup failed:", e)
        raise SystemExit(1)

    if not getattr(env, "render_freq", 0) or getattr(env, "viewer", None) is None:
        print("Warning: viewer not created (render_freq was 0?). Exiting.")
        env.close_env()
        return

    viewer = env.viewer
    if rollout:
        print("Rolling out task (play_once)...")
        env.play_once()
        print("Rollout done. Close the viewer window to exit.")
    else:
        print("Scene ready. Close the viewer window to exit.")

    while not viewer.closed:
        env.scene.step()
        env.scene.update_render()
        viewer.render()

    env.close_env()
    print("Done.")


if __name__ == "__main__":
    main()
