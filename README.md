# RoboPRO

**P**erturbation-**R**esilient **O**bstacle-awareness — a bimanual manipulation benchmark for policy robustness evaluation.

**Project page:** https://anonymous.4open.science/w/RoboPRO-EDE0/index.html

RoboPRO extends the RoboTwin simulation framework with:
- **Realistic scenes** across office, study, kitchen (small & large) domains
- **Systematic perturbation suite** — Language, Vision, and Object axes for evaluating policy robustness
- **Aloha-Agilex** bimanual embodiment with CuRobo motion planning

## Installation

```bash
git clone https://anonymous.4open.science/r/RoboPRO-EDE0
cd RoboPRO
```

Follow the RoboTwin install guide: https://robotwin-platform.github.io/doc/usage/robotwin-install.html for the simulator and policy dependencies.

Then fetch the asset bundle (~15 GB, not tracked in git) from HuggingFace:

```bash
python scripts/install/download_assets.py
```

This populates `benchmark/assets/` (objects, embodiments, background_texture, backgrounds, files). Two one-time setup notes after the download:

1. Edit absolute paths inside `benchmark/assets/embodiments/aloha-agilex/curobo_left.yml` and `curobo_right.yml` (`urdf_path`, `collision_spheres`) so they match the absolute path of your local `benchmark/assets/embodiments/aloha-agilex/` checkout.
2. Fetch the large `box2_Link.dae` mesh (not tracked here — see `docs/install.html`, Step 4).

### CuRobo cache patch

In `customized_robotwin/envs/curobo/src/curobo/geom/sdf/world_mesh.py`, replace `clear_cache` with:

```python
def clear_cache(self):
    self._wp_mesh_cache = {}
    if self._mesh_tensor_list is not None:
        self._mesh_tensor_list[2][:] = 0
    if self._env_n_mesh is not None:
        self._env_n_mesh[:] = 0
    if self._env_mesh_names is not None:
        for i in range(self.n_envs):
            for j in range(len(self._env_mesh_names)):
                self._env_mesh_names[i][j] = None
    super().clear_cache()
```

## Usage

All commands run from `customized_robotwin/` with the bench env exported:

```bash
cd customized_robotwin
source set_env.sh                  # exports BENCH_ROOT + ROBOTWIN_ROOT
export ROBOTWIN_BENCH_TASK=bench   # routes loaders to BENCH_ROOT/{bench_task_config, bench_envs}
```

### Collect demonstrations

```bash
bash collect_data.sh <task_name> <task_config> <gpu_id>
# Example:
bash collect_data.sh put_mouse_on_pad bench_demo_office_clean 0
```

Episodes land in `customized_robotwin/data/<task_name>/<task_config>/`.

### Run inference (policy eval)

Eval rolls a trained checkpoint out against a `(task, config)` pair and writes a per-rollout success log. Two modes depending on whether your policy fits in the same Python env as the simulator.

**Args (shared by both modes):**

| Arg | Meaning |
|---|---|
| `task_name` | Bench env class, e.g. `put_mouse_on_pad` (file at `benchmark/bench_envs/<scene>/<task>.py`) |
| `task_config` | Perturbation YAML name, e.g. `bench_demo_office_clean` (in `benchmark/bench_task_config/`) |
| `train_config_name` | Training config used to fine-tune the checkpoint |
| `model_name` | Subdir name under `checkpoints/<train_config_name>/` |
| `checkpoint_id` | Step number, e.g. `30000` |
| `seed` | RNG seed for episode initialisation |
| `gpu_id` | CUDA device, or `<server_gpu>:<client_gpu>` for dual-env |

**Mode A — single-process** (policy + sim share one Python env, e.g. when openpi is conda-installable alongside SAPIEN):

```bash
bash policy/pi05/eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id>
# Example:
bash policy/pi05/eval.sh put_mouse_on_pad bench_demo_office_clean my_office_train pi05_ckpt 0 0
```

**Mode B — dual-env / dual-process** (recommended for pi05 since openpi+jax need an isolated uv venv at `policy/pi05/.venv/`):

```bash
bash policy/pi05/eval_double_env.sh <task_name> <task_config> <train_config_name> <model_name> <checkpoint_id> <seed> <gpu_spec>
# Example (single GPU):
bash policy/pi05/eval_double_env.sh put_mouse_on_pad bench_demo_office_clean my_office_train pi05_ckpt 30000 0 0
# Example (split: server on GPU 0, sim client on GPU 1):
bash policy/pi05/eval_double_env.sh put_mouse_on_pad bench_demo_office_clean my_office_train pi05_ckpt 30000 0 0:1
```

The script spawns a `policy_model_server.py` in the pi05 venv and an `eval_policy_client.py` in the RoboTwin conda env, communicating over a free socket port.

**Direct Python invocation** (bypassing the shell wrappers):

```bash
python script/eval_policy.py \
    --config policy/pi05/deploy_policy.yml \
    --overrides \
    --task_name put_mouse_on_pad \
    --task_config bench_demo_office_clean \
    --train_config_name my_office_train \
    --model_name pi05_ckpt \
    --checkpoint_id 30000 \
    --ckpt_setting "my_office_train_pi05_ckpt_30000" \
    --policy_name pi05 \
    --seed 0 \
    --instruction_type seen \
    --test_num 10
```

**Where results land:**

```
customized_robotwin/eval_result/bench_eval_result/<task_name>/<policy_name>/<task_config>/<ckpt_setting>/<timestamp>/
    _result.txt        # success count, per-seed pass/fail
    *.mp4              # rollout videos (if eval_video_save is enabled)
```

### Batch eval on SLURM

For sweeping many `(task, config)` pairs across nodes:

```bash
sbatch scripts/slurm/slurm_eval_bench.sh \
    <task_name> <task_config> <train_config_name> <model_name> <checkpoint_id> <seed> <test_num>
```

Set `--chdir` and `--output` in the sbatch header to your local checkout (see comments at the top of `scripts/slurm/slurm_eval_bench.sh`). Pin a specific Python with `export PI05_PYTHON=/path/to/miniconda3/envs/pi05/bin/python`.

For arrayed sweeps over a `(tasks × configs)` grid, see `customized_robotwin/robotwin_*.sbatch` for templates.

## Perturbation configs

Drop-in YAMLs under `benchmark/bench_task_config/`:

| Config | What it perturbs |
|---|---|
| `bench_demo_*_clean.yml` | Baseline (no perturbation) |
| `bench_demo_language.yml` | Per-episode instruction sampled from `instruction_bank.json` |
| `bench_demo_vision.yml` | Lighting (L1–L4) + blur (cycle 5 types) + per-frame pixel shake |
| `bench_demo_vision_lighting.yml` | Lighting only |
| `bench_demo_vision_blur.yml` | Blur only |
| `bench_demo_vision_shake.yml` | Pixel shake only |
| `bench_demo_object.yml` | Target texture swap + unseen obstacles + background_plus |

See the YAMLs in `benchmark/bench_task_config/` for parameter-level details, and `docs/videos/` (`vision*.mp4`, `language.mp4`, `object.mp4`) for sample outputs.

## Scenes and tasks

| Scene | Tasks |
|---|---|
| Office | `put_mouse_on_pad`, `put_phone_on_holder`, `put_book_on_book`, `put_book_in_fileholder`, `put_milktea_on_shelf`, `put_stapler_in_drawer`, `open_drawer`, `close_drawer`, ... |
| Study | `put_book_on_stand`, `put_pen_in_cup`, ... |
| Kitchen (Small) | `put_dish_in_rack`, `place_in_sink`, ... |
| Kitchen (Large) | `microwave_heat`, `fridge_store`, ... |

Full list in `benchmark/bench_envs/`.

## New tasks

1. Write the task env under `benchmark/bench_envs/<scene>/<task>.py`.
2. Add `_eval_step_lim.yml` entry under `benchmark/bench_task_config/`.
3. Add a description template under `benchmark/bench_description/task_instructions/`.

Naming tip: never reuse an existing RoboTwin task name. Start from an analogous sibling task (`kitchenl/`, `office/`, `study/`) — copying a proven recipe is faster than inventing from scratch.

## License

See `LICENSE`.
