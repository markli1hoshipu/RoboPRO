# Collision Metrics Analysis — RoboTwin Bench

## Overview

The system tracks **6 counters** for unintended contact events during task execution.
Implemented in `bench_envs/_bench_base_task.py`, enabled per-task via `enable_collision_metrics` kwarg.

---

## Metrics Tracked

| Metric | Description |
|--------|-------------|
| `robot_to_furniture` | Contact-point count: robot link vs. furniture |
| `robot_to_static_object` | Contact-point count: robot link vs. static object |
| `target_to_static_object` | Contact-point count: target object vs. static object |
| `robot_to_furniture_steps` | Steps where any robot-to-furniture contact occurred |
| `robot_to_static_object_steps` | Steps where any robot-to-static contact occurred |
| `target_to_static_object_steps` | Steps where any target-to-static contact occurred |

Also maintained for debug: `filtered_contacts_for_log` — list of `{body0, body1, impulse, position}` dicts.

---

## Object Classification

The scene is partitioned into 4 non-overlapping name sets, built by `_build_collision_name_sets()`:

### `robot_link_names`
All articulation links from left and right robot arms.

### `furniture_names` (hard-coded per scene subclass)
| Scene | Names |
|-------|-------|
| Office (`_office_base_task.py:42`) | `{table, wall, shelf, ground}` |
| Study (`_study_base_task.py:45`) | `{table, wall, floor, ground, 014_bookcase, 042_wooden_box}` |
| Kitchen Small (`_kitchens_base_task.py:42`) | `{table, wall, ground}` |
| Kitchen Large / base | `{table, wall, ground}` |

### `target_object_names`
Returned by each task's `_get_target_object_names()` override. Base class raises `NotImplementedError`.

### `static_object_names`
All named actors **not** in furniture, robot links, or target objects. These are clutter/obstacle objects.

---

## Detection Rules (`check_collisions()`, `_bench_base_task.py:503`)

### Physics Constants
```
COLLISION_FORCE_THRESHOLD_N        = 10.0 N
collision_impulse_threshold        = max(10.0 * timestep, 1e-6)   # computed in setup_scene()
STATIC_OBJECT_POSITION_THRESHOLD_M = 0.01 m  (1 cm)
STATIC_OBJECT_ORIENTATION_THRESHOLD_RAD = 0.1 rad (~5.7°)
GRIPPER_LINK_NAMES = {fr_link7, fr_link8, fl_link7, fl_link8}
```

---

### 1. `robot_to_furniture` — Robot Link vs. Furniture

**Counted per contact-point when ALL hold:**
1. One body ∈ `robot_link_names`, other ∈ `furniture_names`
2. The robot body is **not** a gripper link (gripper-furniture contact is excluded as expected)
3. Any contact point has impulse `> collision_impulse_threshold`

**Logged** if contact impulse `> collision_impulse_threshold`.

---

### 2. `robot_to_static_object` — Robot Link vs. Static Object

**Counted per contact-point when ALL hold:**
1. One body ∈ `robot_link_names`, other ∈ `static_object_names`
2. The robot body is **not** a gripper link
3. The static object has **significant pose change** from the previous step:
   - position delta `> 0.01 m`, OR
   - orientation delta `> 0.1 rad`

**No impulse threshold** — only pose-change filter applies.

**Logged** if any contact impulse `> 0`.

---

### 3. `target_to_static_object` — Target Object vs. Static Object

**Counted per contact-point when ALL hold:**
1. One body ∈ `target_object_names`, other ∈ `static_object_names`
2. The static object has **significant pose change** from the previous step (same thresholds as above)

**No gripper exclusion, no impulse threshold** — only pose-change filter applies.

**Logged** if any contact impulse `> 0`.

---

### Step-Level Aggregation

After processing all contacts in a step:
- `robot_to_furniture_steps` += 1 if **any** furniture contact was counted
- `robot_to_static_object_steps` += 1 if **any** static contact was counted
- `target_to_static_object_steps` += 1 if **any** target-static contact was counted

Pose history (`static_object_pose_prev`) is updated at end of each step.

---

## Lifecycle

### Initialization
```
_init_task_env_()
  ├─ self.enable_collision_metrics = kwags.get("enable_collision_metrics", False)
  ├─ self.collision_list = []
  ├─ self._init_collision_metrics()        # zeros all 6 counters
  ├─ ... load robot, furniture, all actors ...
  └─ if self.enable_collision_metrics:
         self._build_collision_name_sets() # MUST be after all actors are loaded
```

### Per-Step Check (`take_action()`, line ~1097)
```python
if getattr(self, 'enable_collision_metrics', False) and hasattr(self, 'robot_link_names'):
    self.check_collisions()
```

### Retrieval
```python
metrics = self.get_collision_metrics()  # shallow copy of self.collision_metrics dict
```

---

## Key Files

| File | Method | Lines |
|------|--------|-------|
| `_bench_base_task.py` | `_init_collision_metrics()` | 432–443 |
| `_bench_base_task.py` | `_build_collision_name_sets()` | 453–501 |
| `_bench_base_task.py` | `check_collisions()` | 503–601 |
| `_bench_base_task.py` | `get_collision_metrics()` | 603–605 |
| `_bench_base_task.py` | collision check call in `take_action()` | 1097–1098 |
| `office/_office_base_task.py` | enable flag, init, build call | 78, 187, 209–210 |
| `study/_study_base_task.py` | enable flag, init, build call | 86, 164–165 |
| `kitchens/_kitchens_base_task.py` | enable flag, init, build call | 64, 128, 153–154 |

---

## Known Gaps / Items to Verify

- [ ] **Office `FURNITURE_NAMES` missing floor parts** — `floor_0..3` actors exist but `FURNITURE_NAMES = {table, wall, shelf, ground}`. Floor tile collisions are untracked. Confirm intentional.
- [ ] **Asymmetric impulse filter** — `robot_to_furniture` uses impulse threshold; `robot_to_static` and `target_to_static` use only pose-change. Confirm intentional.
- [ ] **Gripper not excluded from `target_to_static`** — If a gripper-held target bumps a static object, it is counted. This seems correct (held-object collisions are meaningful).
- [ ] **`_get_target_object_names()` coverage** — Every task under `bench_envs/*/tasks/` must override this. Audit all task files for compliance.
- [ ] **1-step lag in pose detection** — Pose delta is computed from previous step. An object knocked and settled within 1 step will not register. Assess false-negative rate.
- [ ] **Kitchen Large init pattern** — Verify `_kitchenl_base_task.py` follows the same enable/init/build pattern as other scenes.
