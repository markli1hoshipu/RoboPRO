# benchmark/asset_unsure/

Bench-team asset variants that existed in the old `benchmark/bench_assets/`
location but were **never wired into the runtime loaders**. Recovered here
during the asset consolidation refactor (PR / commit `d97bb67`) so the work
isn't lost while we triage.

| Item | Files | Why preserved |
|---|---|---|
| `044_microwave/` | 316 | Custom `mobility.urdf` referencing 134 collision meshes (CoACD output) — runtime uses the upstream URDF with 16 meshes from `benchmark/assets/objects/044_microwave/` |
| `034_knife/` | 4 | `model_data0.json` had different `contact_points_pose` (likely re-annotated grasp points) |
| `122_file-holder/` | 1 | Bench had a sub-named `122_file_holder/base.glb`; runtime expects `base.glb` at the dir root |
| `121_cabinet_cjcyed/` | 2 | Bench-only articulated, never referenced by any code |
| `123_drawer_dsbcxl/` | 11 | Bench-only articulated, never referenced |
| `126_fridge_hivvdfn/` | 10 | Bench-only articulated, never referenced |
| `120_storage-rack/` | 73 | Was in upstream `assets/objects/` too, but no code references it |

## To use one

If a bench task should actually load one of these, either:
1. **Replace runtime version**: copy the variant into `benchmark/assets/objects/<name>/` (overwrites upstream copy). Re-run `python scripts/install/download_assets.py` would undo this — see (2) for a more durable fix.
2. **Wire up loader**: add explicit handling in the relevant `bench_envs/` file to point at `BENCH_ROOT/asset_unsure/<name>/...`.

## To discard

If after triage you decide an item is truly dead, `rm -rf benchmark/asset_unsure/<name>` and commit.
