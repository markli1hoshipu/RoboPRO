import os

import yaml

from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from bench_envs.utils.scene_gen_utils import get_actor_boundingbox_urdf, print_c
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d


class put_bottle_in_fridge(Kitchen_base_large):
    BOTTLE_MASS = 0.1
    BOTTLE_SPAWN_Z_OFFSET = 0.02

    # Placement target in fridge base-link local frame
    FRIDGE_PLACE_LOCAL = np.array([-0.10, 0.00, 0.05], dtype=float)

    def setup_demo(self, is_test: bool = False, **kwargs):
        self.bottle_modelname = "001_bottle"
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r") as f:
            task_objs = yaml.safe_load(f)

        self.bottle_model_ids =  task_objs['objects']['kitchenl']['targets'][self.bottle_modelname]
        self.bottle_spawn_local_x_range = (-0.22, 0.12)
        self.bottle_spawn_local_y_range = (-0.32, -0.06)
        # Same convention as `basket_right_rot` in `_kitchen_base_large`: degrees (roll, pitch, yaw).
        self.bottle_spawn_rot_deg = [0.0, 0.0, 90.0]
        rot_cfg = kwargs.pop("bottle_spawn_rot_deg", None)
        if rot_cfg is not None:
            self.bottle_spawn_rot_deg = [float(rot_cfg[0]), float(rot_cfg[1]), float(rot_cfg[2])]
        # Uniform scale multiplier on `model_data` intrinsic (cf. `basket_right_scale` × intrinsic).
        self.bottle_scale = 0.7
        bs = kwargs.pop("bottle_scale", None)
        if bs is not None:
            self.bottle_scale = float(bs)

        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

        if getattr(self, "microwave_left", None) is not None:
            try:
                mw_qpos = np.array(self.microwave_left.get_qpos(), dtype=float)
                if mw_qpos.size > 0:
                    mw_qpos[:] = 0.0
                    self.microwave_left.set_qpos(mw_qpos)
            except Exception:
                pass

        self.set_fridge_open()

    def _sample_bottle_spawn_pose(self, table_center: np.ndarray) -> sapien.Pose:
        # Keep current spawn behavior exactly as implemented.
        x = float(np.random.uniform(table_center[0] - 0.2, table_center[0]))
        y = float(np.random.uniform(table_center[1] - 0.1, table_center[1] + 0.1))
        z = float(table_center[2] + self.BOTTLE_SPAWN_Z_OFFSET)

        roll_deg, pitch_deg, yaw_deg = self.bottle_spawn_rot_deg
        ax = math.radians(roll_deg)
        ay = math.radians(pitch_deg)
        az = math.radians(yaw_deg)
        qx, qy, qz, qw = t3d.euler.euler2quat(ax, ay, az)
        bottle_quat = [qw, qx, qy, qz]
        return sapien.Pose([x, y, z], bottle_quat)

    def load_actors(self):
        if getattr(self, "fridge_closed_qpos", None) is None:
            self._init_fridge_states()

        self.set_fridge_open()            

        table_center = np.array(self.table.get_pose().p, dtype=float)
        self.bottle_model_id = int(np.random.choice(self.bottle_model_ids))
        bottle_pose = self._sample_bottle_spawn_pose(table_center)

        intrinsic_scale = self._get_asset_model_scale_create_actor(
            self.bottle_modelname, self.bottle_model_id
        )
        final_scale = float(intrinsic_scale) * float(self.bottle_scale)

        self.bottle = create_actor(
            scene=self.scene,
            pose=bottle_pose,
            modelname=self.bottle_modelname,
            model_id=self.bottle_model_id,
            is_static=False,
            convex=True,
            scale=final_scale,
        )

        if self.bottle is not None:
            self.bottle.set_mass(self.BOTTLE_MASS)

            self.bottle.set_name("task_bottle")
            if isinstance(self.bottle.config, dict):
                self.bottle.config["scale"] = [final_scale] * 3
            self.add_prohibit_area(self.bottle, padding=0.04, area="table")

    def _fridge_inside_target_pose(self) -> list[float]:
        base_pose = self.fridge_left.get_link_pose("base_link")
        base_tf = base_pose.to_transformation_matrix()
        base_R = np.array(base_tf[:3, :3], dtype=float)
        base_p = np.array(base_tf[:3, 3], dtype=float)
        world_inside = base_p + base_R @ self.FRIDGE_PLACE_LOCAL
        bottle_q = self.bottle.get_pose().q.tolist()
        return world_inside.tolist() + bottle_q

    def _is_bottle_inside_fridge(self) -> bool:
        if self.bottle is None or self.fridge_left is None:
            return False
        bottle_world = np.array(self.bottle.get_pose().p, dtype=float)
        base_pose = self.fridge_left.get_link_pose("base_link")
        base_tf = base_pose.to_transformation_matrix()
        inv_tf = np.linalg.inv(base_tf)
        bottle_local_h = inv_tf @ np.array([bottle_world[0], bottle_world[1], bottle_world[2], 1.0], dtype=float)
        x_l, y_l, z_l = bottle_local_h[:3]
        x_ok = (self.FRIDGE_SUCCESS_X_BOUNDS[0] <= x_l <= self.FRIDGE_SUCCESS_X_BOUNDS[1])
        y_ok = (self.FRIDGE_SUCCESS_Y_BOUNDS[0] <= y_l <= self.FRIDGE_SUCCESS_Y_BOUNDS[1])
        z_ok = (self.FRIDGE_SUCCESS_Z_BOUNDS[0] <= z_l <= self.FRIDGE_SUCCESS_Z_BOUNDS[1])
        return bool(x_ok and y_ok and z_ok)

    def play_once(self):
        arm_tag = ArmTag("right")
        # Close slightly less than "fully closed" to reduce penetration impulse.
        self.move(
            self.grasp_actor(
                self.bottle,
                arm_tag=arm_tag,
                pre_grasp_dis=0.07,
                grasp_dis=0.0,
                gripper_pos=0.0,
            )
        )
        self.attach_object(self.bottle, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{self.bottle_modelname}/collision/base{self.bottle_model_id}.glb", str(arm_tag))

        self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.1, z=0.05))
        self.move(self.back_to_origin(arm_tag=arm_tag))

        self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.1, y=0.30, z=-0.03))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=0.1))
        self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))


        self.info["info"] = {
            "{A}": f"{self.bottle_modelname}/base{self.bottle_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        return self._is_bottle_inside_fridge()
