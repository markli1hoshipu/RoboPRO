from bench_envs._kitchen_base_large import Kitchen_base_large
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d


class put_bottle_in_fridge(Kitchen_base_large):
    def setup_demo(self, is_test: bool = False, **kwargs):
        self.bottle_modelname = "001_bottle"
        self.bottle_model_ids = [1, 11, 14, 16]
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

    def check_stable(self):
        """Dynamic bottle can wobble while settling; ignore init instability for `task_bottle`."""
        is_stable, unstable_list = super().check_stable()
        unstable_list = [n for n in unstable_list if n != "task_bottle"]
        return len(unstable_list) == 0, unstable_list

    def load_actors(self):
        if getattr(self, "fridge_closed_qpos", None) is None:
            self._init_fridge_states()
        self.set_fridge_open()

        table_center = np.array(self.table.get_pose().p, dtype=float)
        self.bottle_model_id = int(np.random.choice(self.bottle_model_ids))
        # Randomize initial bottle table center with fixed offsets:
        #   x in [table_center.x - 0.2, table_center.x + 0.2]
        #   y in [table_center.y - 0.3, table_center.y + 0.1]
        x = float(np.random.uniform(table_center[0] - 0.2, table_center[0] + 0.2))
        y = float(np.random.uniform(table_center[1] - 0.3, table_center[1] + 0.1))
        z = float(table_center[2] + 0.02)

        intrinsic_scale = self._get_asset_model_scale_create_actor(
            self.bottle_modelname, self.bottle_model_id
        )
        final_scale = float(intrinsic_scale) * float(self.bottle_scale)

        roll_deg, pitch_deg, yaw_deg = self.bottle_spawn_rot_deg
        ax = math.radians(roll_deg)
        ay = math.radians(pitch_deg)
        az = math.radians(yaw_deg)
        qx, qy, qz, qw = t3d.euler.euler2quat(ax, ay, az)
        bottle_quat = [qw, qx, qy, qz]

        bottle_pose = sapien.Pose([x, y, z], bottle_quat)
        bottle_mass = 0.1

        self.bottle = create_actor(
            scene=self.scene,
            pose=bottle_pose,
            modelname=self.bottle_modelname,
            model_id=self.bottle_model_id,
            is_static=False,
            # Match `create_actor_custom.create_glb_actor` default: nonconvex collision
            # (often reduces micro-wobble for thin dynamic objects during settle).
            convex=False,
            scale=final_scale,
        )

        if self.bottle is not None:
            # Match `scene_gen_utils.place_actor` default mass override.
            self.bottle.set_mass(bottle_mass)

            self.bottle.set_name("task_bottle")
            if isinstance(self.bottle.config, dict):
                self.bottle.config["scale"] = [final_scale] * 3
            self.add_prohibit_area(self.bottle, padding=0.04, area="table")

    def _fridge_inside_target_pose(self) -> list[float]:
        base_pose = self.fridge_left.get_link_pose("base_link")
        base_tf = base_pose.to_transformation_matrix()
        base_R = np.array(base_tf[:3, :3], dtype=float)
        base_p = np.array(base_tf[:3, 3], dtype=float)
        local_inside = np.array([-0.10, 0.00, 0.05], dtype=float)
        world_inside = base_p + base_R @ local_inside
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
        x_ok = (-0.33 <= x_l <= 0.16)
        y_ok = (-0.22 <= y_l <= 0.22)
        z_ok = (-0.34 <= z_l <= 0.24)
        return bool(x_ok and y_ok and z_ok)

    def play_once(self):
        arm_tag = ArmTag("right")
        self.move(self.grasp_actor(self.bottle, arm_tag=arm_tag, pre_grasp_dis=0.10, grasp_dis=0.0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12))
        place_pose = self._fridge_inside_target_pose()
        self.move(
            self.place_actor(
                self.bottle,
                arm_tag=arm_tag,
                target_pose=place_pose,
                constrain="auto",
                pre_dis=0.10,
                dis=0.03,
            )
        )
        self.info["info"] = {
            "{A}": f"{self.bottle_modelname}/base{self.bottle_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        return self._is_bottle_inside_fridge()
