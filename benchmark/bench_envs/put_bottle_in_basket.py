from bench_envs._kitchen_base_large import Kitchen_base_large
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d


class put_bottle_in_basket(Kitchen_base_large):
    BOTTLE_MASS = 0.1
    BOTTLE_SPAWN_Z_OFFSET = 0.01
    BOTTLE_WORLD_XY_JITTER = 0.025

    # Basket interior bounds in basket local frame.
    BASKET_X_BOUNDS = (-0.20, 0.20)
    BASKET_Y_BOUNDS = (-0.20, 0.20)
    BASKET_Z_BOUNDS = (-0.10, 0.25)

    # Left-arm motion tuning.
    APPROACH_DELTA_1 = dict(x=-0.05, y=0.285, z=0.10)
    APPROACH_DELTA_2 = dict(z=-0.10)
    RETREAT_DELTA = dict(y=-0.20)
    GRASP_PRE_DIS = 0.07
    GRASP_DIS = 0.0
    GRASP_CLOSE_POS = 0.0
    GRASP_CONTACT_POINT_ID = 1

    def setup_demo(self, is_test: bool = False, **kwargs):
        # Match bottle asset setup used in pick_bottle_from_fridge.
        self.bottle_modelname = "001_bottle"
        self.bottle_model_ids = [1, 11, 14, 16]
        self.bottle_spawn_rot_deg = [-45.0, 0.0, 90.0]

        rot_cfg = kwargs.pop("bottle_spawn_rot_deg", None)
        if rot_cfg is not None:
            self.bottle_spawn_rot_deg = [float(rot_cfg[0]), float(rot_cfg[1]), float(rot_cfg[2])]

        self.bottle_scale = 0.7
        bs = kwargs.pop("bottle_scale", None)
        if bs is not None:
            self.bottle_scale = float(bs)

        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _bottle_quat_from_cfg(self) -> list[float]:
        roll_deg, pitch_deg, yaw_deg = self.bottle_spawn_rot_deg
        ax = math.radians(roll_deg)
        ay = math.radians(pitch_deg)
        az = math.radians(yaw_deg)
        qx, qy, qz, qw = t3d.euler.euler2quat(ax, ay, az)
        return [qw, qx, qy, qz]

    def _table_center_spawn_pose(self, table_center: np.ndarray) -> sapien.Pose:
        # Spawn near table center with small world-frame jitter.
        x = float(np.random.uniform(table_center[0] - 0.1, table_center[0] + 0.05))
        y = float(np.random.uniform(table_center[1] - 0.1, table_center[1] + 0.05))
        z = float(table_center[2] + self.BOTTLE_SPAWN_Z_OFFSET)
        return sapien.Pose([x, y, z], self._bottle_quat_from_cfg())

    def load_actors(self):
        table_center = np.array(self.table.get_pose().p, dtype=float)
        self.bottle_model_id = int(np.random.choice(self.bottle_model_ids))
        spawn_pose = self._table_center_spawn_pose(table_center)

        intrinsic_scale = self._get_asset_model_scale_create_actor(self.bottle_modelname, self.bottle_model_id)
        final_scale = float(intrinsic_scale) * float(self.bottle_scale)

        self.bottle = create_actor(
            scene=self.scene,
            pose=spawn_pose,
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

    def _bottle_local_in_basket(self) -> np.ndarray | None:
        if self.bottle is None or self.basket_right is None:
            return None
        bottle_world = np.array(self.bottle.get_pose().p, dtype=float)
        basket_pose = self.basket_right.get_pose()
        inv_tf = np.linalg.inv(basket_pose.to_transformation_matrix())
        bottle_local_h = inv_tf @ np.array([bottle_world[0], bottle_world[1], bottle_world[2], 1.0], dtype=float)
        return np.array(bottle_local_h[:3], dtype=float)

    def _is_bottle_inside_basket(self) -> bool:
        bottle_local = self._bottle_local_in_basket()
        if bottle_local is None:
            return False
        x_l, y_l, z_l = bottle_local
        x_ok = (self.BASKET_X_BOUNDS[0] <= x_l <= self.BASKET_X_BOUNDS[1])
        y_ok = (self.BASKET_Y_BOUNDS[0] <= y_l <= self.BASKET_Y_BOUNDS[1])
        z_ok = (self.BASKET_Z_BOUNDS[0] <= z_l <= self.BASKET_Z_BOUNDS[1])
        return bool(x_ok and y_ok and z_ok)

    def play_once(self):
        arm_tag = ArmTag("left")
        self.move(
            self.grasp_actor(
                self.bottle,
                arm_tag=arm_tag,
                pre_grasp_dis=self.GRASP_PRE_DIS,
                grasp_dis=self.GRASP_DIS,
                gripper_pos=self.GRASP_CLOSE_POS,
                contact_point_id=self.GRASP_CONTACT_POINT_ID,
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15))
        self.move(self.back_to_origin(arm_tag=arm_tag))

        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.APPROACH_DELTA_1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.APPROACH_DELTA_2))
        self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.RETREAT_DELTA))
        self.move(self.back_to_origin(arm_tag=arm_tag))

        self.info["info"] = {
            "{A}": f"{self.bottle_modelname}/base{self.bottle_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        return self._is_bottle_inside_basket()
