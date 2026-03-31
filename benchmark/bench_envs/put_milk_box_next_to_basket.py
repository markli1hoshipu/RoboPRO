from bench_envs._kitchen_base_large import Kitchen_base_large
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d


class put_milk_box_next_to_basket(Kitchen_base_large):
    MILK_BOX_MASS = 0.1
    MILK_BOX_SPAWN_Z_OFFSET = 0.02
    TABLE_WORLD_XY_JITTER = 0.05

    # Basket interior bounds in basket local frame.
    BASKET_X_BOUNDS = (-0.20, 0.20)
    BASKET_Y_BOUNDS = (-0.20, 0.20)
    BASKET_Z_BOUNDS = (-0.10, 0.25)

    # Success region multiplier around basket bounds.
    BASKET_EXPANSION_RATIO = 1

    # Left-arm motion tuning.
    APPROACH_DELTA_1 = dict(x=-0.22, y=0.22, z=-0.14)
    RETREAT_DELTA = dict(y=-0.07, z=0.02)
    GRASP_PRE_DIS = 0.07
    GRASP_DIS = 0.01
    GRASP_CLOSE_POS = 0.0

    @staticmethod
    def _default_milk_box_contact_points(y_center: float) -> list:
        # Fallback contact-point set (4 side grasps) for assets lacking metadata.
        return [
            [[6.123233995736766e-17, -1.0, 0.0, 0.0], [1.0, 6.123233995736766e-17, 0.0, y_center], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[6.123233995736766e-17, -6.123233995736766e-17, 1.0, 0.0], [1.0, 3.749399456654644e-33, -6.123233995736766e-17, y_center], [0.0, 1.0, 6.123233995736766e-17, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[6.123233995736766e-17, 1.0, 1.2246467991473532e-16, 0.0], [1.0, -6.123233995736766e-17, -7.498798913309288e-33, y_center], [0.0, 1.2246467991473532e-16, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[6.123233995736766e-17, -6.123233995736766e-17, -1.0, 0.0], [1.0, 3.749399456654644e-33, 6.123233995736766e-17, y_center], [0.0, -1.0, 6.123233995736766e-17, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ]

    def _ensure_milk_box_grasp_metadata(self) -> None:
        if self.milk_box is None or not isinstance(self.milk_box.config, dict):
            return
        cfg = self.milk_box.config
        if "contact_points_pose" in cfg and isinstance(cfg["contact_points_pose"], list) and len(cfg["contact_points_pose"]) > 0:
            return
        y_center = float((cfg.get("center") or [0.0, 0.0, 0.0])[1])
        cfg["contact_points_pose"] = self._default_milk_box_contact_points(y_center)
        cfg.setdefault("contact_points_group", [list(range(len(cfg["contact_points_pose"])))])
        cfg.setdefault("contact_points_mask", [True])

    def setup_demo(self, is_test: bool = False, **kwargs):
        self.milk_box_modelname = "038_milk-box"
        self.milk_box_model_ids = [0, 1, 2]
        self.milk_box_spawn_rot_deg = [-45.0, 0.0, 90.0]

        rot_cfg = kwargs.pop("milk_box_spawn_rot_deg", None)
        if rot_cfg is not None:
            self.milk_box_spawn_rot_deg = [float(rot_cfg[0]), float(rot_cfg[1]), float(rot_cfg[2])]

        self.milk_box_scale = 0.7
        ms = kwargs.pop("milk_box_scale", None)
        if ms is not None:
            self.milk_box_scale = float(ms)

        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _milk_box_quat_from_cfg(self) -> list[float]:
        roll_deg, pitch_deg, yaw_deg = self.milk_box_spawn_rot_deg
        ax = math.radians(roll_deg)
        ay = math.radians(pitch_deg)
        az = math.radians(yaw_deg)
        qx, qy, qz, qw = t3d.euler.euler2quat(ax, ay, az)
        return [qw, qx, qy, qz]

    def _table_center_spawn_pose(self, table_center: np.ndarray) -> sapien.Pose:
        # Spawn at table center with small world-frame randomization.
        x = float(table_center[0] + np.random.uniform(-self.TABLE_WORLD_XY_JITTER, self.TABLE_WORLD_XY_JITTER))
        y = float(table_center[1] + np.random.uniform(-self.TABLE_WORLD_XY_JITTER, self.TABLE_WORLD_XY_JITTER))
        z = float(table_center[2] + self.MILK_BOX_SPAWN_Z_OFFSET)
        return sapien.Pose([x, y, z], self._milk_box_quat_from_cfg())

    def load_actors(self):
        table_center = np.array(self.table.get_pose().p, dtype=float)
        self.milk_box_model_id = int(np.random.choice(self.milk_box_model_ids))
        spawn_pose = self._table_center_spawn_pose(table_center)

        intrinsic_scale = self._get_asset_model_scale_create_actor(self.milk_box_modelname, self.milk_box_model_id)
        final_scale = float(intrinsic_scale) * float(self.milk_box_scale)

        self.milk_box = create_actor(
            scene=self.scene,
            pose=spawn_pose,
            modelname=self.milk_box_modelname,
            model_id=self.milk_box_model_id,
            is_static=False,  # requested: start as static
            convex=True,
            scale=final_scale,
        )
        if self.milk_box is not None:
            self.milk_box.set_mass(self.MILK_BOX_MASS)
            self.milk_box.set_name("task_milk_box")
            if isinstance(self.milk_box.config, dict):
                self.milk_box.config["scale"] = [final_scale] * 3
            self._ensure_milk_box_grasp_metadata()
            self.add_prohibit_area(self.milk_box, padding=0.04, area="table")

    def _milk_box_local_in_basket(self) -> np.ndarray | None:
        if self.milk_box is None or self.basket_right is None:
            return None
        milk_world = np.array(self.milk_box.get_pose().p, dtype=float)
        basket_pose = self.basket_right.get_pose()
        inv_tf = np.linalg.inv(basket_pose.to_transformation_matrix())
        milk_local_h = inv_tf @ np.array([milk_world[0], milk_world[1], milk_world[2], 1.0], dtype=float)
        return np.array(milk_local_h[:3], dtype=float)

    def _is_milk_box_next_to_basket(self) -> bool:
        milk_local = self._milk_box_local_in_basket()
        if milk_local is None:
            return False
        x_l, y_l, z_l = milk_local
        ratio = float(self.BASKET_EXPANSION_RATIO)
        x_ok = (self.BASKET_X_BOUNDS[0] * ratio <= x_l <= self.BASKET_X_BOUNDS[1] * ratio)
        y_ok = (self.BASKET_Y_BOUNDS[0] * ratio <= y_l <= self.BASKET_Y_BOUNDS[1] * ratio)
        z_ok = (self.BASKET_Z_BOUNDS[0] * ratio <= z_l <= self.BASKET_Z_BOUNDS[1] * ratio)
        return bool(x_ok and y_ok and z_ok)

    def play_once(self):
        arm_tag = ArmTag("left")
        self.move(
            self.grasp_actor(
                self.milk_box,
                arm_tag=arm_tag,
                pre_grasp_dis=self.GRASP_PRE_DIS,
                grasp_dis=self.GRASP_DIS,
                gripper_pos=self.GRASP_CLOSE_POS,
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15))
        self.move(self.back_to_origin(arm_tag=arm_tag))
        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.APPROACH_DELTA_1))
        self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.RETREAT_DELTA))
        self.move(self.back_to_origin(arm_tag=arm_tag))

        self.info["info"] = {
            "{A}": f"{self.milk_box_modelname}/base{self.milk_box_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        return self._is_milk_box_next_to_basket()
