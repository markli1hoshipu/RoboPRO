from bench_envs._kitchen_base_large import Kitchen_base_large
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d


class pick_can_from_basket(Kitchen_base_large):
    CAN_MASS = 0.1
    CAN_MODELNAME = "071_can"
    CAN_MODEL_IDS = [0]

    CAN_SPAWN_Z_OFFSET = 0.02
    BASKET_CAN_LOCAL = np.array([0.0, 0.035, 0.03], dtype=float)

    BASKET_X_BOUNDS = (-0.20, 0.20)
    BASKET_Y_BOUNDS = (-0.20, 0.20)
    BASKET_Z_BOUNDS = (-0.10, 0.25)

    PLACE_WORLD_X_OFFSET = 0.08
    PLACE_WORLD_Y_OFFSET = -0.08
    PLACE_SUCCESS_X_TOL = 0.2
    PLACE_SUCCESS_Y_TOL = 0.2
    TABLE_SURFACE_Z_BOUNDS = (-0.08, 0.35)
    IN_HAND_TCP_DIST_THRESHOLD = 0.18

    GRASP_PRE_DIS = 0.06
    GRASP_DIS = 0.0
    GRASP_CLOSE_POS = 0.0
    GRASP_CONTACT_POINT_ID = 0
    PLACE_HEIGHT_ABOVE_TABLE = 0.14
    DESCEND_BEFORE_RELEASE = 0.1
    RETREAT_AFTER_RELEASE = dict(z=0.12, y=-0.12)

    @staticmethod
    def _world_point_in_entity_local(entity, world_xyz: np.ndarray) -> np.ndarray:
        inv_tf = np.linalg.inv(entity.get_pose().to_transformation_matrix())
        h = inv_tf @ np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0], dtype=float)
        return np.array(h[:3], dtype=float)

    @staticmethod
    def _behind_side_can_contact_points(y_center: float) -> list:
        return [
            [[6.123233995736766e-17, -1.0, 0.0, 0.0], [1.0, 6.123233995736766e-17, 0.0, y_center], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ]

    def _ensure_can_grasp_metadata(self) -> None:
        if self.can is None or not isinstance(self.can.config, dict):
            return
        cfg = self.can.config
        y_center = float((cfg.get("center") or [0.0, 0.0, 0.0])[1])
        cfg["contact_points_pose"] = self._behind_side_can_contact_points(y_center)
        cfg["contact_points_group"] = [[0]]
        cfg["contact_points_mask"] = [True]

    def setup_demo(self, is_test: bool = False, **kwargs):
        self.can_modelname = self.CAN_MODELNAME
        self.can_model_ids = list(self.CAN_MODEL_IDS)
        self.can_spawn_rot_deg = [90.0, -90.0, 90.0]

        rot_cfg = kwargs.pop("can_spawn_rot_deg", None)
        if rot_cfg is not None:
            self.can_spawn_rot_deg = [float(rot_cfg[0]), float(rot_cfg[1]), float(rot_cfg[2])]

        self.can_scale = 0.7
        cs = kwargs.pop("can_scale", None)
        if cs is not None:
            self.can_scale = float(cs)

        mids = kwargs.pop("can_model_ids", None)
        if mids is not None:
            self.can_model_ids = [int(x) for x in mids]

        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _can_quat_from_cfg(self) -> list[float]:
        roll_deg, pitch_deg, yaw_deg = self.can_spawn_rot_deg
        ax = math.radians(roll_deg)
        ay = math.radians(pitch_deg)
        az = math.radians(yaw_deg)
        qx, qy, qz, qw = t3d.euler.euler2quat(ax, ay, az)
        return [qw, qx, qy, qz]

    def _table_spawn_pose(self, table_center: np.ndarray) -> sapien.Pose:
        z = float(table_center[2] + self.CAN_SPAWN_Z_OFFSET)
        return sapien.Pose([float(table_center[0]), float(table_center[1]), z], self._can_quat_from_cfg())

    def _basket_spawn_pose(self) -> sapien.Pose:
        basket_pose = self.basket_right.get_pose()
        basket_tf = basket_pose.to_transformation_matrix()
        basket_R = np.array(basket_tf[:3, :3], dtype=float)
        basket_p = np.array(basket_tf[:3, 3], dtype=float)
        world_pos = basket_p + basket_R @ np.array(self.BASKET_CAN_LOCAL, dtype=float)
        return sapien.Pose(world_pos.tolist(), self._can_quat_from_cfg())

    def _can_local_in_basket(self) -> np.ndarray | None:
        if self.can is None or self.basket_right is None:
            return None
        return self._world_point_in_entity_local(self.basket_right, np.array(self.can.get_pose().p, dtype=float))

    def _is_can_inside_basket(self) -> bool:
        loc = self._can_local_in_basket()
        if loc is None:
            return False
        x_l, y_l, z_l = loc
        return bool(
            self.BASKET_X_BOUNDS[0] <= x_l <= self.BASKET_X_BOUNDS[1]
            and self.BASKET_Y_BOUNDS[0] <= y_l <= self.BASKET_Y_BOUNDS[1]
            and self.BASKET_Z_BOUNDS[0] <= z_l <= self.BASKET_Z_BOUNDS[1]
        )

    def _can_local_in_table(self) -> np.ndarray | None:
        if self.can is None or self.table is None:
            return None
        return self._world_point_in_entity_local(self.table, np.array(self.can.get_pose().p, dtype=float))

    def _place_target_world_xy(self) -> np.ndarray:
        p = np.array(self.table.get_pose().p, dtype=float)
        return np.array([p[0] + self._place_world_x_off, p[1] + self._place_world_y_off], dtype=float)

    def _place_anchor_table_local(self) -> np.ndarray | None:
        if self.table is None:
            return None
        xy_w = self._place_target_world_xy()
        p = np.array(self.table.get_pose().p, dtype=float)
        return self._world_point_in_entity_local(self.table, np.array([xy_w[0], xy_w[1], p[2]], dtype=float))

    def _is_can_near_place_xy(self) -> bool:
        can_local = self._can_local_in_table()
        anchor = self._place_anchor_table_local()
        if can_local is None or anchor is None:
            return False
        return bool(
            abs(can_local[0] - anchor[0]) <= float(self.PLACE_SUCCESS_X_TOL)
            and abs(can_local[1] - anchor[1]) <= float(self.PLACE_SUCCESS_Y_TOL)
        )

    def _is_can_on_table_surface(self) -> bool:
        can_local = self._can_local_in_table()
        if can_local is None:
            return False
        z_l = can_local[2]
        return bool(self.TABLE_SURFACE_Z_BOUNDS[0] <= z_l <= self.TABLE_SURFACE_Z_BOUNDS[1])

    def _left_tcp_to_can_dist_m(self) -> float | None:
        if self.can is None:
            return None
        tcp = np.array(self.get_arm_pose(ArmTag("left")), dtype=float)
        can_p = np.array(self.can.get_pose().p, dtype=float)
        return float(np.linalg.norm(can_p - tcp[:3]))

    def _is_can_in_left_hand(self) -> bool:
        d = self._left_tcp_to_can_dist_m()
        if d is None:
            return False
        return bool(d < float(self.IN_HAND_TCP_DIST_THRESHOLD) and self.is_left_gripper_close())

    def _success_debug(self) -> dict:
        can_local = self._can_local_in_table()
        anchor = self._place_anchor_table_local()
        dx = dy = None
        if can_local is not None and anchor is not None:
            dx = float(abs(can_local[0] - anchor[0]))
            dy = float(abs(can_local[1] - anchor[1]))
        z_l = float(can_local[2]) if can_local is not None else None
        on_table = self._is_can_on_table_surface()
        not_in_basket = not self._is_can_inside_basket()
        not_in_left = not self._is_can_in_left_hand()
        near_xy = self._is_can_near_place_xy()
        ok = bool(on_table and not_in_basket and not_in_left and near_xy)
        return {
            "on_table": on_table,
            "not_in_basket": not_in_basket,
            "not_in_left_hand": not_in_left,
            "near_place_xy": near_xy,
            "table_local_xy_abs_err": [dx, dy],
            "place_xy_tol": [float(self.PLACE_SUCCESS_X_TOL), float(self.PLACE_SUCCESS_Y_TOL)],
            "table_local_z": z_l,
            "table_surface_z_bounds": [self.TABLE_SURFACE_Z_BOUNDS[0], self.TABLE_SURFACE_Z_BOUNDS[1]],
            "tcp_dist_m": self._left_tcp_to_can_dist_m(),
            "in_hand_tcp_threshold_m": float(self.IN_HAND_TCP_DIST_THRESHOLD),
            "left_gripper_close": bool(self.is_left_gripper_close()),
            "plan_success": getattr(self, "plan_success", None),
            "all_ok": ok,
        }

    def _ee_pose_above_place_target(self, arm_tag: ArmTag) -> np.ndarray:
        ee_pose = np.array(self.get_arm_pose(arm_tag), dtype=float)
        table_p = np.array(self.table.get_pose().p, dtype=float)
        xy_w = self._place_target_world_xy()
        target = ee_pose.copy()
        target[0] = float(xy_w[0])
        target[1] = float(xy_w[1])
        target[2] = float(table_p[2] + self.PLACE_HEIGHT_ABOVE_TABLE)
        return target

    def load_actors(self):
        self._place_world_x_off = float(self.PLACE_WORLD_X_OFFSET)
        self._place_world_y_off = float(self.PLACE_WORLD_Y_OFFSET)

        self.can_model_id = int(np.random.choice(self.can_model_ids))
        spawn_pose = self._basket_spawn_pose()

        intrinsic_scale = self._get_asset_model_scale_create_actor(self.can_modelname, self.can_model_id)
        final_scale = float(intrinsic_scale) * float(self.can_scale)

        self.can = create_actor(
            scene=self.scene,
            pose=spawn_pose,
            modelname=self.can_modelname,
            model_id=self.can_model_id,
            is_static=False,
            convex=True,
            scale=final_scale,
        )
        if self.can is not None:
            self.can.set_mass(self.CAN_MASS)
            self.can.set_name("task_can")
            if isinstance(self.can.config, dict):
                self.can.config["scale"] = [final_scale] * 3
            self._ensure_can_grasp_metadata()
            self.add_prohibit_area(self.can, padding=0.04, area="table")

    def play_once(self):
        arm_tag = ArmTag("left")
        self.move(
            self.grasp_actor(
                self.can,
                arm_tag=arm_tag,
                pre_grasp_dis=self.GRASP_PRE_DIS,
                grasp_dis=self.GRASP_DIS,
                gripper_pos=self.GRASP_CLOSE_POS,
                contact_point_id=self.GRASP_CONTACT_POINT_ID,
            )
        )
        self.move(self.back_to_origin(arm_tag=arm_tag))
        self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=self._ee_pose_above_place_target(arm_tag)))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=-self.DESCEND_BEFORE_RELEASE))
        self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, **self.RETREAT_AFTER_RELEASE))
        self.move(self.back_to_origin(arm_tag=arm_tag))

        self.info["info"] = {
            "{A}": f"{self.can_modelname}/base{self.can_model_id}",
            "{a}": str(arm_tag),
        }
        chk = self.check_success()
        self.info["episode_check_success"] = chk
        self.info["success_debug"] = dict(self._last_success_debug)
        return self.info

    def check_success(self):
        d = self._success_debug()
        self._last_success_debug = d
        return bool(d["all_ok"])
