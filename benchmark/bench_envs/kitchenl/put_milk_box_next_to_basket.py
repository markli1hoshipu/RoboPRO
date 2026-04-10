import os
import yaml
import math
import numpy as np
import sapien
import transforms3d as t3d

from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from bench_envs.utils.scene_gen_utils import get_actor_boundingbox, point_to_box_distance, print_c
from envs.utils import *


class put_milk_box_next_to_basket(Kitchen_base_large):

    def setup_demo(self, is_test: bool = False, **kwargs):
        self.milk_box_modelname = "038_milk-box"
        with open(os.path.join(os.environ["BENCH_ROOT"], 'bench_task_config', 'task_objects.yml'), "r") as f:
            task_objs = yaml.safe_load(f)
        self.milk_box_model_ids = task_objs['objects']['kitchenl']['targets'][self.milk_box_modelname]
        kwargs["scene_id"] = np.random.choice([0, 1])
        kwargs["include_collision"] = True
        self.milk_box_spawn_rot_deg = [0.0, 0.0, 90.0]
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        table_center = np.array(self.table.get_pose().p, dtype=float)
        self.milk_box_model_id = int(np.random.choice(self.milk_box_model_ids))

        if self.scene_id == 1:
            xlim = [-0.45, -0.2]
            self.contact_id = 1
        else:
            xlim = [-0.15, 0]
            self.contact_id = None
        x = float(np.random.uniform(xlim[0], xlim[1]))
        y = float(np.random.uniform(-0.1, 0))
        z = float(table_center[2] + 0.02)

        roll_deg, pitch_deg, yaw_deg = self.milk_box_spawn_rot_deg
        qx, qy, qz, qw = t3d.euler.euler2quat(
            math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
        )

        intrinsic_scale = self._get_asset_model_scale_create_actor(self.milk_box_modelname, self.milk_box_model_id)
        final_scale = float(intrinsic_scale) * 0.7

        self.milk_box = create_actor(
            scene=self.scene,
            pose=sapien.Pose([x, y, z], [qw, qx, qy, qz]),
            modelname=self.milk_box_modelname,
            model_id=self.milk_box_model_id,
            is_static=False,
            convex=True,
            scale=final_scale,
        )
        if self.milk_box is not None:
            self.milk_box.set_mass(0.1)
            self.milk_box.set_name("task_milk_box")
            if isinstance(self.milk_box.config, dict):
                cfg = self.milk_box.config
                cfg["scale"] = [final_scale] * 3
                if not (cfg.get("contact_points_pose") and len(cfg["contact_points_pose"]) > 0):
                    y_center = float((cfg.get("center") or [0.0, 0.0, 0.0])[1])
                    cfg["contact_points_pose"] = [
                        [[6.123233995736766e-17, -1.0, 0.0, 0.0], [1.0, 6.123233995736766e-17, 0.0, y_center], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                        [[6.123233995736766e-17, -6.123233995736766e-17, 1.0, 0.0], [1.0, 3.749399456654644e-33, -6.123233995736766e-17, y_center], [0.0, 1.0, 6.123233995736766e-17, 0.0], [0.0, 0.0, 0.0, 1.0]],
                        [[6.123233995736766e-17, 1.0, 1.2246467991473532e-16, 0.0], [1.0, -6.123233995736766e-17, -7.498798913309288e-33, y_center], [0.0, 1.2246467991473532e-16, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                        [[6.123233995736766e-17, -6.123233995736766e-17, -1.0, 0.0], [1.0, 3.749399456654644e-33, 6.123233995736766e-17, y_center], [0.0, -1.0, 6.123233995736766e-17, 0.0], [0.0, 0.0, 0.0, 1.0]],
                    ]
                    cfg.setdefault("contact_points_group", [list(range(len(cfg["contact_points_pose"])))])
                    cfg.setdefault("contact_points_mask", [True])
            self.add_prohibit_area(self.milk_box, padding=0.04, area="table")

        self.move_thr = 0.05
        bb_box = get_actor_boundingbox(self.basket_right.actor)
        self.des_obj_pose = [
            np.random.uniform(low=bb_box[0][0] + 0.02, high=bb_box[1][0]),
            np.random.uniform(low=bb_box[0][1] - 0.05, high=bb_box[0][1] - 0.08),
            0.8,
        ] + [1, 0, 0, 0]
        self.add_prohibit_area(self.des_obj_pose, padding=0.0, area="table")
        print_c(f"Placement destination pose {self.des_obj_pose}", "RED")

    def play_once(self):
        arm_tag = ArmTag("left")
        self.move(
            self.grasp_actor(
                self.milk_box,
                arm_tag=arm_tag,
                pre_grasp_dis=0.07,
                grasp_dis=0.01,
                contact_point_id=self.contact_id,
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05))
        self.attach_object(
            self.milk_box,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{self.milk_box_modelname}/collision/base{self.milk_box_model_id}.glb",
            str(arm_tag),
        )
        self.move(
            self.place_actor(
                self.milk_box,
                arm_tag=arm_tag,
                target_pose=self.des_obj_pose,
                constrain="auto",
                pre_dis=0.07,
                dis=0.002,
            )
        )
        self.info["info"] = {
            "{A}": f"{self.milk_box_modelname}/base{self.milk_box_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        dist_thr = 0.15
        box_bb = get_actor_boundingbox(self.basket_right.actor)
        dist_to_box = point_to_box_distance(self.milk_box.get_pose().p, box_bb[0], box_bb[1])
        return (
            dist_to_box < dist_thr
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
