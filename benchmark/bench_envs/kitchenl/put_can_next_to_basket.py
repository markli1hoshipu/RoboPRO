import os

import yaml

from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from bench_envs.utils.scene_gen_utils import get_actor_boundingbox, place_actor, point_to_box_distance, print_c
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d


class put_can_next_to_basket(Kitchen_base_large):
    MILK_BOX_MASS = 0.1
    MILK_BOX_SPAWN_Z_OFFSET = 0
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

    def setup_demo(self, is_test: bool = False, **kwargs):
        self.can_box_modelname = "071_can"
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r") as f:
            task_objs = yaml.safe_load(f)
        self.can_box_model_ids = task_objs["objects"]["kitchenl"]["targets"][self.can_box_modelname]
        kwargs["scene_id"] = np.random.choice([0,1])
        kwargs["include_collision"] = True
        self.can_box_spawn_rot_deg = [0.45, 0.0, 90.0]

        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _can_box_quat_from_cfg(self) -> list[float]:
        roll_deg, pitch_deg, yaw_deg = self.can_box_spawn_rot_deg
        ax = math.radians(roll_deg)
        ay = math.radians(pitch_deg)
        az = math.radians(yaw_deg)
        qx, qy, qz, qw = t3d.euler.euler2quat(ax, ay, az)
        return [qw, qx, qy, qz]

    def _table_center_spawn_pose(self, table_center: np.ndarray) -> sapien.Pose:
        if self.scene_id == 1:
            xlim = [-0.45, -0.2]
            self.contact_id = 1
        else:
            xlim = [-0.15, 0]
            self.contact_id = None
        x = float(np.random.uniform(xlim[0], xlim[1]))
        y = float(np.random.uniform(-0.1,0))
        z = float(table_center[2] + self.MILK_BOX_SPAWN_Z_OFFSET)

        return sapien.Pose([x, y, z], self._can_box_quat_from_cfg())

    def load_actors(self):
        table_center = np.array(self.table.get_pose().p, dtype=float)
        self.can_box_model_id = int(np.random.choice(self.can_box_model_ids))
        spawn_pose = self._table_center_spawn_pose(table_center)


        self.can_box = create_actor(
            scene=self.scene,
            pose=spawn_pose,
            modelname=self.can_box_modelname,
            model_id=self.can_box_model_id,
            is_static=False,  # requested: start as static
            convex=True,
            scale=0.05,
        )
        self.add_prohibit_area(self.can_box, padding=0.04, area="table")

        self.move_thr = 0.05
        bb_box = get_actor_boundingbox(self.basket_right.actor)        
        self.des_obj_pose = [ np.random.uniform(low=bb_box[0][0]+0.02, high=bb_box[1][0]),
             np.random.uniform(low=bb_box[0][1]-0.05, high=bb_box[0][1]-0.08),
             0.8] + [1,0,0,0]
        
        self.add_prohibit_area(self.des_obj_pose, padding=0.0, area="table")

        print_c(f"Placement destination pose {self.des_obj_pose}", "RED")
        

    def play_once(self):
        arm_tag = ArmTag("left")
        self.move(
            self.grasp_actor(
                self.can_box,
                arm_tag=arm_tag,
                pre_grasp_dis=self.GRASP_PRE_DIS,
                grasp_dis=self.GRASP_DIS,
                contact_point_id= self.contact_id
                # gripper_pos=self.GRASP_CLOSE_POS,
            )
        )
        self.attach_object(self.can_box, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{self.can_box_modelname}/collision/base{self.can_box_model_id}.glb", str(arm_tag))

        self.move(
            self.place_actor(
                self.can_box,
                arm_tag=arm_tag,
                target_pose= self.des_obj_pose,
                constrain= "auto",
                pre_dis=0.01,
                dis=0.002,
            ))
        self.move(self.move_by_displacement(arm_tag, z = 0.04))
        self.info["info"] = {
            "{A}": f"{self.can_box_modelname}/base{self.can_box_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        dist_thr = 0.15
        box_bb = get_actor_boundingbox(self.basket_right.actor)
        dist_to_box = point_to_box_distance(self.can_box.get_pose().p, box_bb[0], box_bb[1])

        return (dist_to_box < dist_thr
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())

