# from envs._base_task import Base_Task
import sapien
import math
import glob
import yaml
import os
import numpy as np

from bench_envs.study._study_base_task import Study_base_task
from envs.utils import *
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
from bench_envs.utils.scene_gen_utils import get_position_limits, get_actor_boundingbox, get_collison_with_objs
from bench_envs.utils.scene_gen_utils import print_c, place_actor
from transforms3d.euler import euler2quat
from envs.utils.rand_create_actor import rand_pose

class move_books_onto_table(Study_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r") as f:
            task_objs = yaml.safe_load(f)
        xlim, ylim, self.side_to_place = get_position_limits(self.table,
                                      boundary_thr=0.20, side="right")
        
        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]        
        bcs_bb = get_actor_boundingbox(self.bookcase)
        x_w = bcs_bb[1][0] - bcs_bb[0][0]
        y_l = bcs_bb[1][1] - bcs_bb[0][1]

        q = (0,180,90)
        p_1 = [bcs_bb[0][0]+ x_w/5, bcs_bb[0][1] + y_l/2, 
               self.table.get_pose().p[-1] + 0.03]
        p_2 = [bcs_bb[0][0]+ x_w - x_w/5, bcs_bb[0][1] + y_l/2, 
               self.table.get_pose().p[-1] + 0.03] 
 
        self.arm_side = "right"
        self.target_name = "043_book"
        
        self.target_obj_1, self.target_id_1, _ = \
        place_actor(self.target_name, self, task_objs = task_objs,obj_id = 0,
                     obj_pose=[p_1,q], mass = 0.5, rotation=False)
        
        self.target_obj_2, _ , _  = \
        place_actor(self.target_name, self, task_objs = task_objs, obj_id = 0,
                     obj_pose=[p_2,q], mass = 0.5, rotation=False)

        self.lift_height = 0.2
        self.ep_lift = -0.2 

        while True:
            self.des_obj_pose  = rand_pose(
                xlim=xlim,
                ylim=ylim,
                zlim=[0.76],
                qpos=euler2quat(*[np.deg2rad(d) for d in [0,0,0]]), 
                rotate_rand=None,
            )
            if not get_collison_with_objs(object_bounds, self.des_obj_pose, 0.25):
                break

        print_c(f"placing book at {self.des_obj_pose}", "RED")
        self.add_prohibit_area(self.target_obj_1, padding=0.12, area="table")

      
    def play_once(self, pre_dis= 0.07, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.arm_side) 


        for target_pose in [self.target_obj_1, self.target_obj_2]:
            # Grasp the mouse with the selected arm
            self.move(self.grasp_actor(target_pose, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))

            # Lift the mouse upward by 0.1 meters in z-direction
            self.move(self.move_by_displacement(arm_tag=arm_tag, y= self.ep_lift, 
                                                z=self.lift_height, 
                                                constraint_pose=None))

            self.attach_object(target_pose, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{self.target_name}/collision/base{self.target_id_1}.glb", str(arm_tag))

            self.move(
                self.place_actor(
                    target_pose,
                    arm_tag=arm_tag,
                    target_pose= self.des_obj_pose,
                    constrain= [0,0,0,1,1,1],
                    actor_axis="world",
                    pre_dis=pre_dis,
                    dis=dis,
                    align_axis=[1, 1, 0]
                ))
            self.detach_object(arm_tag)

        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"{self.target_name}/base{self.target_id_1}",
            "{B}": f"red",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        eps1 = 0.015

        return abs(self.target_obj_1.get_pose().p[-1] - self.lift_height) < eps1
