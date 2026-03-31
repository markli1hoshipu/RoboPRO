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
from bench_envs.utils.scene_gen_utils import get_position_limits, get_actor_boundingbox
from bench_envs.utils.scene_gen_utils import print_c, place_actor

class lift_cup_from_book(Study_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r") as f:
            task_objs = yaml.safe_load(f)
        
        xlim, ylim, self.arm_side = get_position_limits(self.table, boundary_thr=0.25)
       
        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]

        self.des_obj_name = "043_book"
        self.des_obj, self.des_obj_id, self.des_obj_pose = \
        place_actor(self.des_obj_name, self, col_thr =0.15,
                     xlim=xlim, ylim=ylim, qpos=(90,0,90),
                     object_bounds = object_bounds, task_objs = task_objs,
                     obj_id = None, mass = 0.5, rotation=False)
       
        # self.collision_list.append({
        #     "actor":self.des_obj,
        #     "collision_path": self.col_temp.format(object=self.des_obj_name, object_id=self.des_obj_id)
        # })

        des_bb = get_actor_boundingbox(self.des_obj.actor)

        self.target_name = "021_cup"
        
        self.target_pose = self.des_obj_pose
        self.target_pose.set_p([*self.target_pose.p[:2], des_bb[1][-1]]) 

        self.target_obj, self.target_id, self.target_pose = \
            place_actor(self.target_name, self, task_objs = task_objs,
                    obj_pose=self.target_pose, mass = 0.2)
        self.init_tar_pose = self.target_obj.get_pose()
        # self.collision_list.append({
        #     "actor":self.target_obj,
        #     "collision_path": self.col_temp.format(object=self.target_name, object_id=self.target_id)
        # })

        self.lift_height = 0.2
        xy_thr = 0.2
        self.ep_lift = xy_thr if self.arm_side == "right" else -xy_thr
        print_c(f"Lifting by {self.lift_height}", "RED")
        self.add_prohibit_area(self.target_obj, padding=0.0, area="table")
        self.add_prohibit_area(self.des_obj, padding=0.0, area="table")

     
      
    def play_once(self, pre_dis= 0.07, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.arm_side)
        # Grasp the mouse with the selected arm
        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))

        # Lift the mouse upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, x= self.ep_lift, 
                                            z=self.lift_height, 
                                            constraint_pose=None))

        self.attach_object(self.target_obj, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{self.target_name}/collision/base{self.target_id}.glb", str(arm_tag))


        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"{self.target_name}/base{self.target_id}",
            "{B}": f"red",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        eps1 = 0.05
        return abs(self.target_obj.get_pose().p[-1] - (self.init_tar_pose.p[-1] + self.lift_height) )< eps1
