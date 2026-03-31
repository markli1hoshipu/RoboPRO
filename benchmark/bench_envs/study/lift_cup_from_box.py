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

class lift_cup_from_box(Study_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r") as f:
            task_objs = yaml.safe_load(f)

        self.des_obj = self.box
    
        des_bb = get_actor_boundingbox(self.des_obj.actor)

        self.target_name = "021_cup"

        place_pose =  [[self.des_obj.get_pose().p[0], 
                        self.des_obj.get_pose().p[1] +
                        np.random.uniform(low=-0.2, high=0),
                        des_bb[0][-1] + 0.03],(90,0,90)]

        self.target_obj, self.target_id, self.target_pose = \
            place_actor(self.target_name, self, task_objs = task_objs,
                    obj_pose=place_pose, mass = 0.2)
        
        if np.random.rand() > self.clean_background_rate:
            box_obs = "001_bottle"
            gap = 0.1
            place_pose =  [[np.random.choice([des_bb[0][0]+gap,
                                              des_bb[1][1]-gap]), 
                        des_bb[1][1]-gap,
                        des_bb[0][-1]],(90,0,0)]
            
            box_obs_tar, obs_tar_id, _= place_actor(box_obs, self, 
                           task_objs = task_objs, obj_id = 1,
                          obj_pose=place_pose, mass = 0.5, is_static=False)
            self.collision_list.append({
                "actor":box_obs_tar,
                "collision_path": self.col_temp.format(object=box_obs,
                                                        object_id=obs_tar_id)
            })


        self.init_tar_pose = self.target_obj.get_pose()
        self.lift_height = 0.15
        self.ep_lift = 0.15 if self.scene_id == 0 else -0.15
        self.arm_side = "left" if self.scene_id == 0 else "right"
        
        print_c(f"Lifting by {self.lift_height}", "RED")
        self.add_prohibit_area(self.target_obj, padding=0, area="table")

     
      
    def play_once(self, pre_dis= 0.07, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.arm_side) #("right" if self.target_obj.get_pose().p[0] > 0 else "left")

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
