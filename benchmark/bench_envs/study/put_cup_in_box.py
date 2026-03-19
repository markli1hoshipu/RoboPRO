# from envs._base_task import Base_Task
import sapien
import math
import glob
import yaml
import os
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
from bench_envs.study._study_base_task import Study_base_task
from envs.utils import *
from bench_envs.utils.scene_gen_utils import get_position_limits, get_actor_boundingbox, get_collison_with_objs

class put_cup_in_box(Study_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r") as f:
            task_objs = yaml.safe_load(f)
        
        xlim, ylim, self.side_to_place = get_position_limits(self.table, boundary_thr=0.20, side="left")
       
        print(xlim, ylim, self.side_to_place)
        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]

        for bb, o in zip(object_bounds, self.scene_objs):
            print(o.get_name(), bb)
        # Threshold between the objects
        col_thr = 0.15


        while True:
            tar_obj_rand_pos = rand_pose(
                xlim=xlim,
                ylim=ylim,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                # rotate_lim=[0, 3.14, 0],
            )
            if not get_collison_with_objs(object_bounds, tar_obj_rand_pos, col_thr):
                break
                    
            # if abs(tar_obj_rand_pos.p[0]) > 0.3:
            #     break
      
        self.target_name = "021_cup"# np.random.choice(list(task_objs['train']['study']['targets'].keys()))
        self.target_id = np.random.choice(task_objs['objects']['study']['targets'][self.target_name])
        
        print(f"Generating {self.target_name} with id {self.target_id} at position {tar_obj_rand_pos}")

        self.target_obj = create_actor(
            scene=self,
            pose=tar_obj_rand_pos,
            modelname=self.target_name,
            convex=True,
            model_id= self.target_id ,
            scale= None if task_objs['scales'].get(self.target_name) is None else  task_objs['scales'][self.target_name].get(str(self.target_id)) 
        )
        self.target_obj.set_mass(0.1)
        # Create destination object
     
        self.des_obj = self.box

        des_bb = get_actor_boundingbox(self.des_obj.actor)
        p = self.des_obj.get_pose().p.tolist() 
        p[-1] = des_bb[1][-1]
        self.des_obj_pose = p + [1, 0, 0, 0]
        print(f"Placement destination pose {self.des_obj_pose}")


        self.add_prohibit_area(self.target_obj, padding=0.12, area="table")
        self.add_prohibit_area(self.des_obj, padding=0.12, area="table")

     
      
    def play_once(self, z = 0.1, pre_dis= 0.07, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side_to_place ) #("right" if self.target_obj.get_pose().p[0] > 0 else "left")

        # Grasp the mouse with the selected arm
        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))

        # Lift the mouse upward by 0.1 meters in z-direction
        # self.move(self.move_by_displacement(arm_tag=arm_tag, z=z))

        self.attach_object(self.target_obj, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{self.target_name}/collision/base{self.target_id}.glb", str(arm_tag))

        # Place the mouse at the target location with alignment constraint
        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose= self.des_obj_pose,
                constrain="auto",
                pre_dis=pre_dis,
                dis=dis,
            ))

        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"{self.target_name}/base{self.target_id}",
            "{B}": f"red",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        target_pose = self.target_obj.get_pose().p
        target_qpose = np.abs(self.target_obj.get_pose().q)
        target_des_pos = self.target_obj.get_pose().p
        eps1 = 0.015
        eps2 = 0.012

        return (np.all(abs(target_pose[:2] - target_des_pos[:2]) < np.array([eps1, eps2]))
                and (np.abs(target_qpose[2] * target_qpose[3] - 0.49) < eps1
                     or np.abs(target_qpose[0] * target_qpose[1] - 0.49) < eps1) and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
