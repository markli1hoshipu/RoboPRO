# from envs._base_task import Base_Task
import sapien
import math
import glob
import yaml
import os
import numpy as np
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
from bench_envs.study._study_base_task import Study_base_task
from envs.utils import *
from bench_envs.utils.scene_gen_utils import get_position_limits, get_actor_boundingbox, get_collison_with_objs
from bench_envs.utils.scene_gen_utils import print_c, place_actor
from transforms3d.euler import euler2quat


class put_pen_in_pencup(Study_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        print_c(self.seed, "YELLOW")
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r") as f:
            task_objs = yaml.safe_load(f)
        
        xlim, ylim, self.side_to_place = get_position_limits(self.table,
                                         boundary_thr=[0.15, 0.25], side="right")
      
        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]
        
        self.target_name = "058_markpen"
        self.target_obj, self.target_id, self.target_pose = \
        place_actor(self.target_name, self, col_thr=0.15, xlim=[xlim[0], (xlim[0]+xlim[1])/2],
                    ylim=ylim, qpos=(90,0,90),object_bounds= object_bounds,
                     task_objs=task_objs,  obj_id = 0, mass = 0.1)
        

        tar_bb = get_actor_boundingbox(self.target_obj.actor)
        object_bounds.append(tar_bb)

        self.des_obj, self.des_obj_id, self.des_obj_pose = \
            place_actor("059_pencup", self, col_thr=0.15, xlim=[(xlim[0]+xlim[1])/2, xlim[1]],
                        ylim= ylim, qpos=(90,0,90),
                        object_bounds=object_bounds, task_objs=task_objs,
                        obj_id = 1, mass = 0.2, rotation=False)
        
        # Get the placement pose
        # des_bb = get_actor_boundingbox(self.des_obj.actor)
        # p = self.des_obj.get_pose().p.tolist() 
       
        print(f"target bb {tar_bb}")

        # tar_center = np.add(tar_bb[0], tar_bb[1])/2
        # offset = np.subtract(tar_center, self.target_obj.get_pose().p)
        # p = np.add(p, offset).tolist()
        # p[-1] = des_bb[1][-1]
        self.des_obj_pose = self.des_obj_pose.p.tolist() + euler2quat(0,0,np.deg2rad(90)).tolist()
        self.des_obj_pose[2] = 0.90
        #[1,0,0,0]
        # euler2quat(np.deg2rad(90),0,0).tolist()

        print_c(f"Placement destination pose {self.des_obj_pose}", "RED")


        self.add_prohibit_area(self.target_obj, padding=0.12, area="table")
        self.add_prohibit_area(self.des_obj, padding=0.12, area="table")

     
      
    def play_once1(self, z = 0.1, pre_dis= 0.07, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side_to_place ) #("right" if self.target_obj.get_pose().p[0] > 0 else "left")

        # Grasp the mouse with the selected arm
        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))
        self.attach_object(self.target_obj, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{self.target_name}/collision/base{self.target_id}.glb", str(arm_tag))

        # Lift the mouse upward by 0.1 meters in z-direction
        # self.move(self.move_by_displacement(arm_tag=arm_tag, z=z))

        # center the gripper
        # xy_disp = - self.target_obj.get_pose().p
        # self.move(self.move_by_displacement(arm_tag=arm_tag, x=xy_disp[0]))
        # self.move(self.move_by_displacement(arm_tag=arm_tag, y=-z*2))
       
        # # rotate the pen
        # self.move(self.move_by_displacement(arm_tag=arm_tag, z=z))
        
        self.move(self.move_to_pose(arm_tag, [0,-0.2,0.95,1,0,0,0]))

        self.move(self.move_by_displacement(arm_tag=arm_tag, quat=euler2quat(0,0,np.deg2rad(90)).tolist()))


        # move above the cup
        xy_disp = self.des_obj.get_pose().p - self.target_obj.get_pose().p
        self.move(self.move_by_displacement(arm_tag=arm_tag, x = xy_disp[0], y=xy_disp[1]))
        
        # move to cup
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=-z*2))
        self.move(self.open_gripper(arm_tag, 1))

        # self.des_obj_pose = self.des_obj_pose[:3] + self.target_obj.get_pose().q.tolist()

        # Place the mouse at the target location with alignment constraint
        # self.move(
        #     self.place_actor(
        #         self.target_obj,
        #         arm_tag=arm_tag,
        #         target_pose= self.des_obj_pose,
        #         constrain= [0.8,0.8,0.8,0.8,0.8,0.8],
        #         pre_dis=pre_dis,
        #         dis=dis,
        #     ))

        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"{self.target_name}/base{self.target_id}",
            "{B}": f"red",
            "{a}": str(arm_tag),
        }
        return self.info
    def play_oncebk(self, z = 0.15, pre_dis= 0.07, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side_to_place ) #("right" if self.target_obj.get_pose().p[0] > 0 else "left")

        # Grasp the mouse with the selected arm
        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))
        self.attach_object(self.target_obj, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{self.target_name}/collision/base{self.target_id}.glb", str(arm_tag))

        self.move(self.move_by_displacement(arm_tag=arm_tag, y= -z, z=z))
        # self.move(self.move_by_displacement(arm_tag=arm_tag, y = -z)
        self.move(self.move_by_displacement(arm_tag=arm_tag, quat=euler2quat(0,0,np.deg2rad(90)).tolist()))

        xy_disp = self.des_obj.get_pose().p[:2] - self.target_obj.get_pose().p[:2]

        xy_disp = xy_disp.tolist() + [0.941] +[1,0,0,0]
        self.move(self.move_to_pose(arm_tag, xy_disp, [0.8,0.8,0,0.8,0.8,0.8]))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=-z*2))
        self.move(self.open_gripper(arm_tag, 1))
        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"{self.target_name}/base{self.target_id}",
            "{B}": f"red",
            "{a}": str(arm_tag),
        }
        return self.info

    def play_once(self, z = 0.15, pre_dis= 0.05, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side_to_place ) #("right" if self.target_obj.get_pose().p[0] > 0 else "left")

        # Grasp the mouse with the selected arm
        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))
        self.attach_object(self.target_obj, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{self.target_name}/collision/base{self.target_id}.glb", str(arm_tag))

        self.move(self.move_by_displacement(arm_tag=arm_tag, y= -z, z=z))
        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose= self.des_obj_pose,
                constrain= "free",
                pre_dis=pre_dis,
                dis=dis,
                actor_axis_type="world"
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
        target_des_pos = self.des_obj.get_pose().p
        eps1 = 0.03
        eps2 = 0.03

        return (np.all(abs(target_pose[:2] - target_des_pos[:2]) < np.array([eps1, eps2]))
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
