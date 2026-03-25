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
from bench_envs.utils.scene_gen_utils import place_actor
from transforms3d.euler import euler2quat
from envs.utils.rand_create_actor import rand_pose


def _robotwin_log_move():
    return os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"


class move_books_onto_table(Study_base_task):

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj_1.get_name(), self.target_obj_2.get_name()}

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
        self.target_name = "043_book"# np.random.choice(list(task_objs['train']['study']['targets'].keys()))
        
        self.target_obj_1, self.target_id_1, _ = \
        place_actor(self.target_name, self, task_objs = task_objs,obj_id = 0,
                     obj_pose=[p_1,q], mass = 0.5, rotation=False)
        
        self.target_obj_2, _ , _  = \
        place_actor(self.target_name, self, task_objs = task_objs, obj_id = 0,
                     obj_pose=[p_2,q], mass = 0.5, rotation=False)

        self.lift_height = 0.2
        self.ep_lift = -0.2 #if self.arm_side == "right" else -xy_thr

        book_ylim = [ylim[0] + 0.0, ylim[1]-0.1]
        book_xlim = [xlim[0] + 0.01, xlim[1]-0.1]
        if _robotwin_log_move():
            print(f"[move_books_onto_table] book_xlim: {book_xlim}")
            print(f"[move_books_onto_table] book_ylim: {book_ylim}")
        
        while True:
            base_pose = rand_pose(
                xlim=xlim,
                ylim=book_ylim,
                zlim=[0.84],
                qpos=euler2quat(*[np.deg2rad(d) for d in [25, 180, 0]]),
                rotate_rand=None,
            )
            if not get_collison_with_objs(object_bounds, base_pose, 0.25):
                break

        self.des_obj_poses = [
            base_pose,
            sapien.Pose(
                [base_pose.p[0], base_pose.p[1]-0.015, base_pose.p[2] + 0.005],
                base_pose.q,
            ),
        ]
        if _robotwin_log_move():
            print(
                f"[move_books_onto_table] placing books at low={self.des_obj_poses[0]} "
                f"high={self.des_obj_poses[1]}"
            )
        self.add_prohibit_area(self.target_obj_1, padding=0.12, area="table")

      
    def play_once(self, pre_dis= 0.07, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.arm_side) #("right" if self.target_obj.get_pose().p[0] > 0 else "left")


        for i, target_obj in enumerate([self.target_obj_1, self.target_obj_2]):
            _p = target_obj.get_pose()
            if _robotwin_log_move():
                print(
                    f"[move_books_onto_table] target_obj (actor {target_obj.get_name()}): "
                    f"p={np.round(np.asarray(_p.p), 4)} q={np.round(np.asarray(_p.q), 4)}"
                )
            # Grasp the mouse with the selected arm
            self.move(self.grasp_actor(target_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))

            # Lift the mouse upward by 0.1 meters in z-direction
            # self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=self.lift_pose,
            #                              constraint_pose=[0,0,0,1,0,0,0]))
            # self.move(self.move_by_displacement(arm_tag=arm_tag, y= self.ep_lift, 
            #                                     z=self.lift_height, 
            #                                     constraint_pose=None))
            self.move(self.move_by_displacement(arm_tag=arm_tag, y= self.ep_lift, 
                                                z=self.lift_height))

            self.attach_object(target_obj, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{self.target_name}/collision/base{self.target_id_1}.glb", str(arm_tag))

            self.move(
                self.place_actor(
                    target_obj,
                    arm_tag=arm_tag,
                    target_pose=self.des_obj_poses[i],
                    constrain= "free",
                    actor_axis_type="world",
                    pre_dis=pre_dis,
                    dis=dis,
                    align_axis=[0, 0, 0]
                ))
            self.detach_object(arm_tag)
            
            self.move(self.move_by_displacement(arm_tag=arm_tag, x = -0.06, 
                                                y = -0.05, z = 0.03)) #DEBUG number

        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"{self.target_name}/base{self.target_id_1}",
            "{B}": f"red",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        eps1 = 0.05
        p1 = np.asarray(self.target_obj_1.get_pose().p)
        p2 = np.asarray(self.target_obj_2.get_pose().p)
        d0 = np.asarray(self.des_obj_poses[0].p)
        d1 = np.asarray(self.des_obj_poses[1].p)
        xy_ok = np.all(np.abs(p1[:2] - d0[:2]) < eps1) and np.all(
            np.abs(p2[:2] - d1[:2]) < eps1
        )
        expected_dz = float(d1[2] - d0[2])
        actual_dz = float(p2[2] - p1[2])
        z_stack_ok = actual_dz > 0 and abs(actual_dz - expected_dz) < eps1

        if _robotwin_log_move():
            e1_xy = p1[:2] - d0[:2]
            e2_xy = p2[:2] - d1[:2]
            dz_err = actual_dz - expected_dz
            print(
                f"[move_books_onto_table] check_success "
                f"xy_err_obj1={np.round(e1_xy, 4)} max_abs={np.max(np.abs(e1_xy)):.4f} "
                f"xy_err_obj2={np.round(e2_xy, 4)} max_abs={np.max(np.abs(e2_xy)):.4f} "
                f"dz_err={dz_err:.4f} (actual_dz={actual_dz:.4f} expected_dz={expected_dz:.4f}) "
                f"xy_ok={xy_ok} z_stack_ok={z_stack_ok}"
            )

        return bool(xy_ok and z_stack_ok)
