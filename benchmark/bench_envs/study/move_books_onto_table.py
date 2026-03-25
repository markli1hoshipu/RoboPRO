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
                                      boundary_thr=0.20, side="right" if self.scene_id == 0 else "left")
        
        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]        
        bcs_bb = get_actor_boundingbox(self.bookcase)
        x_w = bcs_bb[1][0] - bcs_bb[0][0]
        y_l = bcs_bb[1][1] - bcs_bb[0][1]

        q = (0,180,90)
        p_1 = [bcs_bb[0][0]+ x_w/5, bcs_bb[0][1] + y_l/2, 
               self.table.get_pose().p[-1] + 0.03]
        p_2 = [bcs_bb[0][0]+ x_w - x_w/5, bcs_bb[0][1] + y_l/2, 
               self.table.get_pose().p[-1] + 0.03] 
 
        self.arm_side = self.side_to_place
        self.target_name = "043_book"
        
        self.target_obj_1, self.target_id_1, _ = \
        place_actor(self.target_name, self, task_objs = task_objs,obj_id = 0,
                     obj_pose=[p_1,q], mass = 0.5, rotation=False)
        
        self.target_obj_2, _ , _  = \
        place_actor(self.target_name, self, task_objs = task_objs, obj_id = 0,
                     obj_pose=[p_2,q], mass = 0.5, rotation=False)

        self.lift_height = 0.2
        self.ep_lift = -0.2

        book_xlim = [xlim[0] + 0.05, xlim[1]-0.1]
        book_ylim = [ylim[0] + 0.0, ylim[1]-0.1]

        if _robotwin_log_move():
            print(f"[move_books_onto_table] book_xlim: {book_xlim}")
            print(f"[move_books_onto_table] book_ylim: {book_ylim}")

        while True:
            base_pose = rand_pose(
                xlim=book_xlim,
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
                    align_axis=None
                ))
            self.detach_object(arm_tag)
            
            self.move(self.move_by_displacement(arm_tag=arm_tag, x = -0.06, 
                                                y = -0.05, z = 0.03))

        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"{self.target_name}/base{self.target_id_1}",
            "{B}": f"red",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        eps_xy = 0.05  # 5cm
        p1 = np.asarray(self.target_obj_1.get_pose().p)
        p2 = np.asarray(self.target_obj_2.get_pose().p)
        table_z = float(self.table.get_pose().p[2])
        table_bb = get_actor_boundingbox(self.table)

        # Both books must be within the table surface on xy
        on_table_1 = (table_bb[0][0] <= p1[0] <= table_bb[1][0] and
                      table_bb[0][1] <= p1[1] <= table_bb[1][1])
        on_table_2 = (table_bb[0][0] <= p2[0] <= table_bb[1][0] and
                      table_bb[0][1] <= p2[1] <= table_bb[1][1])

        # Lower book: 0-5cm above table surface
        dz1 = float(p1[2] - table_z)
        z1_ok = 0 <= dz1 <= eps_xy

        # Upper book: 0-5cm above lower book
        dz2 = float(p2[2] - p1[2])
        z2_ok = 0 <= dz2 <= eps_xy

        # XY: second book within 5cm of first book
        xy_ok = bool(np.all(np.abs(p2[:2] - p1[:2]) < eps_xy))

        if _robotwin_log_move():
            print(
                f"[move_books_onto_table] check_success "
                f"table_bb_x=[{table_bb[0][0]:.4f}, {table_bb[1][0]:.4f}] "
                f"table_bb_y=[{table_bb[0][1]:.4f}, {table_bb[1][1]:.4f}] "
                f"on_table_1={on_table_1} on_table_2={on_table_2} "
                f"table_z={table_z:.4f} "
                f"dz1(book1-table)={dz1:.4f} z1_ok={z1_ok} "
                f"dz2(book2-book1)={dz2:.4f} z2_ok={z2_ok} "
                f"xy_diff={np.round(p2[:2] - p1[:2], 4)} xy_ok={xy_ok}"
            )

        return bool(on_table_1 and on_table_2 and z1_ok and z2_ok and xy_ok)
