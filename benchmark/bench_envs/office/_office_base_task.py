import os
import re
import sapien.core as sapien
from sapien.render import clear_cache as sapien_clear_cache
from sapien.utils.viewer import Viewer
import numpy as np
import gymnasium as gym
import pdb
import toppra as ta
import json
import transforms3d as t3d
from transforms3d.euler import euler2quat
from collections import OrderedDict
import torch, random

from bench_envs._bench_base_task import Bench_base_task
from envs.utils import *
from bench_envs.utils import *
import math
from envs.robot import Robot
from envs.camera import Camera
from envs.utils.actor_utils import Actor, ArticulationActor

from copy import deepcopy
import subprocess
from pathlib import Path
import trimesh
import imageio
import glob


from envs._GLOBAL_CONFIGS import *

from typing import Optional, Literal

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


class Office_base_task(Bench_base_task):

    FURNITURE_NAMES = {"table", "wall", "shelf", "ground"}

    def __init__(self):
        pass

    def _init_task_env_(self, table_xy_bias=[0, 0], table_height_bias=0, **kwags):
        """
        Initialization TODO
        - `self.FRAME_IDX`: The index of the file saved for the current scene.
        - `self.fcitx5-configtool`: Left gripper pose (close <=0, open >=0.4).
        - `self.ep_num`: Episode ID.
        - `self.task_name`: Task name.
        - `self.save_dir`: Save path.`
        - `self.left_original_pose`: Left arm original pose.
        - `self.right_original_pose`: Right arm original pose.
        - `self.left_arm_joint_id`: [6,14,18,22,26,30].
        - `self.right_arm_joint_id`: [7,15,19,23,27,31].
        - `self.render_fre`: Render frequency.
        """
        super().__init__() # need to fix this. right now this does nothing. it should be calling gym.Env.__init__()
        ta.setup_logging("CRITICAL")  # hide logging
        np.random.seed(kwags.get("seed", 0))
        torch.manual_seed(kwags.get("seed", 0))
        # random.seed(kwags.get('seed', 0))
        self.seed = kwags.get("seed", 0)

        self.FRAME_IDX = 0
        self.task_name = kwags.get("task_name")
        self.save_dir = kwags.get("save_path", "data")
        self.ep_num = kwags.get("now_ep_num", 0)
        self.render_freq = kwags.get("render_freq", 10)
        self.data_type = kwags.get("data_type", None)
        self.save_data = kwags.get("save_data", False)
        self.dual_arm = kwags.get("dual_arm", True)
        self.eval_mode = kwags.get("eval_mode", False)
        self.sample_d = kwags.get("sample_d", "objects")
        self.enable_collision_metrics = kwags.get("enable_collision_metrics", False)

        self.need_topp = True  # TODO

        # Random
        random_setting = kwags.get("domain_randomization")
        self.random_background = random_setting.get("random_background", False)
        self.cluttered_table = random_setting.get("cluttered_table", False)
        self.clean_background_rate = random_setting.get("clean_background_rate", 1)
        self.random_head_camera_dis = random_setting.get("random_head_camera_dis", 0)
        self.random_table_height = random_setting.get("random_table_height", 0)
        self.random_light = random_setting.get("random_light", False)
        self.crazy_random_light_rate = random_setting.get("crazy_random_light_rate", 0)
        self.crazy_random_light = (0 if not self.random_light else np.random.rand() < self.crazy_random_light_rate)
        self.random_embodiment = random_setting.get("random_embodiment", False)  # TODO
        self.obstacle_height = random_setting.get("obstacle_height", "short")
        self.obstacle_density = random_setting.get("obstacle_density", 3)

        self.file_path = []
        self.plan_success = True
        self.step_lim = None
        self.fix_gripper = False
        self.setup_scene(**kwags)

        self.left_js = None
        self.right_js = None
        self.raw_head_pcl = None
        self.real_head_pcl = None
        self.real_head_pcl_color = None

        self.now_obs = {}
        self.take_action_cnt = 0
        self.eval_video_path = kwags.get("eval_video_save_dir", None)

        self.save_freq = kwags.get("save_freq")
        self.world_pcd = None

        self.key_objects = []
        self.size_dict = list()
        self.cluttered_objs = list()
        self.prohibited_area = {"table": [], "shelf0": [], "shelf1": []} # shelf 0 for lower shelf, shelf 1 for upper shelf
        self.unstable_objects = ["050_bell"] # objects that are not stable and should be avoided
        # self.short_obstacles = [
        #     # "010_pen",
        #     # "015_laptop",
        #     "017_calculator",
        #     "021_cup",
        #     "022_cup-with-liquid",
        #     "023_tissue-box",
        #     "039_mug",
        #     "043_book",
        #     "047_mouse",
        #     "048_stapler",
        #     "055_small-speaker",
        #     "059_pencup",
        #     "063_tabletrashbin",
        #     "074_displaystand",
        #     "077_phone",
        #     "078_phonestand",
        #     "092_notebook",
        #     "099_fan",
        #     "100_seal",
        #     "116_keyboard",
        #     "120_plant"
        # ]
        # self.tall_obstacles = [
        #     "001_bottle",
        #     "089_globe",
        #     "095_glue",
        #     "097_screen",
        #     "098_speaker",
        #     "114_bottle",
        #     "119_mini-chalkboard"
        # ]
        self.record_cluttered_objects = list()  # record cluttered objects info
        self.cluttered_objects_info = get_cluttered_objects_info()

        self.eval_success = False
        self.table_z_bias = (np.random.uniform(low=-self.random_table_height, high=0) + table_height_bias)  # TODO
        self.office_info = {
            "table_height": 0.74,
            "table_area":[1.2, 0.7], # x,y area 
            "shelf_heights":[0.9, 1.13], # heights of the shelf levels
            "shelf_area":[0.62, 0.26], # x,y area 
            "file_holder_area":[0.22, 0.16], # x,y area 
            "furn_x_v": { # x position of furniture for each arrangement version
                "shelf": [-0.24,0,0.24],
                "cabinet": [0.23,0.48,-0.48],
                "file_holder": [0.48,-0.48,-0.23],
            },
        }
        self.item_info = get_task_objects_config()
        self.target_objects_info = get_target_objects_subset("office", self.sample_d)
        self.need_plan = kwags.get("need_plan", True)
        self.left_joint_path = kwags.get("left_joint_path", [])
        self.right_joint_path = kwags.get("right_joint_path", [])
        self.left_cnt = 0
        self.right_cnt = 0

        self.instruction = None  # for Eval

        self.collision_list = [] # list of collision objects for curobo planner
        self.cuboid_collision_list = [] # list of cuboid collision objects for curobo planner
        self._init_collision_metrics()

        self.arr_v = np.random.choice([-1,1], 1)[0] # which version to use for furniture arrangement

        self.load_robot(**kwags)
        self.create_static_elements(table_xy_bias=table_xy_bias)
        self.load_camera(**kwags)
        self.robot.move_to_homestate()

        render_freq = self.render_freq
        self.render_freq = 0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

        self.robot.set_origin_endpose()
        self.load_actors()

        if self.cluttered_table:
            self.load_basic_office_items()
            self.get_cluttered_surfaces()

        if self.enable_collision_metrics:
            self._build_collision_name_sets()  # build collision name sets for collision metrics

        is_stable, unstable_list = self.check_stable()
        if not is_stable:
            raise UnStableError(
                f'Objects is unstable in seed({kwags.get("seed", 0)}), unstable objects: {", ".join(unstable_list)}')
            # print(f'Objects is unstable in seed({kwags.get("seed", 0)}), unstable objects: {", ".join(unstable_list)}')

        self.update_world()

        if self.eval_mode:
            with open(os.path.join(os.environ["BENCH_ROOT"], "bench_task_config", "_bench_eval_step_limit.yml"), "r") as f:
                try:
                    data = yaml.safe_load(f)
                    self.step_lim = data[self.task_name]
                except:
                    print(f"{self.task_name} not in step limit file, set to 1000")
                    self.step_lim = 1000

        # info
        self.info = dict()
        self.info["cluttered_table_info"] = self.record_cluttered_objects
        self.info["texture_info"] = {
            "wall_texture": self.wall_texture,
            "table_texture": self.table_texture,
            "floor_texture": self.floor_texture,
        }
        self.info["info"] = {}

        self.stage_success_tag = False

    def create_static_elements(self, table_xy_bias=[0, 0]):
        # create static furniture elements
        self.table_xy_bias = table_xy_bias
        wall_texture, table_texture, floor_texture = None, None, None
        table_height = self.office_info["table_height"] + self.table_z_bias

        if self.random_background:
            texture_type = "seen" if not self.eval_mode else "unseen"
            directory_path = f"./assets/background_texture/{texture_type}"
            file_count = len(
                [name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])

            # wall_texture, table_texture = np.random.randint(0, file_count), np.random.randint(0, file_count)
            wall_texture = 43
            table_texture = 141
            floor_texture = 38

            self.wall_texture = f"seen/{wall_texture}"
            self.table_texture = f"seen/{table_texture}"
            self.floor_texture = f"seen/{floor_texture}"
            if np.random.rand() <= self.clean_background_rate:
                self.wall_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.table_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.floor_texture = None
        else:
            self.wall_texture, self.table_texture, self.floor_texture = None, None, None
        
        # floor--------------------------------------------------------
        self.floor_parts = []

        positions = [
            [1, 1, 0],    # top-right
            [-1, 1, 0],   # top-left
            [1, -1, 0],   # bottom-right
            [-1, -1, 0],  # bottom-left
        ]

        for i, pos in enumerate(positions):
            floor_part = create_visual_textured_box(
                self.scene,
                sapien.Pose(p=pos),
                half_size=[1, 1, 0.005],
                color=(0.85, 0.85, 0.85),
                name=f"floor_{i}",
                texture_id=self.floor_texture,
            )
            self.floor_parts.append(floor_part)
        
        # wall--------------------------------------------------------
        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=(1, 0.9, 0.9),
            name="wall",
            texture_id=self.wall_texture,
            is_static=True,
        )

        self.table = create_table(
            self.scene,
            sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], table_height]),
            length=1.2,
            width=0.7,
            height=table_height,
            thickness=0.05,
            is_static=True,
            texture_id=self.table_texture,
        )
        self.office_info["table_lims"] = [-self.office_info["table_area"][0]/2, -self.office_info["table_area"][1]/2, self.office_info["table_area"][0]/2, self.office_info["table_area"][1]/2]
        # ------------------------------------------------------------
        depth = 0.28
        shelf_scale = [1.7,0.86,1.8] # length, height, depth
        pose = [self.office_info["furn_x_v"]["shelf"][self.arr_v],depth,table_height+0.27]
        self.shelf = create_glb_actor(
            scene=self.scene,
            pose=sapien.Pose(p=pose, q=[0.7071, 0.7071, 0.0, 0.0]),
            model_name="121_wall-shelf",
            scale=shelf_scale,
            convex=False,
            is_static=True,
            mass=2
        )
        self.collision_list.append((self.shelf, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects_bench/121_wall-shelf/cc0_wall_shelf_4.glb", shelf_scale))
        xmin = pose[0] - self.office_info["shelf_area"][0]/2
        xmax = pose[0] + self.office_info["shelf_area"][0]/2
        ymin = pose[1] - self.office_info["shelf_area"][1]/2
        ymax = pose[1] + self.office_info["shelf_area"][1]/2
        self.office_info["shelf_lims"] = [xmin, ymin, xmax, ymax]
        self.prohibited_area["table"].append([xmin-0.03, ymin-0.02, xmax+0.03, ymax+0.02])
        # ------------------------------------------------------------
        self.cabinet = create_sapien_urdf_obj(
            scene=self,
            modelname="036_cabinet",
            modelid=46653,
            pose=sapien.Pose(p=[
                self.office_info["furn_x_v"]["cabinet"][self.arr_v],
                depth,
                table_height
                ], q=[0.7071, 0,0,0.7071]),
            fix_root_link=True,
        )
        self.add_prohibit_area(self.cabinet, padding=0.01)
        self.cabinet.set_mass(0.5)
        # ------------------------------------------------------------
        self.wooden_box = create_actor(
            scene=self,
            pose=sapien.Pose(p=[
                self.office_info["furn_x_v"]["file_holder"][self.arr_v],
                depth-0.06,
                0.813
                ], q=euler2quat(-np.pi/2,0,0, axes='sxyz')),
            modelname="042_wooden_box",
            convex=False,
            model_id=0,
            scale=[0.09,0.07,0.1],
            is_static=True,
        )
        # ------------------------------------------------------------
        self.file_holder = create_glb_actor(
            scene=self.scene,
            pose=sapien.Pose(p=[
                self.office_info["furn_x_v"]["file_holder"][self.arr_v],
                depth-0.05,
                table_height+0.075
                ], q=[0.7071, 0.7071, 0, 0]),
            model_name="122_file-holder",
            scale=[0.38,0.7,0.4], # width, height, depth
            convex=False,
            is_static=True,
            mass=0.1
        )
        pose = self.file_holder.get_pose().p
        xmin = pose[0] - self.office_info["file_holder_area"][0]/2
        xmax = pose[0] + self.office_info["file_holder_area"][0]/2
        ymin = pose[1] - self.office_info["file_holder_area"][1]/2
        ymax = pose[1] + self.office_info["file_holder_area"][1]/2
        self.office_info["file_holder_lims"] = [xmin, ymin, xmax, ymax]
        self.prohibited_area["table"].append([xmin-0.01, ymin, xmax+0.01, ymax])
        self.collision_list.append((self.file_holder, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects_bench/122_file-holder/base.glb", [1,1,1]))
        
    def load_basic_office_items(self):
        # load office items: items that are always placed as obstacles ie key obstacles
        entities = self.scene.get_all_actors()
        arts = self.scene.get_all_articulations()
        names = np.array([entity.get_name() for entity in entities])
        art_names = np.array([art.get_name() for art in arts])
        full_names = np.concatenate([names, art_names])

        size_dict = list()
        if "015_laptop" not in full_names:
            laptop_id = np.random.choice([9748,9912,9960,9968,9992,9996,10040,10098,10101,10125,10211])
            laptop_id = 9912
            laptop_file = {
                9748: "original-97",
                9912: "new-4",
                9960: "new-0",
                9968: "new-8",
                9992: "new-1",
                9996: "new-4",
                10040: "new-5",
                10098: "new-4",
                10101: "new-1",
                10125: "new-7",
                10211: "new-1",
            }
            success, self.laptop = rand_create_cluttered_actor(
                scene=self.scene,
                xlim=[-0.4, 0.4],
                ylim=[0.06],
                zlim=[self.office_info["table_height"]],
                modelname="015_laptop",
                modelid=laptop_id,
                modeltype="sapien_urdf",
                rotate_rand=True,
                rotate_lim=[0, 0, np.pi / 6],
                qpos=[0.7, 0, 0, 0.7],
                size_dict=size_dict,
                obj_radius=0.06,
                z_offset=0,
                z_max=0.1,
                prohibited_area=self.prohibited_area["table"],
                constrained=False,
                is_static=False,
            )
            if not success:
                raise RuntimeError("Failed to load laptop")
            self.laptop.set_mass(0.1)
            limit = self.laptop.get_qlimits()[0]
            self.laptop.set_qpos([limit[0] + (limit[1] - limit[0]) * 0.9])
            self.add_prohibit_area(self.laptop, padding=0.01)
            # custom collision because laptop has too many meshes
            cuboid_pose = self.laptop.get_pose().p.tolist() + [1, 0, 0, 0]
            cuboid_pose[1] += 0.04
            cuboid_pose[2] = self.office_info["table_height"] + 0.07
            self.cuboid_collision_list.append(("015_laptop", [0.2, 0.07, 0.14], cuboid_pose))
        # ------------------------------------------------------------
        if "plant" not in full_names:
            plant_id = 0
            success, self.plant = rand_create_cluttered_actor(
                scene=self.scene,
                xlim=[-0.5, 0.5],
                ylim=[0,0.11],
                zlim=[self.office_info["table_height"]],
                modelname="120_plant",
                modelid=plant_id,
                modeltype="glb",
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 1, 0],
                size_dict=size_dict,
                obj_radius=0.03,
                z_offset=self.cluttered_objects_info["120_plant"]["params"][f"{plant_id}"]["z_offset"],
                z_max=self.cluttered_objects_info["120_plant"]["params"][f"{plant_id}"]["z_max"],
                prohibited_area=self.prohibited_area["table"],
                constrained=False,
                is_static=False,
            )
            if not success:
                raise RuntimeError("Failed to load plant")
            self.plant.set_mass(1)
            pose = self.plant.get_pose().p
            self.prohibited_area["table"].append([pose[0]-0.03, pose[1]-0.03, pose[0]+0.03, pose[1]+0.03]) # manual because plant extents are incorrect
            self.collision_list.append((self.plant, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/120_plant/collision/base{plant_id}.glb", [1,1,1]))
    
    def get_cluttered_surfaces(self):
        # clutter surfaces with additional random obstacles
        # table ------------------------------------------------------
        xlim = [-0.59, 0.59]
        ylim = [-0.34, 0.34]
        zlim = [self.office_info["table_height"]]
        xlim[0] += self.table_xy_bias[0]
        xlim[1] += self.table_xy_bias[0]
        ylim[0] += self.table_xy_bias[1]
        ylim[1] += self.table_xy_bias[1]
        zlim = np.array(zlim) + self.table_z_bias
        
        self.clutter_surface_2(xlim=xlim, ylim=ylim, zlim=zlim, env_name="office", prohibited_area=self.prohibited_area["table"], obstacle_count=self.obstacle_density)
        # # shelves ----------------------------------------------------
        # xlim = [self.shelf.get_pose().p[0]-0.1, self.shelf.get_pose().p[0]+0.1]
        # ylim = [self.shelf.get_pose().p[1]-0.3, self.shelf.get_pose().p[1]+0.3]
        # zlim = [self.shelf_heights[0]]
        # self.clutter_surface(xlim=xlim, ylim=ylim, zlim=zlim, object_names=self.short_obstacles, prohibited_area=self.prohibited_area["shelf0"], obstacle_count=3)
        # zlim = [self.shelf_heights[1]]
        # self.clutter_surface(xlim=xlim, ylim=ylim, zlim=zlim, object_names=self.short_obstacles, prohibited_area=self.prohibited_area["shelf1"], obstacle_count=3)
    
    def load_table_obstacles_in_line(self, spacing=0.10, ground_y=-5.0, ground_z=0.02):
        """Used for debugging. Load all table_obstacles that are available into the scene in a line on the ground at y=ground_y, spacing (m) apart."""
        task_objects_list = []
        for entity in self.scene.get_all_actors():
            actor_name = entity.get_name()
            if actor_name == "":
                continue
            if actor_name in ["table", "wall", "ground"]:
                continue
            task_objects_list.append(actor_name)
        cluttered_item_info, obj_names = get_cluttered_objects_subset("office", task_objects_list)
        if not obj_names:
            return
        n = len(obj_names)
        # Center the line along x: first object at x = -(n-1)*spacing/2, then x += spacing
        x_start = -(n - 1) * spacing / 2.0
        identity_quat = [1, 0, 0, 0]
        for i, obj_name in enumerate(obj_names):
            if obj_name in self.unstable_objects:
                continue
            ids = cluttered_item_info[obj_name]["ids"]
            if not ids:
                continue
            obj_idx = ids[0]
            info = cluttered_item_info[obj_name]
            model_type = info["type"]
            x_i = x_start + i * spacing
            pose = sapien.Pose(
                [
                    x_i, ground_y, 
                    ground_z + cluttered_item_info[obj_name]["params"][obj_idx]["z_offset"]
                ], 
                identity_quat)
            if model_type == "urdf":
                obj = create_cluttered_urdf_obj(
                    scene=self.scene,
                    pose=pose,
                    modelname=f"objects/objaverse/{obj_name}/{obj_idx}",
                    fix_root_link=True,
                    scale=1,
                )
            else:
                obj = create_actor(
                    scene=self.scene,
                    pose=pose,
                    modelname=obj_name,
                    model_id=obj_idx,
                    convex=True,
                    is_static=True,
                    scale = [1,1,1]
                )
            if obj is None:
                continue
            obj.set_name(obj_name)
    
    def add_extra_cameras(self):
        self.cameras.add_extra_cameras(f"{os.environ['BENCH_ROOT']}/bench_assets/embodiments/office_config.yml")