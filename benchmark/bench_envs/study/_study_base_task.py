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
from collections import OrderedDict
import torch, random

from bench_envs._bench_base_task import Bench_base_task
from envs.utils import *
from bench_envs.utils import *
import math
from envs.robot import Robot
from envs.camera import Camera
from envs.utils.actor_utils import Actor, ArticulationActor
from bench_envs.utils.scene_gen_utils import get_actor_boundingbox
from bench_envs.utils.scene_gen_utils import print_c

from copy import deepcopy
import subprocess
from pathlib import Path
import trimesh
import imageio
import glob
import yaml
import importlib

from envs._GLOBAL_CONFIGS import *

from typing import Optional, Literal
from transforms3d.euler import euler2quat

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


class Study_base_task(Bench_base_task):
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
        print_c(f"#### Seed value {self.seed} ####", "YELLOW")
        self.FRAME_IDX = 0
        self.task_name = kwags.get("task_name")
        self.save_dir = kwags.get("save_path", "data")
        self.ep_num = kwags.get("now_ep_num", 0)
        self.render_freq = kwags.get("render_freq", 10) 
        self.data_type = kwags.get("data_type", None)
        self.save_data = kwags.get("save_data", False)
        self.dual_arm = kwags.get("dual_arm", True)
        self.eval_mode = kwags.get("eval_mode", False)
        self.scene_objs = []
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

        self.file_path = []
        self.plan_success = True
        self.step_lim = None
        self.fix_gripper = False
        self.setup_scene()

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

        self.size_dict = list()
        self.cluttered_objs = list()
        self.prohibited_area = {"table": [], "shelf0": [], "shelf1": []} # shelf 0 for lower shelf, shelf 1 for upper shelf
        self.record_cluttered_objects = list()  # record cluttered objects info

        self.eval_success = False
        self.table_z_bias = (np.random.uniform(low=-self.random_table_height, high=0) + table_height_bias)  # TODO
        self.shelf_heights = [0.845, 1.265] # heights of the shelf levels
        self.need_plan = kwags.get("need_plan", True)
        self.left_joint_path = kwags.get("left_joint_path", [])
        self.right_joint_path = kwags.get("right_joint_path", [])
        self.left_cnt = 0
        self.right_cnt = 0
        self.scene_id = kwags.get("scene_id", 0)

        self.instruction = None  # for Eval

        self.collision_list = [] # list of collision objects for curobo planner

        self.load_robot(**kwags)
        self.create_static_elements(table_xy_bias=table_xy_bias, table_height=0.74)
        self.load_camera(**kwags)
        self.robot.move_to_homestate()

        render_freq = self.render_freq
        self.render_freq = 0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq
        self.cuboid_collision_list = []
        self.robot.set_origin_endpose()
        self.load_actors()

        if self.cluttered_table:
            self.get_cluttered_surfaces()

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


    def create_static_elements(self, table_xy_bias=[0, 0], table_height=0.74):
        self.table_xy_bias = table_xy_bias
        wall_texture, table_texture, floor_texture = None, None, None
        table_height += self.table_z_bias

        # Setup textures 
        if self.random_background:
            texture_type = "seen" if not self.eval_mode else "unseen"
            directory_path = f"./assets/background_texture/{texture_type}"
            file_count = len(
                [name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])

            # wall_texture, table_texture = np.random.randint(0, file_count), np.random.randint(0, file_count)
            if texture_type == "seen":
                wall_texture = np.random.randint(0, file_count)
                table_texture = np.random.choice(
                    # simple textures, not distracting
                    [0,2,4,5,7,9,14,16,18,19]
                    )
                floor_texture = np.random.choice(
                    # simple textures, not distracting
                    [2,3,4,5,6,17,47,71,110]
                    )
            else:
                wall_texture = np.random.randint(0, file_count)
                table_texture = np.random.choice(
                    # simple textures, not distracting
                    [1,8,9,27,29,30,37,55]
                    )
                floor_texture = np.random.choice(
                    # simple textures, not distracting
                    [0,6,8,23,34,41,47,61,64]
                    )

            self.wall_texture = f"{texture_type}/{wall_texture}"
            self.table_texture = f"{texture_type}/{table_texture}"
            self.floor_texture = f"{texture_type}/{floor_texture}"
            if np.random.rand() <= self.clean_background_rate:
                self.wall_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.table_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.floor_texture = None
        else:
            self.wall_texture, self.table_texture, self.floor_texture = None, None, None

        # Create room
        self.floor = create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[0, 0, 0]),
            half_size=[2, 2, 0.005],
            color=(0.85, 0.85, 0.85),
            name="floor",
            texture_id=self.floor_texture,
        )

        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=(1, 0.9, 0.9),
            name="wall",
            texture_id=self.wall_texture,
            is_static=True,
        )

        # Create scene specific elements
        with open(f"{os.environ['BENCH_ROOT']}/bench_envs/scene_configs/study.yaml", "r") as f:
            obj_config = yaml.safe_load(f)
        for obj, param in obj_config[f"scene_{self.scene_id}"].items():
            pose = sapien.Pose(
                    p = param.get("p", [*table_xy_bias, table_height]),
                    q = euler2quat(*[np.deg2rad(d) for d in param.get("q", [0,0,0])])
            )
            if param.get("obj_type", "actor") == "actor":
                value = create_actor(
                    scene=self,
                    pose=pose,
                    modelname=obj,
                    convex=param.get("convex", True),
                    scale=param.get("scale"),
                    model_id= param.get('model_id', 0),
                    is_static=param.get('is_static', True)
                )
                self.scene_objs.append(value)
                # if "bookcase" not in obj:
                #     self.collision_list.append((value, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{obj}/collision/base{param.get('model_id', 0)}.glb", param.get("scale", [1,1,1])))

                self.add_prohibit_area(value, padding=0.12, area="table")

            elif param.get("obj_type", "actor") == "table":
                value= create_table(
                        self.scene,
                        pose=pose,
                        length=param.get("length", 1.2),
                        width=param.get("width", 0.7),
                        height=param.get("height", table_height),
                        thickness=param.get("tickness", 0.05),
                        is_static=param.get('is_static', True),
                        texture_id=param.get('texture_id', self.table_texture),
                    )
            setattr(self, obj.split("_")[-1], value)

    def create_static_elementsv2(self, table_xy_bias=[0, 0], table_height=0.74):
        self.table_xy_bias = table_xy_bias
        wall_texture, table_texture, floor_texture = None, None, None
        table_height += self.table_z_bias

        # Setup textures 
        if self.random_background:
            texture_type = "seen" if not self.eval_mode else "unseen"
            directory_path = f"./assets/background_texture/{texture_type}"
            file_count = len(
                [name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])

            # wall_texture, table_texture = np.random.randint(0, file_count), np.random.randint(0, file_count)
            if texture_type == "seen":
                wall_texture = np.random.randint(0, file_count)
                table_texture = np.random.choice(
                    # simple textures, not distracting
                    [0,2,4,5,7,9,14,16,18,19]
                    )
                floor_texture = np.random.choice(
                    # simple textures, not distracting
                    [2,3,4,5,6,17,47,71,110]
                    )
            else:
                wall_texture = np.random.randint(0, file_count)
                table_texture = np.random.choice(
                    # simple textures, not distracting
                    [1,8,9,27,29,30,37,55]
                    )
                floor_texture = np.random.choice(
                    # simple textures, not distracting
                    [0,6,8,23,34,41,47,61,64]
                    )

            self.wall_texture = f"{texture_type}/{wall_texture}"
            self.table_texture = f"{texture_type}/{table_texture}"
            self.floor_texture = f"{texture_type}/{floor_texture}"
            if np.random.rand() <= self.clean_background_rate:
                self.wall_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.table_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.floor_texture = None
        else:
            self.wall_texture, self.table_texture, self.floor_texture = None, None, None

        # Create room
        self.floor = create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[0, 0, 0]),
            half_size=[2, 2, 0.005],
            color=(0.85, 0.85, 0.85),
            name="floor",
            texture_id=self.floor_texture,
        )

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
        with open(f"{os.environ['BENCH_ROOT']}/bench_task_config/task_objects.yml", "r") as f:
            obj_config = yaml.safe_load(f)
        obj_name = "058_markpen"
        obj = obj_config["scales"][obj_name]
        max_num = len(obj)//2
        for i, (obj_ins, param) in enumerate(obj.items()):
            print(obj_name, obj_ins, param)

            pose = sapien.Pose(
                    p = [(-max_num* 0.2)+(i*0.2), 0, table_height],
                    # q = [0.7171,0,0,0.7171] #[1,0,0,0]
                    q = [1,0,0,0]

            )
            value = create_actor(
                    scene=self,
                    pose=pose,
                    modelname=obj_name,
                    convex= True,
                    scale= param if param != 1 else None,
                    model_id= int(obj_ins),
                    is_static= True
            )
            bbox = get_actor_boundingbox(value.actor)
            if bbox[0][1] < pose.p[1]:
                value.actor.set_pose(
                    sapien.Pose(
                        p = [pose.p[0], pose.p[1]+ pose.p[1]- bbox[0][1], pose.p[2]], 
                        q = pose.q)
                )
    
    def get_cluttered_surfaces(self):
        self.get_cluttered_table()

    def get_cluttered_table(self, cluttered_numbers=10, xlim=[-0.59, 0.59], ylim=[-0.34, 0.34], zlim=[0.741]):
        self.record_cluttered_objects = []  # record cluttered objects

        xlim[0] += self.table_xy_bias[0]
        xlim[1] += self.table_xy_bias[0]
        ylim[0] += self.table_xy_bias[1]
        ylim[1] += self.table_xy_bias[1]

        if np.random.rand() < self.clean_background_rate:
            return

        task_objects_list = []
        for entity in self.scene.get_all_actors():
            actor_name = entity.get_name()
            if actor_name == "":
                continue
            if actor_name in ["table", "wall", "ground"]:
                continue
            task_objects_list.append(actor_name)
        self.obj_names, self.cluttered_item_info = get_available_cluttered_objects(task_objects_list)

        success_count = 0
        max_try = 50
        trys = 0

        while success_count < cluttered_numbers and trys < max_try:
            obj = np.random.randint(len(self.obj_names))
            obj_name = self.obj_names[obj]
            # if obj_name in self.unstable_objects:
            #     continue
            obj_idx = np.random.randint(len(self.cluttered_item_info[obj_name]["ids"]))
            obj_idx = self.cluttered_item_info[obj_name]["ids"][obj_idx]
            obj_radius = self.cluttered_item_info[obj_name]["params"][obj_idx]["radius"]
            obj_offset = self.cluttered_item_info[obj_name]["params"][obj_idx]["z_offset"]
            obj_maxz = self.cluttered_item_info[obj_name]["params"][obj_idx]["z_max"]

            success, self.cluttered_obj = rand_create_cluttered_actor(
                self.scene,
                xlim=xlim,
                ylim=ylim,
                zlim=np.array(zlim) + self.table_z_bias,
                modelname=obj_name,
                modelid=obj_idx,
                modeltype=self.cluttered_item_info[obj_name]["type"],
                rotate_rand=True,
                rotate_lim=[0, 0, math.pi],
                size_dict=self.size_dict,
                obj_radius=obj_radius,
                z_offset=obj_offset,
                z_max=obj_maxz,
                prohibited_area=self.prohibited_area["table"],
            )
            if not success or self.cluttered_obj is None:
                trys += 1
                continue
            self.cluttered_obj.set_name(f"{obj_name}")

            # manage stability as distractors
            self.stabilize_object(self.cluttered_obj)

            self.cluttered_objs.append(self.cluttered_obj)
            pose = self.cluttered_obj.get_pose().p.tolist()
            pose.append(obj_radius)
            self.size_dict.append(pose)
            success_count += 1
            self.record_cluttered_objects.append({"object_type": obj_name, "object_index": obj_idx})

            # add to collision list--------------------------------------------------------------------------------
            if self.cluttered_item_info[obj_name]["type"] == "urdf":
                path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/objaverse/{obj_name}/{obj_idx}/coacd_collision.obj"
            else:
                path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{obj_name}/collision/base{obj_idx}.glb"
            self.collision_list.append((self.cluttered_obj, path, self.cluttered_obj.scale))

        if success_count < cluttered_numbers:
            print(f"Warning: Only {success_count} cluttered objects are placed on the table.")

        self.size_dict = None
        self.cluttered_objs = []

    # def get_cluttered_shelf(self, cluttered_numbers=6, xlim=[0.95], ylim=[-0.65], zlim=[0.97], shelf_level=0):
    #     self.record_cluttered_objects = []  # record cluttered objects
    #     self.size_dict = list()

    #     xlim = [self.shelf.get_pose().p[0]-0.1, self.shelf.get_pose().p[0]+0.1]
    #     ylim = [self.shelf.get_pose().p[1]-0.3, self.shelf.get_pose().p[1]+0.3]
    #     zlim = [self.shelf_heights[shelf_level]]

    #     if np.random.rand() < self.clean_background_rate:
    #         return

    #     task_objects_list = []
    #     for entity in self.scene.get_all_actors():
    #         actor_name = entity.get_name()
    #         if actor_name == "":
    #             continue
    #         if actor_name in ["table", "wall", "ground"]:
    #             continue
    #         task_objects_list.append(actor_name)
    #     self.obj_names, self.cluttered_item_info = get_available_cluttered_objects(task_objects_list)

    #     success_count = 0
    #     max_try = 50
    #     trys = 0

    #     while success_count < cluttered_numbers and trys < max_try:
    #         obj = np.random.randint(len(self.obj_names))
    #         obj_name = self.obj_names[obj]
    #         # if obj_name in self.unstable_objects:
    #         #     continue
    #         obj_idx = np.random.randint(len(self.cluttered_item_info[obj_name]["ids"]))
    #         obj_idx = self.cluttered_item_info[obj_name]["ids"][obj_idx]
    #         obj_radius = self.cluttered_item_info[obj_name]["params"][obj_idx]["radius"]
    #         obj_offset = self.cluttered_item_info[obj_name]["params"][obj_idx]["z_offset"]
    #         obj_maxz = self.cluttered_item_info[obj_name]["params"][obj_idx]["z_max"]

    #         success, self.cluttered_obj = rand_create_cluttered_actor(
    #             self.scene,
    #             xlim=xlim,
    #             ylim=ylim,
    #             zlim=zlim,
    #             modelname=obj_name,
    #             modelid=obj_idx,
    #             modeltype=self.cluttered_item_info[obj_name]["type"],
    #             rotate_rand=True,
    #             rotate_lim=[0, 0, math.pi],
    #             size_dict=self.size_dict,
    #             obj_radius=obj_radius,
    #             z_offset=obj_offset,
    #             z_max=obj_maxz,
    #             prohibited_area=self.prohibited_area[f"shelf{shelf_level}"],
    #             shelf=True,
    #         )
    #         if not success or self.cluttered_obj is None:
    #             trys += 1
    #             continue
    #         self.cluttered_obj.set_name(f"{obj_name}")

    #         # manage stability as distractors
    #         self.stabilize_object(self.cluttered_obj)

    #         self.cluttered_objs.append(self.cluttered_obj)
    #         pose = self.cluttered_obj.get_pose().p.tolist()
    #         pose.append(obj_radius)
    #         self.size_dict.append(pose)
    #         success_count += 1
    #         self.record_cluttered_objects.append({"object_type": obj_name, "object_index": obj_idx})

    #         # add to collision list--------------------------------------------------------------------------------
    #         if self.cluttered_item_info[obj_name]["type"] == "urdf":
    #             path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/objaverse/{obj_name}/{obj_idx}/coacd_collision.obj"
    #         else:
    #             path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{obj_name}/collision/base{obj_idx}.glb"
    #         self.collision_list.append((self.cluttered_obj, path, self.cluttered_obj.scale))

    #     # if success_count < cluttered_numbers:
    #         # print(f"Warning: Only {success_count} cluttered objects are placed on the shelf.")

    #     self.size_dict = None
    #     self.cluttered_objs = []
    
    def add_extra_cameras(self):
        self.cameras.add_extra_cameras(f"{os.environ['BENCH_ROOT']}/bench_assets/embodiments/office_config.yml")