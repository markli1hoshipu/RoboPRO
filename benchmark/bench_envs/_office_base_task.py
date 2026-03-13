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

        self.key_objects = []
        self.size_dict = list()
        self.cluttered_objs = list()
        self.prohibited_area = {"table": [], "shelf0": [], "shelf1": []} # shelf 0 for lower shelf, shelf 1 for upper shelf
        self.unstable_objects = ["050_bell"] # objects that are not stable and should be avoided
        self.short_obstacles = [
            # "010_pen",
            # "015_laptop",
            "017_calculator",
            "021_cup",
            "022_cup-with-liquid",
            "023_tissue-box",
            "039_mug",
            "043_book",
            "047_mouse",
            "048_stapler",
            "055_small-speaker",
            "059_pencup",
            "063_tabletrashbin",
            "074_displaystand",
            "077_phone",
            "078_phonestand",
            "092_notebook",
            "099_fan",
            "100_seal",
            "116_keyboard",
            "120_plant"
        ]
        self.tall_obstacles = [
            "001_bottle",
            "089_globe",
            "095_glue",
            "097_screen",
            "098_speaker",
            "114_bottle",
            "119_mini-chalkboard"
        ]
        self.record_cluttered_objects = list()  # record cluttered objects info

        self.eval_success = False
        self.table_z_bias = (np.random.uniform(low=-self.random_table_height, high=0) + table_height_bias)  # TODO
        self.shelf_heights = [0.845, 1.265] # heights of the shelf levels
        self.need_plan = kwags.get("need_plan", True)
        self.left_joint_path = kwags.get("left_joint_path", [])
        self.right_joint_path = kwags.get("right_joint_path", [])
        self.left_cnt = 0
        self.right_cnt = 0

        self.instruction = None  # for Eval

        self.collision_list = [] # list of collision objects for curobo planner
        self._init_collision_metrics()

        self.load_robot(**kwags)
        self.create_static_elements(table_xy_bias=table_xy_bias, table_height=0.74)
        self.load_camera(**kwags)
        self.robot.move_to_homestate()

        render_freq = self.render_freq
        self.render_freq = 0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

        self.robot.set_origin_endpose()
        self.load_actors()
        self.load_basic_office_items()

        if self.cluttered_table:
            self.get_cluttered_surfaces()

        self._build_collision_name_sets() # build collision name sets for collision metrics

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
        # create static furniture elements
        self.table_xy_bias = table_xy_bias
        wall_texture, table_texture, floor_texture = None, None, None
        table_height += self.table_z_bias

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

        shelf_scale = [0.6, 0.87, 0.4]
        self.shelf = create_multiple_obj_actor(
            scene=self.scene,
            pose=sapien.Pose(p=[0.9, -0.42, -0.07], q=[0.5, 0.5, 0.5, 0.5]),
            visual_path=f"{os.environ['ROBOTWIN_ROOT']}/assets/objects_bench/120_storage-rack/storage_rack_02.gltf",
            collision_path=f"{os.environ['ROBOTWIN_ROOT']}/assets/objects_bench/120_storage-rack/rack_convex",
            scale=shelf_scale,
            is_static=True,
            name="shelf"
        )
        self.collision_list.append((self.shelf, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects_bench/120_storage-rack/rack_convex", shelf_scale))
    
    def load_basic_office_items(self):
        # load office items: items that are always placed as obstacles ie key obstacles
        size_dict = list()
        if "screen" not in self.key_objects:
            screen_id = np.random.choice([0, 1, 2, 3])
            for i in range(10):
                success, self.screen = rand_create_cluttered_actor(
                    scene=self.scene,
                    xlim=[-0.2,0.2],
                    ylim=[0.05,0.22],
                    zlim=[0.741],
                    modelname="097_screen",
                    modelid=screen_id,
                    modeltype="glb",
                    qpos=[0.7071, 0.7071, 0, 0],
                    rotate_rand=True,
                    rotate_lim=[0, 0.5, 0],
                    size_dict=size_dict,
                    obj_radius=0.08,
                    z_offset=0,
                    z_max=0.35,
                    prohibited_area=self.prohibited_area["table"],
                    constrained=False,
                    is_static=False,
                )
                if success:
                    break
            if success:
                self.screen.set_mass(0.1)
                self.add_prohibit_area(self.screen, padding=0.01)
                self.collision_list.append((self.screen, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/097_screen/collision/base{screen_id}.glb", [1,1,1]))
        # ------------------------------------------------------------
        if "plant" not in self.key_objects:
            plant_id = 0
            for i in range(10):
                success, self.plant = rand_create_cluttered_actor(
                    scene=self.scene,
                    xlim=[-0.35,0.35],
                    ylim=[0.2, 0.3],
                    zlim=[0.741],
                    modelname="120_plant",
                    modelid=plant_id,
                    modeltype="glb",
                    qpos=[0.5, 0.5, 0.5, 0.5],
                    rotate_rand=True,
                    rotate_lim=[0, 1, 0],
                    size_dict=size_dict,
                    obj_radius=0.03,
                    z_offset=0,
                    z_max=0.25,
                    prohibited_area=self.prohibited_area["table"],
                    constrained=False,
                    is_static=False,
                )
                if success:
                    break
            if success:
                self.plant.set_mass(1)
                self.add_prohibit_area(self.plant, padding=0.01)
                self.collision_list.append((self.plant, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/120_plant/collision/base{plant_id}.glb", [1,1,1]))
        # ------------------------------------------------------------
        if "keyboard" not in self.key_objects:
            keyboard_id = np.random.choice([0, 1, 2, 3])
            for i in range(10):
                success, self.keyboard = rand_create_cluttered_actor(
                    scene=self.scene,
                    xlim=[-0.2,0.2],
                    ylim=[-0.3,-0.05],
                    zlim=[0.741],
                    modelname="116_keyboard",
                    modelid=keyboard_id,
                    modeltype="glb",
                    qpos=[0.7071, 0.7071, 0, 0],
                    rotate_rand=True,
                    rotate_lim=[0, 0.5, 0],
                    size_dict=size_dict,
                    obj_radius=0.12,
                    z_offset=0,
                    z_max=0.03,
                    prohibited_area=self.prohibited_area["table"],
                    constrained=False,
                    is_static=False,
                )
                if success:
                    break
            if success:
                self.keyboard.set_mass(0.1)
                self.add_prohibit_area(self.keyboard, padding=0.01)
                self.collision_list.append((self.keyboard, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/116_keyboard/collision/base{keyboard_id}.glb", [1,1,1]))
        # # ------------------------------------------------------------
        # if "mouse" not in self.key_objects:
        #     mouse_id = np.random.choice([0, 1, 2], 1)[0]
        #     success = False
        #     for i in range(10):
        #         success, self.mouse = rand_create_cluttered_actor(
        #             scene=self.scene,
        #             xlim=[-0.3,0.3],
        #             ylim=[-0.25,-0.05],
        #             zlim=[0.741],
        #             modelname="047_mouse",
        #             modelid=mouse_id,
        #             modeltype="glb",
        #             qpos=[0.0, 0.0, 0.7071, 0.7071],
        #             rotate_rand=True,
        #             rotate_lim=[0, 1, 0],
        #             size_dict=size_dict,
        #             obj_radius=0.045,
        #             z_offset=0,
        #             z_max=0.04,
        #             prohibited_area=self.prohibited_area["table"],
        #             constrained=False,
        #             is_static=False,
        #         )
        #         if success:
        #             break
        #     if success:
        #         self.mouse.set_mass(0.05)
        #         self.add_prohibit_area(self.mouse, padding=0.01)
        #         self.collision_list.append((self.mouse, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/047_mouse/collision/base{mouse_id}.glb", [1,1,1]))
        # # ------------------------------------------------------------
        # if "phone_stand" not in self.key_objects:
        #     stand_id = np.random.choice([0,1,2,3,4,5,6], 1)[0]
        #     success = False
        #     for i in range(10):
        #         success, self.phone_stand = rand_create_cluttered_actor(
        #             scene=self.scene,
        #             xlim=[-0.3,0.3],
        #             ylim=[-0.25,-0.05],
        #             zlim=[0.741],
        #             modelname="078_phonestand",
        #             modelid=stand_id,
        #             modeltype="glb",
        #             qpos=[0.7071, 0.7071, 0.0, 0.0],
        #             rotate_rand=True,
        #             rotate_lim=[0, 1, 0],
        #             size_dict=size_dict,
        #             obj_radius=0.04,
        #             z_offset=0,
        #             z_max=0.1,
        #             prohibited_area=self.prohibited_area["table"],
        #             constrained=False,
        #             is_static=False,
        #         )
        #         if success:
        #             break
        #     if success:
        #         self.phone_stand.set_mass(0.05)
        #         self.add_prohibit_area(self.phone_stand, padding=0.01)
        #         self.collision_list.append((self.phone_stand, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/078_phonestand/collision/base{stand_id}.glb", [1,1,1]))
    
    def get_cluttered_surfaces(self):
        # clutter surfaces with additional random obstacles
        # table ------------------------------------------------------
        xlim = [-0.59, 0.59]
        ylim = [-0.34, 0.34]
        zlim = [0.741]
        xlim[0] += self.table_xy_bias[0]
        xlim[1] += self.table_xy_bias[0]
        ylim[0] += self.table_xy_bias[1]
        ylim[1] += self.table_xy_bias[1]
        zlim = np.array(zlim) + self.table_z_bias
        if self.obstacle_height == "short":
            object_names = self.short_obstacles
        elif self.obstacle_height == "tall":
            object_names = self.tall_obstacles
        else:
            raise ValueError(f"Invalid obstacle height: {self.obstacle_height}")
        self.clutter_surface(xlim=xlim, ylim=ylim, zlim=zlim, object_names=object_names, prohibited_area=self.prohibited_area["table"], obstacle_count=self.obstacle_density)
        # shelves ----------------------------------------------------
        xlim = [self.shelf.get_pose().p[0]-0.1, self.shelf.get_pose().p[0]+0.1]
        ylim = [self.shelf.get_pose().p[1]-0.3, self.shelf.get_pose().p[1]+0.3]
        zlim = [self.shelf_heights[0]]
        self.clutter_surface(xlim=xlim, ylim=ylim, zlim=zlim, object_names=self.short_obstacles, prohibited_area=self.prohibited_area["shelf0"], obstacle_count=3)
        zlim = [self.shelf_heights[1]]
        self.clutter_surface(xlim=xlim, ylim=ylim, zlim=zlim, object_names=self.short_obstacles, prohibited_area=self.prohibited_area["shelf1"], obstacle_count=3)
    
    def load_table_obstacles_in_line(self, spacing=0.10, ground_y=-5.0, ground_z=0.02):
        """Load all table_obstacles that are available into the scene in a line on the ground at y=ground_y, spacing (m) apart. Used for debugging."""
        task_objects_list = []
        for entity in self.scene.get_all_actors():
            actor_name = entity.get_name()
            if actor_name == "":
                continue
            if actor_name in ["table", "wall", "ground"]:
                continue
            task_objects_list.append(actor_name)
        obj_names, cluttered_item_info = get_cluttered_objects_subset(self.tall_obstacles, task_objects_list)
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
            # self.stabilize_object(obj)
            # self.record_cluttered_objects.append({"object_type": obj_name, "object_index": obj_idx})
            # if model_type == "urdf":
            #     path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/objaverse/{obj_name}/{obj_idx}/coacd_collision.obj"
            # else:
            #     path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{obj_name}/collision/base{obj_idx}.glb"
            # self.collision_list.append((obj, path, obj.scale))
    
    def add_extra_cameras(self):
        self.cameras.add_extra_cameras(f"{os.environ['BENCH_ROOT']}/bench_assets/embodiments/office_config.yml")