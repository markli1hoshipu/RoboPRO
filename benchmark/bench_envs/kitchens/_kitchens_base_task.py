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


class KitchenS_base_task(Bench_base_task):

    FURNITURE_NAMES = {"table", "wall", "ground"}

    def __init__(self):
        pass

    def _init_task_env_(self, table_xy_bias=[0, 0], table_height_bias=0, **kwags):
        super().__init__()
        ta.setup_logging("CRITICAL")
        np.random.seed(kwags.get("seed", 0))
        torch.manual_seed(kwags.get("seed", 0))
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

        self.need_topp = True

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
        self.random_embodiment = random_setting.get("random_embodiment", False)
        self.obstacle_height = random_setting.get("obstacle_height", "short")
        self.obstacle_density = random_setting.get("obstacle_density", 3)

        self._parse_perturbation_config(kwags)

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
        self.prohibited_area = {"table": []}
        self.record_cluttered_objects = list()
        self.cluttered_objects_info = get_cluttered_objects_info()

        self.eval_success = False
        self.table_z_bias = 0

        self.scene_id = kwags.get("scene_id") if kwags.get("scene_id") is not None else np.random.randint(0, 3)
        print_c(f"KitchenS scene {self.scene_id} selected", "YELLOW")

        self.kitchens_info = {
            "table_height": 0.74,
            "table_area": [1.2, 0.7],
            "table_lims": [],
        }

        self.item_info = get_task_objects_config()
        self.target_objects_info = get_target_objects_subset("kitchens", self.sample_d)

        self.instruction = None
        self.collision_list = []
        self.cuboid_collision_list = []
        self._init_collision_metrics()

        self.need_plan = kwags.get("need_plan", True)
        self.left_joint_path = kwags.get("left_joint_path", [])
        self.right_joint_path = kwags.get("right_joint_path", [])
        self.left_cnt = 0
        self.right_cnt = 0

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
        self.add_gripper_operating_area()

        if self.cluttered_table:
            self.get_cluttered_surfaces()

        if self.enable_collision_metrics:
            self._build_collision_name_sets()

        is_stable, unstable_list = self.check_stable()
        if not is_stable:
            raise UnStableError(
                f'Objects is unstable in seed({kwags.get("seed", 0)}), unstable objects: {", ".join(unstable_list)}')

        self.update_world()

        if self.eval_mode:
            with open(os.path.join(os.environ["BENCH_ROOT"], "bench_task_config", "_bench_eval_step_limit.yml"), "r") as f:
                try:
                    data = yaml.safe_load(f)
                    self.step_lim = data[self.task_name]
                except:
                    print(f"{self.task_name} not in step limit file, set to 1000")
                    self.step_lim = 1000

        self.info = dict()
        self.info["cluttered_table_info"] = self.record_cluttered_objects
        self.info["texture_info"] = {
            "wall_texture": self.wall_texture,
            "table_texture": self.table_texture,
            "floor_texture": self.floor_texture,
        }
        self.info["info"] = {}

        self.stage_success_tag = False

    # ------------------------------------------------------------------
    # Scene layout helpers
    # ------------------------------------------------------------------

    def _get_scene_obj_locations(self, object_name="microwave"):
        """Return [x, y] positions for kitchen-S fixtures based on scene_id.

        scene_0: MW left, Dishrack center, Sink right
        scene_1: MW left, Sink center, Dishrack right
        scene_2: MW left, Dishrack front-center rotated, Sink right
        """
        if self.scene_id == 0:
            locations = {
                "microwave": [-0.32, 0.18],
                "dishrack": [0.10, 0.25],
                "sink": [0.42, 0.08],
            }
        elif self.scene_id == 1:
            locations = {
                "microwave": [-0.32, 0.18],
                "dishrack": [0.42, 0.25],
                "sink": [0.10, 0.08],
            }
        elif self.scene_id == 2:
            locations = {
                "microwave": [-0.32, 0.18],
                "dishrack": [0.10, 0.05],
                "sink": [0.42, 0.08],
            }
        else:
            raise ValueError(f"Invalid scene_id {self.scene_id}")

        if object_name not in locations:
            raise ValueError(f"Unknown object_name '{object_name}', expected one of {list(locations.keys())}")
        return locations[object_name]

    # ------------------------------------------------------------------
    # Static scene construction
    # ------------------------------------------------------------------

    def create_static_elements(self, table_xy_bias=[0, 0]):
        self.table_xy_bias = table_xy_bias
        table_height = self.kitchens_info["table_height"] + self.table_z_bias

        # Textures -------------------------------------------------------
        if self.random_background:
            texture_type = "seen" if not self.eval_mode else "unseen"
            directory_path = f"./assets/background_texture/{texture_type}"
            file_count = len(
                [name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])

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

        # Floor ----------------------------------------------------------
        self.floor_parts = []
        positions = [
            [1, 1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
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

        # Wall -----------------------------------------------------------
        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=(1, 0.9, 0.9),
            name="wall",
            texture_id=self.wall_texture,
            is_static=True,
        )

        # Countertop -----------------------------------------------------
        counter_length = self.kitchens_info["table_area"][0]
        counter_width = self.kitchens_info["table_area"][1]
        counter_thickness = 0.04

        self.table = create_table(
            self.scene,
            sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], table_height]),
            length=counter_length,
            width=counter_width,
            height=table_height,
            thickness=counter_thickness,
            is_static=True,
            texture_id=self.table_texture,
        )

        self.kitchens_info["table_lims"] = [
            -counter_length / 2, -counter_width / 2,
            counter_length / 2, counter_width / 2,
        ]
        self.cuboid_collision_list.append({
            "name": "table",
            "dims": [counter_length, counter_width, 0.002],
            "pose": [0, 0, table_height, 1, 0, 0, 0],
        })

        # Kitchen-S fixtures ---------------------------------------------
        self._load_microwave(table_height, table_xy_bias)
        self._load_dishrack(table_height, table_xy_bias)
        self._load_sink(table_height, table_xy_bias)

        if self.incl_collision:
            self.add_collision()

    # ------------------------------------------------------------------
    # Fixture loaders
    # ------------------------------------------------------------------

    def _load_microwave(self, table_height, table_xy_bias):
        x, y = self._get_scene_obj_locations("microwave")
        x += table_xy_bias[0]
        y += table_xy_bias[1]
        z = table_height + 0.06

        quat = euler2quat(0, 0, np.pi / 2, axes='sxyz')
        pose = sapien.Pose([x, y, z], [quat[0], quat[1], quat[2], quat[3]])

        try:
            self.microwave = create_sapien_urdf_obj(
                scene=self,
                pose=pose,
                modelname="044_microwave",
                modelid=0,
                fix_root_link=True,
            )
        except Exception as e:
            print(f"[KitchenS] failed to load microwave URDF: {e}")
            self.microwave = None
            return

        self.microwave.set_name("microwave")
        self.add_prohibit_area(self.microwave, padding=0.02, area="table")
        self.collision_list.append({
            "actor": self.microwave,
            "collision_path": f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/044_microwave/visual/base0.glb",
        })

    def _load_dishrack(self, table_height, table_xy_bias):
        x, y = self._get_scene_obj_locations("dishrack")
        x += table_xy_bias[0]
        y += table_xy_bias[1]
        z = table_height

        if self.scene_id == 2:
            q = euler2quat(-np.pi / 2, 0, np.pi / 4, axes='sxyz')
        else:
            q = euler2quat(-np.pi / 2, 0, 0, axes='sxyz')

        self.dishrack = create_glb_actor(
            scene=self.scene,
            pose=sapien.Pose(p=[x, y, z], q=[q[0], q[1], q[2], q[3]]),
            model_name="135_dish-rack",
            scale=[1, 1, 1],
            convex=False,
            is_static=True,
            mass=2,
        )
        self.dishrack.set_name("dishrack")
        self.add_prohibit_area(self.dishrack, padding=0.02, area="table")
        self.collision_list.append({
            "actor": self.dishrack,
            "collision_path": f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/135_dish-rack/collision/base0.glb",
        })

    def _load_sink(self, table_height, table_xy_bias):
        x, y = self._get_scene_obj_locations("sink")
        x += table_xy_bias[0]
        y += table_xy_bias[1]

        hole_hx = 0.13
        hole_hy = 0.20
        depth = 0.09
        inner_hx = 0.12
        inner_hy = 0.19

        sink_z = table_height
        wall_thickness = hole_hx - inner_hx

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")

        material = sapien.render.RenderMaterial(base_color=[0.75, 0.75, 0.78, 1.0])
        material.metallic = 0.6
        material.roughness = 0.3

        # Bottom
        bottom_half = [inner_hx, inner_hy, wall_thickness / 2]
        bottom_pose = sapien.Pose([0, 0, -depth + wall_thickness / 2])
        builder.add_box_collision(pose=bottom_pose, half_size=bottom_half)
        builder.add_box_visual(pose=bottom_pose, half_size=bottom_half, material=material)

        # Four walls
        walls = [
            ([hole_hx - wall_thickness / 2, 0, -depth / 2], [wall_thickness / 2, hole_hy, depth / 2]),  # +x
            ([-(hole_hx - wall_thickness / 2), 0, -depth / 2], [wall_thickness / 2, hole_hy, depth / 2]),  # -x
            ([0, hole_hy - wall_thickness / 2, -depth / 2], [inner_hx, wall_thickness / 2, depth / 2]),  # +y
            ([0, -(hole_hy - wall_thickness / 2), -depth / 2], [inner_hx, wall_thickness / 2, depth / 2]),  # -y
        ]
        for w_pos, w_half in walls:
            wp = sapien.Pose(w_pos)
            builder.add_box_collision(pose=wp, half_size=w_half)
            builder.add_box_visual(pose=wp, half_size=w_half, material=material)

        builder.set_initial_pose(sapien.Pose(p=[x, y, sink_z]))
        self.sink = builder.build(name="sink")

        self.prohibited_area["table"].append([
            x - hole_hx - 0.02, y - hole_hy - 0.02,
            x + hole_hx + 0.02, y + hole_hy + 0.02,
        ])

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------

    def add_extra_cameras(self):
        self.cameras.add_extra_cameras(f"{os.environ['BENCH_ROOT']}/bench_assets/embodiments/kitchen_s_config.yml")

    # ------------------------------------------------------------------
    # Clutter
    # ------------------------------------------------------------------

    def get_cluttered_surfaces(self):
        xlim = [self.kitchens_info["table_lims"][0], self.kitchens_info["table_lims"][2]]
        ylim = [self.kitchens_info["table_lims"][1], self.kitchens_info["table_lims"][3]]
        zlim = [self.kitchens_info["table_height"]]
        xlim[0] += self.table_xy_bias[0]
        xlim[1] += self.table_xy_bias[0]
        ylim[0] += self.table_xy_bias[1]
        ylim[1] += self.table_xy_bias[1]
        zlim = np.array(zlim) + self.table_z_bias

        task_objects_list = []
        for entity in self.scene.get_all_actors():
            actor_name = entity.get_name()
            if actor_name == "" or actor_name in self.FURNITURE_NAMES:
                continue
            task_objects_list.append(actor_name)

        cluttered_item_info, obj_names_short, obj_names_tall = get_obstacle_objects_subset(
            "kitchens", self.sample_d, task_objects_list
        )

        self.clutter_surface_split(
            xlim, ylim, zlim,
            self.prohibited_area["table"],
            self.obstacle_density,
            cluttered_item_info,
            obj_names_short,
            obj_names_tall,
        )

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------

    def enable_table(self, enable: bool):
        names = [f"table_[0, 0, 0.74, 1, 0, 0, 0]_{self.seed}"]
        self.enable_obstacle(enable, obb_names=names)

    def grasp_actor_from_table(
        self,
        actor: Actor,
        arm_tag: ArmTag,
        pre_grasp_dis=0.1,
        grasp_dis=0,
        gripper_pos=0.0,
        contact_point_id: list | float = None,
    ):
        _, actions = self.grasp_actor(
            actor=actor, arm_tag=arm_tag,
            pre_grasp_dis=pre_grasp_dis, grasp_dis=grasp_dis,
            gripper_pos=gripper_pos, contact_point_id=contact_point_id,
        )
        self.move((arm_tag, [actions[0]]))
        self.enable_table(enable=False)
        self.move((arm_tag, actions[1:]))

    # ------------------------------------------------------------------
    # Sink query helpers (for task success checking)
    # ------------------------------------------------------------------

    def is_object_in_sink(self, actor, margin=0.01):
        sx, sy = self._get_scene_obj_locations("sink")
        sx += self.table_xy_bias[0]
        sy += self.table_xy_bias[1]
        hole_hx, hole_hy = 0.13, 0.20

        pos = actor.get_pose().p
        return (
            sx - hole_hx + margin < pos[0] < sx + hole_hx - margin
            and sy - hole_hy + margin < pos[1] < sy + hole_hy - margin
        )

    def is_object_on_dishrack(self, actor, margin=0.02):
        dx, dy = self._get_scene_obj_locations("dishrack")
        dx += self.table_xy_bias[0]
        dy += self.table_xy_bias[1]
        rack_hx, rack_hy = 0.10, 0.08

        pos = actor.get_pose().p
        return (
            dx - rack_hx - margin < pos[0] < dx + rack_hx + margin
            and dy - rack_hy - margin < pos[1] < dy + rack_hy + margin
            and pos[2] > self.kitchens_info["table_height"] + 0.01
        )
