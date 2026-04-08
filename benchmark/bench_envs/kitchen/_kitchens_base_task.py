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


class KitchenS_base_task(Bench_base_task):
    # Layout 1 (odd seeds):  MW left, Dishrack center, Sink right
    # Layout 2 (even seeds): MW left, Sink center, Dishrack right
    ZONE_C = {"xlim": [-0.60, -0.20], "ylim": [0.05, 0.35]}    # Microwave (left, both layouts)
    ZONE_DEF = {"xlim": [-0.60, 0.60], "ylim": [-0.20, 0.05]}  # Task objects

    def __init__(self):
        pass

    def _init_task_env_(self, table_xy_bias=[0, 0], table_height_bias=0, **kwags):
        """
        Kitchen S environment initialization.
        Layout: Sink (left) - Countertop (center) - Range (right)
        Microwave on counter rear-center.
        """
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

        self.need_topp = True

        # Random
        random_setting = kwags.get("domain_randomization") or {}
        self.random_background = random_setting.get("random_background", False)
        self.cluttered_table = random_setting.get("cluttered_table", False)
        self.clean_background_rate = random_setting.get("clean_background_rate", 1)
        self.random_head_camera_dis = random_setting.get("random_head_camera_dis", 0)
        self.random_table_height = random_setting.get("random_table_height", 0)
        self.random_light = random_setting.get("random_light", False)
        self.crazy_random_light_rate = random_setting.get("crazy_random_light_rate", 0)
        self.crazy_random_light = (0 if not self.random_light else np.random.rand() < self.crazy_random_light_rate)
        self.random_embodiment = random_setting.get("random_embodiment", False)

        self.file_path = []
        self.plan_success = True
        self.step_lim = None
        self.fix_gripper = False
        self.setup_scene()

        # Override viewer camera to match countertop camera view
        if self.render_freq:
            self.viewer.set_camera_xyz(x=0.0, y=-0.4, z=2.0)
            self.viewer.set_camera_rpy(r=0, p=-1.1071, y=-1.5708)

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
        # Kitchen uses "counter" instead of "shelf"
        self.prohibited_area = {"table": [], "range": [], "sink": []}
        self.record_cluttered_objects = list()

        self.eval_success = False
        self.table_z_bias = (np.random.uniform(low=-self.random_table_height, high=0) + table_height_bias)
        self.need_plan = kwags.get("need_plan", True)
        self.left_joint_path = kwags.get("left_joint_path", [])
        self.right_joint_path = kwags.get("right_joint_path", [])
        self.left_cnt = 0
        self.right_cnt = 0

        self.instruction = None

        self.collision_list = []

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

        if os.environ.get("DEBUG_PROHIBIT", ""):
            self._visualize_prohibited_areas()

        if self.cluttered_table:
            self.get_cluttered_surfaces()

        is_stable, unstable_list = self.check_stable()
        if not is_stable:
            raise UnStableError(
                f'Objects is unstable in seed({kwags.get("seed", 0)}), unstable objects: {", ".join(unstable_list)}')

        # Reject initial states where check_success is already True
        if hasattr(self, 'check_success') and self.check_success():
            raise ValueError(
                f'Bad initial state in seed({kwags.get("seed", 0)}): check_success=True before task starts')

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

        if self.random_background:
            texture_type = "seen" if not self.eval_mode else "unseen"
            directory_path = f"./assets/background_texture/{texture_type}"
            file_count = len(
                [name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])

            if texture_type == "seen":
                wall_texture = np.random.randint(0, file_count)
                table_texture = np.random.choice([0, 2, 4, 5, 7, 9, 14, 16, 18, 19])
                floor_texture = np.random.choice([2, 3, 4, 5, 6, 17, 47, 71, 110])
            else:
                wall_texture = np.random.randint(0, file_count)
                table_texture = np.random.choice([1, 8, 9, 27, 29, 30, 37, 55])
                floor_texture = np.random.choice([0, 6, 8, 23, 34, 41, 47, 61, 64])

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

        # ── Countertop dimensions (same as original RoboTwin table) ──
        counter_length = 1.2   # X extent
        counter_width = 0.7    # Y extent (depth)
        counter_thickness = 0.04
        counter_cx = table_xy_bias[0]
        counter_cy = table_xy_bias[1]

        # ── Floor — terracotta tile ──
        self.floor = create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[0, 0, 0]),
            half_size=[2, 2, 0.005],
            color=(0.72, 0.62, 0.52),  # warm terracotta tile
            name="floor",
            texture_id=self.floor_texture,
        )

        # ── Back wall — light cream ──
        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=(0.94, 0.92, 0.86),  # warm cream wall
            name="wall",
            texture_id=self.wall_texture,
            is_static=True,
        )

        # ── Backsplash — subway tile look (white with slight blue tint) ──
        backsplash_z = table_height + 0.22
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[counter_cx, counter_cy + counter_width / 2 + 0.005, backsplash_z]),
            half_size=[counter_length / 2 + 0.02, 0.006, 0.22],
            color=(0.92, 0.94, 0.96),  # white with cool tint — reads as tile
            name="backsplash",
        )

        # ── Countertop with sink hole — dark granite ──
        # The counter is split into 4 pieces around the sink opening so
        # objects can physically fall into the basin.
        counter_top_z = table_height - counter_thickness / 2
        th = counter_thickness / 2  # half-thickness

        # Scene selection: random from 3 layouts (matching study pattern)
        self.scene_id = np.random.randint(0, 3)
        self.layout = self.scene_id  # keep self.layout for backward compat
        print(f"\033[33mScene {self.scene_id} is selected\033[0m")

        # Sink hole dimensions (in world coords, before bias)
        sink_hx = 0.13  # half-width of hole in X
        sink_hy = 0.20  # half-width of hole in Y (extended)
        if self.scene_id == 0:
            # Scene 0: Sink right, Dishrack center
            sink_rel_x = 0.42
            sink_rel_y = 0.08
        elif self.scene_id == 1:
            # Scene 1: Sink center, Dishrack right
            sink_rel_x = 0.10
            sink_rel_y = 0.08
        else:
            # Scene 2: Sink right, Dishrack front-center rotated
            sink_rel_x = 0.42
            sink_rel_y = 0.08

        # Counter edges (relative to counter center = 0,0)
        cl = -counter_length / 2  # left edge
        cr = counter_length / 2   # right edge
        cf = -counter_width / 2   # front edge
        cb = counter_width / 2    # back edge

        # Hole edges (relative to counter center)
        hl = sink_rel_x - sink_hx
        hr = sink_rel_x + sink_hx
        hf = sink_rel_y - sink_hy
        hb = sink_rel_y + sink_hy

        # Build material
        if self.table_texture is not None:
            texturepath = f"./assets/background_texture/{self.table_texture}.png"
            texture2d = sapien.render.RenderTexture2D(texturepath)
            counter_mat = sapien.render.RenderMaterial()
            counter_mat.set_base_color_texture(texture2d)
            counter_mat.base_color = [1, 1, 1, 1]
            counter_mat.metallic = 0.1
            counter_mat.roughness = 0.3
        else:
            counter_mat = sapien.render.RenderMaterial(
                base_color=[0.28, 0.27, 0.26, 1])
            counter_mat.metallic = 0.12
            counter_mat.roughness = 0.22

        # 4 pieces around the sink hole:
        # 1) Right piece: from hole right edge to counter right edge (full Y)
        # 2) Left piece: from counter left edge to hole left edge (full Y)
        # 3) Front strip: between hole L/R, from counter front to hole front
        # 4) Back strip: between hole L/R, from hole back to counter back
        counter_pieces = [
            ("right", hr, cr, cf, cb),
            ("left",  cl, hl, cf, cb),
            ("front", hl, hr, cf, hf),
            ("back",  hl, hr, hb, cb),
        ]

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        for name, x1, x2, y1, y2 in counter_pieces:
            hx = (x2 - x1) / 2
            hy = (y2 - y1) / 2
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if hx <= 0 or hy <= 0:
                continue
            pose = sapien.Pose([cx, cy, 0])
            builder.add_box_collision(pose=pose, half_size=[hx, hy, th],
                                     material=self.scene.default_physical_material)
            builder.add_box_visual(pose=pose, half_size=[hx, hy, th], material=counter_mat)

        self.table = builder.build_static()
        self.table.set_pose(sapien.Pose(p=[counter_cx, counter_cy, counter_top_z]))
        self.table.set_name("table")

        # ── Base cabinets — warm oak wood ──
        cabinet_h = (table_height - counter_thickness) / 2
        cabinet_z = cabinet_h
        # Front panel
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[counter_cx, counter_cy - counter_width / 2 + 0.01, cabinet_z]),
            half_size=[counter_length / 2, 0.01, cabinet_h],
            color=(0.75, 0.60, 0.42),
            name="cabinet_front",
        )
        # Cabinet door divider lines (3 vertical lines on front panel)
        for dx in [-0.23, 0.0, 0.23]:
            create_visual_textured_box(
                self.scene,
                sapien.Pose(p=[counter_cx + dx, counter_cy - counter_width / 2 + 0.005, cabinet_z]),
                half_size=[0.003, 0.003, cabinet_h - 0.02],
                color=(0.55, 0.42, 0.28),
                name=f"cabinet_divider_{dx}",
            )
        # Cabinet handles (small horizontal bars on each door)
        for dx in [-0.35, -0.12, 0.12, 0.35]:
            handle_z = cabinet_z + 0.05
            create_visual_textured_box(
                self.scene,
                sapien.Pose(p=[counter_cx + dx, counter_cy - counter_width / 2 - 0.005, handle_z]),
                half_size=[0.025, 0.005, 0.004],
                color=(0.65, 0.65, 0.68),  # brushed nickel
                name=f"cabinet_handle_{dx}",
            )
        # Left side panel
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[counter_cx - counter_length / 2 + 0.01, counter_cy, cabinet_z]),
            half_size=[0.01, counter_width / 2, cabinet_h],
            color=(0.70, 0.56, 0.40),
            name="cabinet_left",
        )
        # Right side panel
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[counter_cx + counter_length / 2 - 0.01, counter_cy, cabinet_z]),
            half_size=[0.01, counter_width / 2, cabinet_h],
            color=(0.70, 0.56, 0.40),
            name="cabinet_right",
        )
        # Back panel
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[counter_cx, counter_cy + counter_width / 2 - 0.01, cabinet_z]),
            half_size=[counter_length / 2, 0.01, cabinet_h],
            color=(0.65, 0.52, 0.38),
            name="cabinet_back",
        )

        # ── Countertop edge trim (thin bright strip along front edge) ──
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[counter_cx, counter_cy - counter_width / 2, table_height - counter_thickness / 2]),
            half_size=[counter_length / 2, 0.003, counter_thickness / 2 + 0.002],
            color=(0.22, 0.21, 0.20),  # dark edge matching granite
            name="counter_edge",
        )

        # ── Microwave (real URDF, articulated with fixed root) ──
        # Randomize positions with collision avoidance (measured half-extents)
        min_gap = 0.05
        mw_half = np.array([0.056, 0.094])    # measured at sf=0.80
        rack_half = np.array([0.135, 0.108])   # from model_data extents at sf=1.17, rotated 90° around X
        sink_half = np.array([0.136, 0.200])   # measured sink footprint (extended Y)
        sink_cx = sink_rel_x + table_xy_bias[0]
        sink_cy = sink_rel_y + table_xy_bias[1]

        # Counter bounds
        cx_lo = -0.6 + table_xy_bias[0]
        cx_hi = 0.6 + table_xy_bias[0]
        cy_lo = -0.35 + table_xy_bias[1]
        cy_hi = 0.35 + table_xy_bias[1]

        # ── Furniture spawning (layout-dependent, no noise) ──
        # Microwave always on left
        mw_x = -0.32 + table_xy_bias[0]
        mw_y = 0.18 + table_xy_bias[1]
        mw_yaw = 0
        mw_base_yaw = np.pi / 2
        mw_q = t3d.quaternions.axangle2quat([0, 0, 1], mw_base_yaw + mw_yaw)

        if self.scene_id == 0:
            # Scene 0: Dishrack center, Sink right
            rack_x_cand = 0.10 + table_xy_bias[0]
            rack_y_cand = 0.25 + table_xy_bias[1]
            self._rack_yaw = 0
        elif self.scene_id == 1:
            # Scene 1: Dishrack right, Sink center
            rack_x_cand = 0.42 + table_xy_bias[0]
            rack_y_cand = 0.25 + table_xy_bias[1]
            self._rack_yaw = 0
        else:
            # Scene 2: Dishrack front-center, rotated 45°
            rack_x_cand = 0.10 + table_xy_bias[0]
            rack_y_cand = 0.05 + table_xy_bias[1]
            self._rack_yaw = np.deg2rad(45)
        self._rack_x_rand = rack_x_cand
        self._rack_y_rand = rack_y_cand
        mw_z = table_height + 0.06  # raised above table to avoid collision
        self.microwave_model_id = 0
        self.microwave = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="044_microwave",
            modelid=self.microwave_model_id,
            xlim=[mw_x, mw_x],
            ylim=[mw_y, mw_y],
            zlim=[mw_z],
            qpos=mw_q.tolist(),
            fix_root_link=True,
        )
        self.microwave.set_mass(0.01)
        self.microwave.set_properties(0.0, 0.0)
        # Store joint limits so tasks can reference them
        mw_limits = self.microwave.get_qlimits()
        self.microwave_joint_lower = mw_limits[0][0]
        self.microwave_joint_upper = mw_limits[0][1]
        self.microwave_joint_range = self.microwave_joint_upper - self.microwave_joint_lower
        self.add_prohibit_area(self.microwave, padding=0.04, area="table")

        # Debug: visualize object centers as tall poles
        if os.environ.get("DEBUG_PROHIBIT", ""):
            mw_pose = self.microwave.get_pose()
            create_visual_textured_box(
                self.scene,
                sapien.Pose(p=[mw_pose.p[0], mw_pose.p[1], mw_pose.p[2] + 0.15]),
                half_size=[0.005, 0.005, 0.15],
                color=(1.0, 0.0, 0.0),
                name="mw_center_pole",
            )

        # ── Sink — true recessed basin ──
        sink_x = sink_rel_x + table_xy_bias[0]
        sink_y = sink_rel_y + table_xy_bias[1]
        sink_depth = 0.09  # 9cm deep basin
        sink_inner_hx = 0.12  # inner basin half-width X
        sink_inner_hy = 0.19  # inner basin half-width Y (extended)
        sink_wall_t = 0.008   # wall thickness
        sink_mid_z = table_height - sink_depth / 2  # center height of walls
        self.sink_pose = sapien.Pose(p=[sink_x, sink_y, table_height - sink_depth])

        # Basin floor (collision + visual)
        self.sink_basin = create_box(
            self.scene,
            sapien.Pose(p=[sink_x, sink_y, table_height - sink_depth]),
            half_size=[sink_inner_hx, sink_inner_hy, 0.004],
            color=(0.75, 0.75, 0.78),
            name="sink_basin",
            is_static=True,
        )
        # Basin walls (collision boxes so objects stay inside)
        # Front wall
        create_box(
            self.scene,
            sapien.Pose(p=[sink_x, sink_y - sink_inner_hy, sink_mid_z]),
            half_size=[sink_inner_hx, sink_wall_t, sink_depth / 2],
            color=(0.72, 0.72, 0.75),
            name="sink_wall_front",
            is_static=True,
        )
        # Back wall
        create_box(
            self.scene,
            sapien.Pose(p=[sink_x, sink_y + sink_inner_hy, sink_mid_z]),
            half_size=[sink_inner_hx, sink_wall_t, sink_depth / 2],
            color=(0.72, 0.72, 0.75),
            name="sink_wall_back",
            is_static=True,
        )
        # Left wall
        create_box(
            self.scene,
            sapien.Pose(p=[sink_x - sink_inner_hx, sink_y, sink_mid_z]),
            half_size=[sink_wall_t, sink_inner_hy, sink_depth / 2],
            color=(0.72, 0.72, 0.75),
            name="sink_wall_left",
            is_static=True,
        )
        # Right wall
        create_box(
            self.scene,
            sapien.Pose(p=[sink_x + sink_inner_hx, sink_y, sink_mid_z]),
            half_size=[sink_wall_t, sink_inner_hy, sink_depth / 2],
            color=(0.72, 0.72, 0.75),
            name="sink_wall_right",
            is_static=True,
        )
        # Sink rim — 4 thin chrome strips forming a frame around the opening
        rim_t = 0.012  # rim strip width
        rim_z = table_height + 0.003
        rim_hz = 0.003
        # Front rim
        create_box(self.scene,
            sapien.Pose(p=[sink_x, sink_y - sink_hy - rim_t / 2, rim_z]),
            half_size=[sink_hx + rim_t, rim_t / 2, rim_hz],
            color=(0.82, 0.82, 0.85), name="sink_rim_front", is_static=True)
        # Back rim
        create_box(self.scene,
            sapien.Pose(p=[sink_x, sink_y + sink_hy + rim_t / 2, rim_z]),
            half_size=[sink_hx + rim_t, rim_t / 2, rim_hz],
            color=(0.82, 0.82, 0.85), name="sink_rim_back", is_static=True)
        # Left rim
        create_box(self.scene,
            sapien.Pose(p=[sink_x - sink_hx - rim_t / 2, sink_y, rim_z]),
            half_size=[rim_t / 2, sink_hy, rim_hz],
            color=(0.82, 0.82, 0.85), name="sink_rim_left", is_static=True)
        # Right rim
        create_box(self.scene,
            sapien.Pose(p=[sink_x + sink_hx + rim_t / 2, sink_y, rim_z]),
            half_size=[rim_t / 2, sink_hy, rim_hz],
            color=(0.82, 0.82, 0.85), name="sink_rim_right", is_static=True)
        # Faucet — tall post + gooseneck spout, at back edge of sink
        faucet_y = sink_y + sink_inner_hy + 0.02
        faucet_z = table_height + 0.10
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[sink_x, faucet_y, faucet_z]),
            half_size=[0.010, 0.010, 0.10],
            color=(0.82, 0.82, 0.85),
            name="faucet_post",
        )
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[sink_x, faucet_y - 0.06, table_height + 0.19]),
            half_size=[0.008, 0.07, 0.008],
            color=(0.82, 0.82, 0.85),
            name="faucet_spout",
        )
        create_visual_textured_box(
            self.scene,
            sapien.Pose(p=[sink_x + 0.04, faucet_y, table_height + 0.06]),
            half_size=[0.012, 0.006, 0.015],
            color=(0.82, 0.82, 0.85),
            name="faucet_handle",
        )
        # Sink prohibited area — basin + walls + faucet
        sink_pad = 0.05
        self.prohibited_area["table"].append([
            sink_x - 0.14 - sink_pad, sink_y - sink_inner_hy - sink_pad,
            sink_x + 0.14 + sink_pad, faucet_y + sink_pad,
        ])

        # ── Dish rack — scale set in model_data0.json ──
        rack_x = self._rack_x_rand
        rack_y = self._rack_y_rand
        rack_z = table_height + 0.148 * 1.17
        rack_base_q = np.array([0.707, 0.707, 0, 0])  # wxyz
        rack_yaw_q = t3d.quaternions.axangle2quat([0, 0, 1], self._rack_yaw)
        rack_q = t3d.quaternions.qmult(rack_yaw_q, rack_base_q)
        self.dish_rack = create_actor(
            scene=self,
            pose=sapien.Pose(p=[rack_x, rack_y, rack_z], q=rack_q.tolist()),
            modelname="135_dish-rack",
            convex=True,
            model_id=0,
            is_static=True,
        )
        # Manual prohibited rect — model_data extents are inaccurate for this mesh.
        # Actual mesh half-extents at sf=1.17: [0.135, 0.108] on XY after 90° X rotation.
        self.add_prohibit_area(self.dish_rack, padding=0.04, area="table")

        if os.environ.get("DEBUG_PROHIBIT", ""):
            rack_pose = self.dish_rack.get_pose()
            create_visual_textured_box(
                self.scene,
                sapien.Pose(p=[rack_pose.p[0], rack_pose.p[1], rack_pose.p[2] + 0.15]),
                half_size=[0.005, 0.005, 0.15],
                color=(0.0, 1.0, 0.0),
                name="rack_center_pole",
            )

        # Robot arm zone (y < -0.20) is excluded by the DEF zone ylim directly,
        # so no prohibited area needed. Adding one here would cause obj_radius
        # expansion to eat into the valid DEF zone spawn area.

        # (Utensil holder, cutting board, breadbasket, trash bin removed from
        #  static scene — tasks that need them spawn as task-specific objects.)

        # ── Side shelf removed ──

        # ── Upper open shelves (decorative, pushed to back wall) ──
        for shelf_side, sx, sw in [("right", 0.28, 0.48), ("left", -0.28, 0.40)]:
            shelf_base_z = table_height + 0.32
            shelf_total_h = 0.38
            shelf_d = 0.10  # thin depth — decoration only, not for tasks
            shelf_back_y = counter_cy + counter_width / 2 - 0.01
            shelf_front_y = shelf_back_y - shelf_d
            shelf_center_y = (shelf_back_y + shelf_front_y) / 2
            plank_thickness = 0.012

            # Back panel (thin, against the wall)
            create_visual_textured_box(
                self.scene,
                sapien.Pose(p=[counter_cx + sx, shelf_back_y, shelf_base_z + shelf_total_h / 2]),
                half_size=[sw / 2, 0.004, shelf_total_h / 2],
                color=(0.68, 0.54, 0.38),  # darker wood back
                name=f"shelf_{shelf_side}_back",
            )
            # Left side panel
            create_visual_textured_box(
                self.scene,
                sapien.Pose(p=[counter_cx + sx - sw / 2, shelf_center_y, shelf_base_z + shelf_total_h / 2]),
                half_size=[plank_thickness / 2, shelf_d / 2, shelf_total_h / 2],
                color=(0.75, 0.60, 0.42),
                name=f"shelf_{shelf_side}_left",
            )
            # Right side panel
            create_visual_textured_box(
                self.scene,
                sapien.Pose(p=[counter_cx + sx + sw / 2, shelf_center_y, shelf_base_z + shelf_total_h / 2]),
                half_size=[plank_thickness / 2, shelf_d / 2, shelf_total_h / 2],
                color=(0.75, 0.60, 0.42),
                name=f"shelf_{shelf_side}_right",
            )
            # Horizontal shelf boards (bottom, middle, top)
            for i, frac in enumerate([0.0, 0.5, 1.0]):
                board_z = shelf_base_z + frac * shelf_total_h
                create_visual_textured_box(
                    self.scene,
                    sapien.Pose(p=[counter_cx + sx, shelf_center_y, board_z]),
                    half_size=[sw / 2, shelf_d / 2, plank_thickness / 2],
                    color=(0.75, 0.60, 0.42),
                    name=f"shelf_{shelf_side}_board_{i}",
                )

    def update_world(self):
        """Override to include MW collision cuboid alongside mesh collision objects."""
        # Build collision_dict the same way as base class
        collision_dict = {"cuboid": {}, "mesh": {}}
        for actor, collision_path, scale in self.collision_list:
            if type(actor) == ArticulationActor or type(actor) == Actor:
                pose = actor.get_pose()
                np_pose = np.concatenate([pose.p, pose.q]).tolist()
                collision_dict["mesh"][f"{actor.get_name()}_{self.seed}"] = {
                    "file_path": collision_path,
                    "pose": np_pose,
                    "scale": scale,
                }
        # Add MW collision cuboid
        if hasattr(self, '_mw_collision'):
            mc = self._mw_collision
            p = mc["pose"]  # [x, y, z, qw, qx, qy, qz]
            collision_dict["cuboid"]["microwave_box"] = {
                "dims": mc["dims"],
                "pose": [p[0], p[1], p[2], 1, 0, 0, 0],  # wxyz identity quat
            }
        if hasattr(self.robot, 'update_world'):
            self.robot.update_world(collision_dict)

    # ── Debug: visualize prohibited areas ──

    def _visualize_prohibited_areas(self, area="table", z=None):
        """Draw semi-transparent colored boxes on the counter for each prohibited rectangle.
        Call after load_actors() to see both furniture and task-object prohibited zones.
        """
        if z is None:
            z = 0.74 + self.table_z_bias + 0.002  # just above counter surface

        colors = [
            (1.0, 0.0, 0.0),  # red
            (0.0, 0.0, 1.0),  # blue
            (0.0, 0.8, 0.0),  # green
            (1.0, 0.6, 0.0),  # orange
            (0.8, 0.0, 0.8),  # purple
            (0.0, 0.8, 0.8),  # cyan
            (1.0, 1.0, 0.0),  # yellow
            (0.6, 0.3, 0.0),  # brown
        ]

        for i, rect in enumerate(self.prohibited_area.get(area, [])):
            x_min, y_min, x_max, y_max = rect
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            hx = (x_max - x_min) / 2
            hy = (y_max - y_min) / 2
            hz = 0.002  # thin slab

            color = colors[i % len(colors)]
            material = sapien.render.RenderMaterial(
                base_color=[*color, 0.35],  # semi-transparent
            )
            material.metallic = 0.0
            material.roughness = 1.0

            entity = sapien.Entity()
            entity.set_name(f"prohib_debug_{area}_{i}")
            render_comp = sapien.render.RenderBodyComponent()
            render_comp.attach(sapien.render.RenderShapeBox([hx, hy, hz], material))
            entity.add_component(render_comp)
            entity.set_pose(sapien.Pose(p=[cx, cy, z]))
            self.scene.add_entity(entity)

    # ── Collision-free spawn helper ──

    def _point_in_prohibited(self, x, y, area="table", radius=0.0):
        """Return True if a disc at (x, y) with given radius overlaps any prohibited rectangle.
        When radius > 0 the prohibited rects are expanded by radius on each side,
        so the check accounts for the object's physical extent, not just its center.
        """
        for rect in self.prohibited_area.get(area, []):
            if ((rect[0] - radius) <= x <= (rect[2] + radius) and
                    (rect[1] - radius) <= y <= (rect[3] + radius)):
                return True
        return False

    def _safe_rand_pose(self, xlim, ylim, zlim, max_tries=200, obj_radius=0.06, **kwargs):
        """rand_pose that retries until the position avoids prohibited areas.
        obj_radius: approximate radius of the object being placed (default 0.06m).
            Expands prohibited rectangles so an object's physical extent — not just
            its center — is kept clear of furniture and other placed objects.
        """
        for _ in range(max_tries):
            pose = rand_pose(xlim=xlim, ylim=ylim, zlim=zlim, **kwargs)
            if not self._point_in_prohibited(pose.p[0], pose.p[1], radius=obj_radius):
                return pose
        # Last attempt — return it even if overlapping
        return pose

    # ── Microwave cavity helper ──

    def _get_microwave_cavity_center(self):
        """Return [x, y, z] of the microwave interior cavity center in world coords.
        Manual offset tuned for sf=0.80 with base yaw π/2.
        """
        mw_p = self.microwave.get_pose().p
        return [mw_p[0] - 0.03, mw_p[1] - 0.20, mw_p[2]]

    # ── Planner collision helpers ──

    def _disable_planner_table(self, arm_tag):
        """Remove table collision from CuRobo so arm can reach into sink."""
        planner = self.robot.left_planner if str(arm_tag) == "left" else self.robot.right_planner
        try:
            planner.motion_gen.clear_world_cache()
            planner.motion_gen.update_world(planner.motion_gen.world_coll_checker.world_model)
        except Exception as e:
            print(f"[_disable_planner_table] warning: {e}")

    def _disable_planner_mw(self, arm_tag):
        """Remove MW collision cuboid from CuRobo so gripper can enter cavity."""
        if not hasattr(self, '_mw_collision'):
            return
        planner = self.robot.left_planner if str(arm_tag) == "left" else self.robot.right_planner
        try:
            planner.motion_gen.clear_world_cache()
            planner.motion_gen.update_world(planner.motion_gen.world_coll_checker.world_model)
        except Exception as e:
            print(f"[_disable_planner_mw] warning: {e}")

    def get_cluttered_surfaces(self):
        """Place clutter objects on the countertop only (avoid range and sink zones)."""
        self.get_cluttered_table()

    def get_cluttered_table(self, cluttered_numbers=8, xlim=[-0.25, 0.25], ylim=[-0.34, 0.10], zlim=[0.741]):
        self.record_cluttered_objects = []

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
            if actor_name in ["table", "wall", "ground", "sink_basin", "sink_rim", "floor"]:
                continue
            task_objects_list.append(actor_name)
        self.obj_names, self.cluttered_item_info = get_available_cluttered_objects(task_objects_list)

        success_count = 0
        max_try = 50
        trys = 0

        while success_count < cluttered_numbers and trys < max_try:
            obj = np.random.randint(len(self.obj_names))
            obj_name = self.obj_names[obj]
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
            self.stabilize_object(self.cluttered_obj)

            self.cluttered_objs.append(self.cluttered_obj)
            pose = self.cluttered_obj.get_pose().p.tolist()
            pose.append(obj_radius)
            self.size_dict.append(pose)
            success_count += 1
            self.record_cluttered_objects.append({"object_type": obj_name, "object_index": obj_idx})

            if self.cluttered_item_info[obj_name]["type"] == "urdf":
                path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/objaverse/{obj_name}/{obj_idx}/coacd_collision.obj"
            else:
                path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{obj_name}/collision/base{obj_idx}.glb"
            self.collision_list.append((self.cluttered_obj, path, self.cluttered_obj.scale))

        if success_count < cluttered_numbers:
            print(f"Warning: Only {success_count} cluttered objects are placed on the kitchen counter.")

        self.size_dict = None
        self.cluttered_objs = []

    def load_camera(self, **kwags):
        from envs.camera import Camera
        self.cameras = Camera(
            bias=self.table_z_bias,
            random_head_camera_dis=self.random_head_camera_dis,
            **kwags,
        )
        config_path = f"{os.environ['BENCH_ROOT']}/bench_assets/embodiments/kitchen_s_config.yml"
        self.cameras.add_extra_cameras(config_path)
        self.cameras.load_camera(self.scene)
        self.scene.step()
        self.scene.update_render()
