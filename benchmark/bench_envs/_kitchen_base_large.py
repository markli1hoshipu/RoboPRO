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


class Kitchen_base_large(Bench_base_task):
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

        # table: main countertop; shelf0/1: pantry rack shelves; fridge/cabinet: internal storage volumes
        self.prohibited_area = {
            "table": [],
            "shelf0": [],
            "shelf1": [],
            "fridge": [],
            "cabinet": [],
        }
        # Base env currently does not spawn distractors; keep this for compatibility.
        self.record_cluttered_objects = []

        self.eval_success = False
        self.table_z_bias = (np.random.uniform(low=-self.random_table_height, high=0) + table_height_bias)  # TODO
        self.need_plan = kwags.get("need_plan", True)
        self.left_joint_path = kwags.get("left_joint_path", [])
        self.right_joint_path = kwags.get("right_joint_path", [])
        self.left_cnt = 0
        self.right_cnt = 0

        # Fridge, microwave, and basket configuration on the table front edge.
        # Rotations in degrees (roll, pitch, yaw) and uniform scales.
        # Fixed at the base env level (shared by all tasks).
        self.fridge_left_rot = [-90.0, 0.0, 90.0]
        self.fridge_left_scale = 0.5

        self.microwave_left_rot = [-90.0, 180.0, 0.0]
        self.microwave_left_scale = 1.4

        self.basket_right_rot = [0.0, 0.0, 90.0]
        self.basket_right_scale = 1.4

        # Cabinet scale: currently only uniform scaling is supported by SAPIEN's URDF loader.
        # This parameter allows you to uniformly resize the cabinet; to truly scale only height,
        # the underlying meshes/URDF would need to encode per-axis scale.
        self.cabinet_scale = 0.5

        self.instruction = None  # for Eval

        self.collision_list = [] # list of collision objects for curobo planner

        # Map semantic appliance roles to underlying assets
        self.kitchen_appliance_assets = {
            "fridge": {"modelname": "036_cabinet", "default_modelid": 46653},
            "cabinet": {"modelname": "036_cabinet", "default_modelid": 46653},
            "drawer": {"modelname": "036_cabinet", "default_modelid": 46653},
        }

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
        # self.load_basic_kitchen_items()
        # if self.cluttered_table:
        #     self.get_cluttered_surfaces()

        # Even for a minimal scene, ensure that articulated objects like the drawer
        # are placed in a stable configuration.
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

        # Unified countertop (single static body).
        # Keep name `table` and place so the top surface is at `table_height`,
        # matching the previous create_table(...) semantics.
        counter_length = 1.2
        counter_width = 0.7
        counter_thickness = 0.05
        tabletop_pose = sapien.Pose([0.0, 0.0, -counter_thickness / 2])
        tabletop_half_size = [counter_length / 2, counter_width / 2, counter_thickness / 2]

        # Add a recessed base volume under the top so it reads like a real
        # countertop/cabinet block, while leaving a front "toe-kick" gap so
        # the robot can get closer.
        front_recess = 0.12
        base_height = max(0.0, table_height - counter_thickness)
        base_depth = max(0.0, counter_width - front_recess)
        base_half_size = [counter_length / 2, base_depth / 2, base_height / 2]
        # Local frame: actor origin is at the top surface (world z = table_height).
        # Shift the base backward (+y) so the front edge is recessed.
        base_pose = sapien.Pose(
            [0.0, front_recess / 2, -(counter_thickness / 2 + base_height / 2)]
        )

        counter_builder = self.scene.create_actor_builder()
        counter_builder.set_physx_body_type("static")
        counter_builder.add_box_collision(
            pose=tabletop_pose,
            half_size=tabletop_half_size,
            material=self.scene.default_physical_material,
        )
        if base_height > 1e-6 and base_depth > 1e-6:
            counter_builder.add_box_collision(
                pose=base_pose,
                half_size=base_half_size,
                material=self.scene.default_physical_material,
            )
        if self.table_texture is not None:
            texturepath = f"./assets/background_texture/{self.table_texture}.png"
            texture2d = sapien.render.RenderTexture2D(texturepath)
            material = sapien.render.RenderMaterial()
            material.set_base_color_texture(texture2d)
            material.base_color = [1, 1, 1, 1]
            material.metallic = 0.1
            material.roughness = 0.3
            counter_builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half_size, material=material)
            # Slightly different roughness for the base so it doesn't look like a floating slab.
            base_material = sapien.render.RenderMaterial(base_color=[0.55, 0.47, 0.38, 1])
            base_material.metallic = 0.0
            base_material.roughness = 0.8
            if base_height > 1e-6 and base_depth > 1e-6:
                counter_builder.add_box_visual(pose=base_pose, half_size=base_half_size, material=base_material)
        else:
            counter_builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half_size, material=(1, 1, 1))
            if base_height > 1e-6 and base_depth > 1e-6:
                counter_builder.add_box_visual(pose=base_pose, half_size=base_half_size, material=(0.55, 0.47, 0.38))

        counter_builder.set_initial_pose(sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], table_height]))
        self.table = counter_builder.build(name="table")

        # Place static appliances on the table front edge.
        self._load_fridge_on_table(table_height, table_xy_bias)
        self._load_microwave_on_table(table_height, table_xy_bias)
        self._load_basket_on_table(table_height, table_xy_bias)
        self._load_cabinet_on_table(table_height, table_xy_bias)

        # Additional kitchen appliances (wall cabinets, pantry rack, etc.)
        # can be re-enabled later via _load_kitchen_appliances if needed.

    def _load_fridge_on_table(self, table_height: float, table_xy_bias):
        """Place the static fridge on the left front edge of the table."""
        y_front = table_xy_bias[1] + 0.30
        x_fridge = table_xy_bias[0] - 0.40
        z_fridge = table_height + 0.20

        fx_roll_deg, fx_pitch_deg, fx_yaw_deg = self.fridge_left_rot
        fx_ax = math.radians(fx_roll_deg)
        fx_ay = math.radians(fx_pitch_deg)
        fx_az = math.radians(fx_yaw_deg)
        fqx, fqy, fqz, fqw = t3d.euler.euler2quat(fx_ax, fx_ay, fx_az)
        fridge_quat = [fqw, fqx, fqy, fqz]

        pose_fridge = sapien.Pose([x_fridge, y_front, z_fridge], fridge_quat)
        self.fridge_left = self._create_objects_bench_cabinet(
            asset_dir_name="124_fridge_hivvdf",
            pose=pose_fridge,
            fix_root_link=True,
            extra_scale=self.fridge_left_scale,
        )
        if self.fridge_left is not None:
            self.fridge_left.set_name("fridge_left")
            self.add_prohibit_area(self.fridge_left, padding=0.02, area="fridge")

    def _load_microwave_on_table(self, table_height: float, table_xy_bias):
        """Place the static microwave in the middle of the front edge of the table."""
        y_front = table_xy_bias[1] + 0.30
        x_microwave = table_xy_bias[0] + 0.0
        z_microwave = table_height + 0.02

        mw_roll_deg, mw_pitch_deg, mw_yaw_deg = self.microwave_left_rot
        mw_ax = math.radians(mw_roll_deg)
        mw_ay = math.radians(mw_pitch_deg)
        mw_az = math.radians(mw_yaw_deg)
        mw_qx, mw_qy, mw_qz, mw_qw = t3d.euler.euler2quat(mw_ax, mw_ay, mw_az)
        microwave_quat = [mw_qw, mw_qx, mw_qy, mw_qz]

        pose_microwave = sapien.Pose([x_microwave, y_front, z_microwave], microwave_quat)
        try:
            microwave_actor = create_sapien_urdf_obj(
                scene=self,
                pose=pose_microwave,
                modelname="044_microwave",
                scale=self.microwave_left_scale,
                modelid=0,
                fix_root_link=True,
            )
        except Exception as e:
            print(f"[Kitchen_base_large] failed to load microwave URDF: {e}")
            microwave_actor = None

        if microwave_actor is not None:
            self.microwave_left = microwave_actor
            self.microwave_left.set_name("microwave_center")
            self.add_prohibit_area(self.microwave_left, padding=0.01, area="table")

    def _load_basket_on_table(self, table_height: float, table_xy_bias):
        """Place the static basket on the right front edge of the table."""
        y_front = table_xy_bias[1] + 0.30
        x_right = table_xy_bias[0] + 0.45
        z_basket = table_height + 0.02

        br_roll_deg, br_pitch_deg, br_yaw_deg = self.basket_right_rot
        br_ax = math.radians(br_roll_deg)
        br_ay = math.radians(br_pitch_deg)
        br_az = math.radians(br_yaw_deg)
        bqx, bqy, bqz, bqw = t3d.euler.euler2quat(br_ax, br_ay, br_az)
        basket_quat = [bqw, bqx, bqy, bqz]

        pose_basket = sapien.Pose([x_right, y_front, z_basket], basket_quat)
        basket_actor = create_actor(
            scene=self.scene,
            pose=pose_basket,
            modelname="110_basket",
            scale=[self.basket_right_scale] * 3,
            is_static=True,
            convex=False,
            model_id=0,
        )
        if basket_actor is not None:
            self.basket_right = basket_actor
            self.basket_right.set_name("basket_right")
            self.add_prohibit_area(self.basket_right, padding=0.01, area="table")

    def _load_cabinet_on_table(self, table_height: float, table_xy_bias):
        """Place the chosen articulated cabinet asset on the opposite end of the table from the drawer."""
        # Mirror the drawer position across the table center in x.
        x_center = table_xy_bias[0]
        y_center = table_xy_bias[1] + 0.32
        z_center = table_height + 0.7

        # Apply 90° rotation around all three axes (roll, pitch, yaw).
        ax = math.radians(90)  # roll (x)
        ay = math.radians(180)  # pitch (y)
        az = math.radians(-90)  # yaw (z)
        # transforms3d.euler.euler2quat expects (ax, ay, az) in radians (x, y, z axes) and returns (qx, qy, qz, qw).
        qx, qy, qz, qw = t3d.euler.euler2quat(ax, ay, az)
        base_pose = sapien.Pose([x_center, y_center, z_center], [qw, qx, qy, qz])
        # Use the generic objects_bench cabinet loader.
        self.cabinet = self._create_objects_bench_cabinet(
            asset_dir_name="125_cabinet_tynnnw",
            pose=base_pose,
            fix_root_link=True,
            extra_scale=self.cabinet_scale,
        )
        if self.cabinet is not None:
            self.cabinet.set_name("cabinet")
            self.add_prohibit_area(self.cabinet, padding=0.02, area="cabinet")

    # -----------------------
    # Fridge articulation state helpers
    # -----------------------
    def _init_fridge_states(self):
        """Initialize canonical closed and open configurations for the fridge articulation."""
        if not hasattr(self, "fridge_left") or self.fridge_left is None:
            self.fridge_closed_qpos = None
            self.fridge_open_qpos = None
            return

        # Closed configuration = current qpos
        qpos = np.array(self.fridge_left.get_qpos(), dtype=float)
        self.fridge_closed_qpos = qpos.copy()

        # Use articulation qlimits to define an "open" configuration:
        # move each finite-interval joint to its upper limit so the door is open.
        qlimits = np.array(self.fridge_left.get_qlimits(), dtype=float)  # shape (dof, 2)
        open_qpos = qpos.copy()

        if len(open_qpos) > 0 and qlimits.shape[0] >= open_qpos.shape[0]:
            for i in range(open_qpos.shape[0]):
                low, high = qlimits[i]
                if np.isfinite(low) and np.isfinite(high) and high > low:
                    open_qpos[i] = high

        self.fridge_open_qpos = open_qpos

    def set_fridge_closed(self):
        """Reset the fridge to its canonical closed configuration."""
        if getattr(self, "fridge_closed_qpos", None) is None:
            return
        if not hasattr(self, "fridge_left") or self.fridge_left is None:
            return
        self.fridge_left.set_qpos(self.fridge_closed_qpos)

    def set_fridge_open(self):
        """Set the fridge to its canonical open configuration."""
        if getattr(self, "fridge_open_qpos", None) is None:
            return
        if not hasattr(self, "fridge_left") or self.fridge_left is None:
            return
        self.fridge_left.set_qpos(self.fridge_open_qpos)

    def is_fridge_open(self, threshold: float = 0.01) -> bool:
        """Return True if the fridge has moved significantly away from the closed configuration."""
        if getattr(self, "fridge_closed_qpos", None) is None:
            return False
        if not hasattr(self, "fridge_left") or self.fridge_left is None:
            return False

        current = np.array(self.fridge_left.get_qpos(), dtype=float)
        diff = np.abs(current - self.fridge_closed_qpos)
        return np.max(diff) > threshold

    def _create_objects_bench_cabinet(
        self,
        asset_dir_name: str,
        pose: sapien.Pose,
        fix_root_link: bool = True,
        extra_scale: float = 1.0,
    ) -> ArticulationActor | None:
        """
        Generic helper to load an articulated cabinet defined under assets/objects_bench/<asset_dir_name>
        as an ArticulationActor, using its local model_data.json for scale and transform.
        """
        modeldir = Path("assets/objects_bench") / asset_dir_name
        # Prefer mobility.urdf if present, otherwise fall back to a SAPIEN-exported URDF name.
        urdf_path = modeldir / "mobility.urdf"
        if not urdf_path.exists():
            # Common pattern for exported URDFs (e.g., sapien_urdf/nkrgez.urdf)
            sapien_urdf_dir = modeldir / "sapien_urdf"
            if sapien_urdf_dir.is_dir():
                # Pick the first .urdf file in sapien_urdf
                urdfs = list(sapien_urdf_dir.glob("*.urdf"))
                if urdfs:
                    urdf_path = urdfs[0]

        json_file = modeldir / "model_data.json"

        if not urdf_path.exists():
            print(f"[Kitchen_base_large] cabinet URDF not found: {urdf_path}")
            return None

        if json_file.exists():
            with open(json_file, "r", encoding="utf-8") as file:
                model_data = json.load(file)
            raw_scale = model_data.get("scale", 1.0)
            # For URDF loader, use a scalar scale like other SAPIEN URDF assets.
            if isinstance(raw_scale, (list, tuple)) and len(raw_scale) > 0:
                scale_scalar = float(raw_scale[0])
            else:
                scale_scalar = float(raw_scale)
            trans_mat = np.array(model_data.get("transform_matrix", np.eye(4)))
        else:
            # Provide a minimal config so downstream code (e.g. add_prohibit_area) always
            # sees a dict instead of None.
            model_data = {"scale": 1.0}
            scale_scalar = 1.0
            trans_mat = np.eye(4)

        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.scale = scale_scalar * float(extra_scale)
        loader.fix_root_link = fix_root_link
        loader.load_multiple_collisions_from_file = True

        try:
            articulation = loader.load_multiple(str(urdf_path))[0][0]
        except Exception as e:
            print(f"[Kitchen_base_large] failed to load cabinet URDF {urdf_path}: {e}")
            return None

        pose_mat = pose.to_transformation_matrix()
        pose_with_offset = sapien.Pose(
            p=pose_mat[:3, 3] + trans_mat[:3, 3],
            q=t3d.quaternions.mat2quat(trans_mat[:3, :3] @ pose_mat[:3, :3]),
        )
        articulation.set_pose(pose_with_offset)

        if model_data is not None:
            init_qpos = model_data.get("init_qpos")
            if init_qpos is not None and len(init_qpos) > 0:
                articulation.set_qpos(np.array(init_qpos, dtype=float))

        # Match drive properties to other articulated objects
        for joint in articulation.get_joints():
            joint.set_drive_properties(damping=1000, stiffness=0)

        articulation.set_name(asset_dir_name)
        return ArticulationActor(articulation, model_data)

    # Drawer-related APIs are deprecated and no longer used now that the
    # base kitchen does not spawn any drawer units. They are retained only
    # so older code that imports them does not break.
    def _init_drawer_states(self):
        self.drawer_closed_qpos = None
        self.drawer_open_qpos = None

    def set_drawer_closed(self):
        """Deprecated: drawers are no longer part of the base kitchen."""
        return

    def set_drawer_open(self):
        """Deprecated: drawers are no longer part of the base kitchen."""
        return

    def is_drawer_open(self, threshold: float = 0.01) -> bool:
        """Deprecated: always returns False because no drawers exist."""
        return False
    
    def _load_kitchen_appliances(self, table_height: float):
        # Fridge on the left side of the counter
        if "fridge" in self.kitchen_appliance_assets:
            cfg = self.kitchen_appliance_assets["fridge"]
            fridge_id = cfg["default_modelid"]
            self.fridge = rand_create_sapien_urdf_obj(
                scene=self.scene,
                modelname=cfg["modelname"],
                modelid=fridge_id,
                xlim=[-0.5, -0.5],
                ylim=[0.2, 0.2],
                rotate_rand=False,
                rotate_lim=[0, 0, 0],
                qpos=[1, 0, 0, 1],
                fix_root_link=True,
            )
            if self.fridge is not None:
                self.fridge.set_name("fridge")
                self.add_prohibit_area(self.fridge, padding=0.02, area="fridge")

        # Wall cabinet on the right, slightly above the counter
        if "cabinet" in self.kitchen_appliance_assets:
            cfg = self.kitchen_appliance_assets["cabinet"]
            cabinet_id = cfg["default_modelid"]
            self.cabinet = rand_create_sapien_urdf_obj(
                scene=self.scene,
                modelname=cfg["modelname"],
                modelid=cabinet_id,
                xlim=[0.55, 0.55],
                ylim=[0.2, 0.2],
                rotate_rand=False,
                rotate_lim=[0, 0, 0],
                qpos=[1, 0, 0, 1],
                fix_root_link=True,
            )
            if self.cabinet is not None:
                self.cabinet.set_name("cabinet")
                self.add_prohibit_area(self.cabinet, padding=0.02, area="cabinet")

        # Legacy "drawer" appliance has been removed from the base kitchen layout.

    def _get_available_model_ids(self, modelname: str):
        asset_path = os.path.join("assets/objects", modelname)
        json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))
        available_ids = []
        for file in json_files:
            base = os.path.basename(file)
            try:
                idx = int(base.replace("model_data", "").replace(".json", ""))
                available_ids.append(idx)
            except ValueError:
                continue
        return available_ids

    def _sample_model_id(self, modelname: str, fallback: int = 0) -> int:
        ids = self._get_available_model_ids(modelname)
        if not ids:
            return fallback
        return int(np.random.choice(ids))

    def add_extra_cameras(self):
        self.cameras.add_extra_cameras(f"{os.environ['BENCH_ROOT']}/bench_assets/embodiments/office_config.yml")