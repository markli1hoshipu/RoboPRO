from bench_envs._kitchen_base_large import Kitchen_base_large
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class open_cabinet(Kitchen_base_large):

    def setup_demo(self, is_test: bool = False, **kwargs):
        # Match collision-cache usage from other benchmark tasks
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        # No additional movable actors are needed for opening the cabinet.
        # Record the closed joint configuration for success checking.
        if hasattr(self, "cabinet") and self.cabinet is not None:
            self.cabinet_closed_qpos = np.array(self.cabinet.get_qpos())
        else:
            self.cabinet_closed_qpos = None

    def play_once(self):
        # Choose arm based on cabinet position (right if on right side, left otherwise)
        cabinet_pose = self.cabinet.get_pose().p
        arm_tag = ArmTag("right" if cabinet_pose[0] > 0 else "left")
        self.arm_tag = arm_tag

        # Ensure we have a baseline closed configuration
        if not hasattr(self, "cabinet_closed_qpos") or self.cabinet_closed_qpos is None:
            self.cabinet_closed_qpos = np.array(self.cabinet.get_qpos())

        # Grasp the cabinet door handle region
        self.move(self.grasp_actor(self.cabinet, arm_tag=arm_tag, pre_grasp_dis=0.1))

        # Pull to open the cabinet door along the robot-facing direction
        for _ in range(4):
            self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.04))

        # Log basic info for downstream use
        self.info["info"] = {
            "{A}": "036_cabinet/base46653",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        # Success: cabinet articulation has deviated from its initial closed configuration
        if self.cabinet is None:
            return False
        if not hasattr(self, "cabinet_closed_qpos") or self.cabinet_closed_qpos is None:
            return False

        current_qpos = np.array(self.cabinet.get_qpos())
        if current_qpos.shape != self.cabinet_closed_qpos.shape:
            return False

        delta = np.abs(current_qpos - self.cabinet_closed_qpos)
        return np.max(delta) > 0.02

