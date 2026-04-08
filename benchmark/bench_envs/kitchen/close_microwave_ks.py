from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class close_microwave_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        # Microwave is already loaded by the base task (self.microwave)
        # Set door to ~90% open for the close task
        self.joint_lower = self.microwave_joint_lower
        self.joint_upper = self.microwave_joint_upper
        self.joint_range = self.microwave_joint_range

        open_angle = self.joint_lower + 0.9 * self.joint_range
        limits = self.microwave.get_qlimits()
        ndof = len(limits)
        qpos = [0.0] * ndof
        qpos[0] = open_angle
        self.microwave.set_qpos(qpos)

    def play_once(self):
        arm_tag = ArmTag("right")

        # Grasp the microwave door
        self.move(self.grasp_actor(self.microwave, arm_tag=arm_tag, pre_grasp_dis=0.10))

        # Push the door closed by moving toward the microwave body
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=0.15))

        # Release gripper
        self.detach_object(arms_tag=str(arm_tag))

        self.info["info"] = {
            "{A}": f"044_microwave/base{self.microwave_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        # Success if door joint angle is less than 20% of its range
        door_angle = self.microwave.get_qpos()[0]
        threshold = self.joint_lower + 0.2 * self.joint_range
        return door_angle < threshold
