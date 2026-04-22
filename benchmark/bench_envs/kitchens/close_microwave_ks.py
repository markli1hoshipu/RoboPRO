from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
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
        # microwave is already loaded by KitchenS_base_task as self.microwave
        # Start with the door mostly open so the task is to close it.
        limits = self.microwave.get_qlimits()
        qpos = self.microwave.get_qpos()
        qpos[0] = limits[0][1] * 0.9  # mostly open
        self.microwave.set_qpos(qpos)

    def play_once(self):
        arm_tag = ArmTag("left")

        # Grasp the door handle
        self.move(self.grasp_actor(self.microwave, arm_tag=arm_tag, pre_grasp_dis=0.08, contact_point_id=0))

        # Random pre-close waypoints (mirrors put_can_close_cabinet's sweep
        # of intermediate moves before the commit) — gives the planner a
        # varied approach and the policy non-trivial pre-commit trajectory.
        for _ in range(3):
            dx = float(np.random.uniform(-0.03, 0.03))
            dy = float(np.random.uniform(-0.02, 0.02))
            dz = float(np.random.uniform(-0.02, 0.03))
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx, y=dy, z=dz))
            if not self.plan_success:
                self.plan_success = True  # don't abort; these are exploratory
                break

        limits = self.microwave.get_qlimits()
        start_qpos = self.microwave.get_qpos()[0]

        # Push the handle toward the microwave body to close the door.
        # The exact direction depends on hinge orientation; +x pushes the
        # handle inward when the microwave opens to the left in world frame.
        for _ in range(15):
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.02))

            new_qpos = self.microwave.get_qpos()[0]
            if not self.plan_success:
                break
            if self.check_success(target=0.1):
                break
            # If nothing is happening, bail out early.
            if abs(new_qpos - start_qpos) <= 0.001 and _ > 2:
                break
            start_qpos = new_qpos

    def check_success(self, target=0.1):
        limits = self.microwave.get_qlimits()
        qpos = self.microwave.get_qpos()
        return qpos[0] <= limits[0][1] * target
