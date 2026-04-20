from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class pick_hamburger_from_microwave_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        # Force microwave door fully open so the interior is accessible.
        limits = self.microwave.get_qlimits()
        qpos_mw = self.microwave.get_qpos()
        qpos_mw[0] = limits[0][1] * 0.9
        self.microwave.set_qpos(qpos_mw)

        # Spawn the hamburger inside the microwave interior with small jitter.
        mw_p = self.microwave.get_pose().p
        self.hamburger_id = int(np.random.choice([0, 1, 2]))
        hpose = sapien.Pose(
            [float(mw_p[0]) + float(np.random.uniform(-0.03, 0.03)),
             float(mw_p[1]) + 0.05,
             float(mw_p[2]) + 0.05],
            [0.5, 0.5, 0.5, 0.5],
        )
        self.target_obj = create_actor(
            scene=self,
            pose=hpose,
            modelname="006_hamburg",
            convex=True,
            model_id=self.hamburger_id,
        )
        self.target_obj.set_mass(0.05)

        # Pick the arm closer to the microwave in world x.
        self.arm_tag = ArmTag("right" if mw_p[0] > 0 else "left")

    def play_once(self):
        arm_tag = self.arm_tag

        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.08)

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.25))

    def check_success(self):
        tp = self.target_obj.get_pose().p
        table_top_z = self.kitchens_info["table_height"] + self.table_z_bias
        gripper_closed = (
            (not self.robot.is_left_gripper_open())
            if self.arm_tag == "left"
            else (not self.robot.is_right_gripper_open())
        )
        return tp[2] > table_top_z + 0.15 and gripper_closed
