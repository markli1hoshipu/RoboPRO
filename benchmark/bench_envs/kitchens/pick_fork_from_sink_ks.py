from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class pick_fork_from_sink_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]

        # Same reasoning as pick_apple_from_sink: keep the spawn tight and
        # shallow so the side-grasp stays within IK reach once the table
        # obstacle is disabled in play_once.
        bx = sink_p[0] + np.random.uniform(-sg["inner_hx"] * 0.15, sg["inner_hx"] * 0.15)
        by = sink_p[1] + np.random.uniform(-sg["inner_hy"] * 0.15, sg["inner_hy"] * 0.15)
        bz = sink_p[2] - 0.02
        rand_pos = sapien.Pose([bx, by, bz], [0.5, 0.5, 0.5, 0.5])

        self.fork_id = 0
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="033_fork",
            convex=True,
            model_id=self.fork_id,
        )
        self.target_obj.set_mass(0.05)

        self.arm_tag = ArmTag("right" if bx > 0 else "left")
        self.lift_z = sink_p[2] + 0.25

    def play_once(self):
        arm_tag = self.arm_tag

        # Fork sits inside the sink basin — disable the table obstacle so the
        # planner can reach below the counter level for a side grasp.
        self.enable_table(enable=False)

        arm_tag, actions = self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.07)
        self.move((arm_tag, actions))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.25))

    def check_success(self):
        tp = self.target_obj.get_pose().p
        table_top_z = self.kitchens_info["table_height"] + self.table_z_bias
        gripper_closed = (not self.robot.is_left_gripper_open()) if self.arm_tag == "left" else (not self.robot.is_right_gripper_open())
        return tp[2] > table_top_z + 0.10 and gripper_closed
