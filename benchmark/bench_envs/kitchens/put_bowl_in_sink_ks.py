from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
import os
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class put_bowl_in_sink_ks(KitchenS_base_task):

    # Mirrors put_spoon_in_sink_ks: grasp bowl from counter with the
    # side-grasp recipe, then place in the sink with constrain="free".
    # Sink is always at sink_x >= +0.10, so the bowl spawns on the right
    # half of the counter and the right arm does the full carry.

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        rand_pos = self.rand_pose_on_counter(
            xlim=[0.05, 0.30],
            ylim=[-0.20, 0.00],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            obj_padding=0.08,
        )

        self.bowl_id = 3
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="002_bowl",
            convex=True,
            model_id=self.bowl_id,
        )
        self.target_obj.set_mass(0.05)

        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")

    def play_once(self):
        arm_tag = ArmTag("right")

        self.grasp_actor_from_table(
            self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.10,
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15))

        self.attach_object(
            self.target_obj,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/002_bowl/collision/base{self.bowl_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        sink_p = self.sink.get_pose().p
        sink_target = [
            float(sink_p[0]) - 0.05,
            float(sink_p[1]),
            float(sink_p[2]) + 0.06,
            0, 0, 0, 1,
        ]
        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose=sink_target,
                constrain="free",
                pre_dis=0.07,
                dis=0.005,
            ))

    def check_success(self):
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]
        tp = self.target_obj.get_pose().p
        in_xy = (abs(tp[0] - sink_p[0]) < sg["hole_hx"]
                 and abs(tp[1] - sink_p[1]) < sg["hole_hy"])
        below_rim = tp[2] < sink_p[2] + 0.02
        return (in_xy and below_rim
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
