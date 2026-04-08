from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import math
import numpy as np
from envs._GLOBAL_CONFIGS import *
import glob


class pick_apple_from_sink_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        # Apple inside sink
        sink_p = self.sink_pose.p
        self.apple_id = 0
        apple_pose = rand_pose(
            xlim=[sink_p[0] - 0.05, sink_p[0] + 0.05],
            ylim=[sink_p[1] - 0.04, sink_p[1] + 0.04],
            zlim=[0.743 + self.table_z_bias],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )
        self.apple = create_actor(
            scene=self,
            pose=apple_pose,
            modelname="035_apple",
            convex=True,
            model_id=self.apple_id,
        )
        self.apple.set_mass(0.15)
        self.collision_list.append((
            self.apple,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/035_apple/collision/base{self.apple_id}.glb",
            self.apple.scale,
        ))

        # Target: counter in front zone
        self.target_pose = [0.0, -0.10, 0.743 + self.table_z_bias + 0.02, 1, 0, 0, 0]

    def play_once(self):
        # Pick arm based on sink position
        sink_x = self.sink_pose.p[0]
        arm_tag = ArmTag("right" if sink_x > 0 else "left")

        # 1. Disable table collision so arm can reach into sink
        self._disable_planner_table(arm_tag)

        # 2. Grasp apple in sink
        self.move(self.grasp_actor(self.apple, arm_tag=arm_tag, pre_grasp_dis=0.08))
        if not self.plan_success:
            return self.info

        # 2. Lift out of sink
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15))
        if not self.plan_success:
            return self.info

        # 3. Attach
        self.attach_object(
            self.apple,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/035_apple/collision/base{self.apple_id}.glb",
            str(arm_tag),
        )

        # 4. Place on counter
        self.move(self.place_actor(
            self.apple,
            arm_tag=arm_tag,
            target_pose=self.target_pose,
            pre_dis=0.07,
            dis=0.005,
        ))

        # 5. Detach
        self.detach_object(arms_tag=str(arm_tag))

        self.info["info"] = {
            "{A}": f"035_apple/base{self.apple_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        apple_pos = self.apple.get_pose().p
        return (
            apple_pos[2] > 0.70
            and abs(apple_pos[0] - self.target_pose[0]) < 0.10
            and abs(apple_pos[1] - self.target_pose[1]) < 0.10
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
