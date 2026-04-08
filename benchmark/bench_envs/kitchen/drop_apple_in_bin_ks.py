from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import math
import numpy as np
from envs._GLOBAL_CONFIGS import *
import glob


class drop_apple_in_bin_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
        self._attached = False

    def load_actors(self):
        self.apple_id = 0
        apple_pose = rand_pose(
            xlim=[-0.15, 0.15],
            ylim=[-0.20, 0.0],
            zlim=[0.743 + self.table_z_bias],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, math.pi / 4],
        )
        self.apple = create_actor(
            scene=self,
            pose=apple_pose,
            modelname="035_apple",
            convex=True,
            model_id=self.apple_id,
        )
        self.apple.set_mass(0.15)
        self.add_prohibit_area(self.apple, padding=0.04, area="table")
        self.collision_list.append((
            self.apple,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/035_apple/collision/base{self.apple_id}.glb",
            self.apple.scale,
        ))

        bin_p = self.bin_pose.p
        self.target_pose = [bin_p[0], bin_p[1], bin_p[2] + 0.25] + [1, 0, 0, 0]

    def play_once(self):
        arm_tag = ArmTag("right")

        self.move(self.grasp_actor(self.apple, arm_tag=arm_tag, pre_grasp_dis=0.08))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15))
        if not self.plan_success:
            return self.info

        collision_path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/035_apple/collision/base{self.apple_id}.glb"
        self._start_kinematic_attach(self.apple, arm_tag)
        self.attach_object(self.apple, collision_path, str(arm_tag))

        # Move above bin
        self._kinematic_move(arm_tag, [
            Action(arm_tag, "move", target_pose=self.target_pose),
        ])

        # Drop
        self.move((arm_tag, [Action(arm_tag, "open", target_gripper_pos=1.0)]))
        self._stop_kinematic_attach()
        for _ in range(200):
            self.scene.step()
        self.detach_object(arms_tag=str(arm_tag))

        self.info["info"] = {
            "{A}": f"035_apple/base{self.apple_id}",
            "{B}": f"063_tabletrashbin/base{self.bin_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        apple_pos = self.apple.get_pose().p
        bin_p = self.bin_pose.p
        eps = 0.15
        return (
            abs(apple_pos[0] - bin_p[0]) < eps
            and abs(apple_pos[1] - bin_p[1]) < eps
            and apple_pos[2] < 0.5
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
