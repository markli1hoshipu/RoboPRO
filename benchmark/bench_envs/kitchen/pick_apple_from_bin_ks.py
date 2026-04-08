from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import math
import numpy as np
from envs._GLOBAL_CONFIGS import *
import glob


class pick_apple_from_bin_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
        self._attached = False

    def load_actors(self):
        # Spawn trash bin on counter as task-specific object
        self.bin_model_id = 0
        bin_x = 0.25 + self.table_xy_bias[0]
        bin_y = -0.05 + self.table_xy_bias[1]
        bin_z = 0.743 + self.table_z_bias + 0.003
        self.trash_bin = create_actor(
            scene=self,
            pose=sapien.Pose(p=[bin_x, bin_y, bin_z], q=[1, 0, 0, 0]),
            modelname="063_tabletrashbin",
            convex=True,
            model_id=self.bin_model_id,
            is_static=True,
        )
        self.bin_pose = sapien.Pose(p=[bin_x, bin_y, bin_z])
        self.add_prohibit_area(self.trash_bin, padding=0.06, area="table")

        bin_p = self.bin_pose.p
        self.apple_id = 0
        apple_pose = self._safe_rand_pose(
            xlim=[bin_p[0] - 0.03, bin_p[0] + 0.03],
            ylim=[bin_p[1] - 0.03, bin_p[1] + 0.03],
            zlim=[bin_p[2] + 0.05],
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
        self.collision_list.append((
            self.apple,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/035_apple/collision/base{self.apple_id}.glb",
            self.apple.scale,
        ))

        self.target_pose = [0.0, -0.05, 0.743 + self.table_z_bias + 0.02, 1, 0, 0, 0]

    def play_once(self):
        arm_tag = ArmTag("right")

        self.move(self._top_down_grasp(self.apple, arm_tag, grasp_z=0.05))
        if not self.plan_success:
            return self.info

        collision_path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/035_apple/collision/base{self.apple_id}.glb"
        self._start_kinematic_attach(self.apple, arm_tag)
        self.attach_object(self.apple, collision_path, str(arm_tag))

        # Lift up to counter level
        self._kinematic_move(arm_tag, [
            Action(arm_tag, "move", target_pose=[
                self.apple.get_pose().p[0], self.apple.get_pose().p[1],
                0.743 + self.table_z_bias + 0.25, 0.707, 0, 0.707, 0
            ]),
        ])

        # Place on counter
        self._kinematic_move(arm_tag, [
            Action(arm_tag, "move", target_pose=self.target_pose),
        ])

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
        return (
            apple_pos[2] > 0.70
            and abs(apple_pos[0] - self.target_pose[0]) < 0.10
            and abs(apple_pos[1] - self.target_pose[1]) < 0.10
            and np.sqrt((apple_pos[0] - bin_p[0])**2 + (apple_pos[1] - bin_p[1])**2) > 0.08
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
