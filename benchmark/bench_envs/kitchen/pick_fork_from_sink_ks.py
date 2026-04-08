from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import math
import numpy as np
from envs._GLOBAL_CONFIGS import *
import glob


class pick_fork_from_sink_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
        self._attached = False

    def load_actors(self):
        sink_p = self.sink_pose.p
        self.fork_id = 0
        fork_pose = rand_pose(
            xlim=[sink_p[0] - 0.05, sink_p[0] + 0.05],
            ylim=[sink_p[1] - 0.04, sink_p[1] + 0.04],
            zlim=[sink_p[2] + 0.02],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, math.pi / 4],
        )
        self.fork = create_actor(
            scene=self,
            pose=fork_pose,
            modelname="033_fork",
            convex=True,
            model_id=self.fork_id,
        )
        self.fork.set_mass(0.05)
        self.collision_list.append((
            self.fork,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/033_fork/collision/base{self.fork_id}.glb",
            self.fork.scale,
        ))

        self.target_pose = [0.0, -0.05, 0.743 + self.table_z_bias + 0.02, 1, 0, 0, 0]

    def play_once(self):
        arm_tag = ArmTag("left")

        self.move(self._top_down_grasp(self.fork, arm_tag, grasp_z=0.03))
        if not self.plan_success:
            return self.info

        collision_path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/033_fork/collision/base{self.fork_id}.glb"
        self._start_kinematic_attach(self.fork, arm_tag)
        self.attach_object(self.fork, collision_path, str(arm_tag))

        # Lift out of sink
        self._kinematic_move(arm_tag, [
            Action(arm_tag, "move", target_pose=[
                self.fork.get_pose().p[0], self.fork.get_pose().p[1],
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
            "{A}": f"033_fork/base{self.fork_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        fork_pos = self.fork.get_pose().p
        return (
            fork_pos[2] > 0.70
            and abs(fork_pos[0] - self.target_pose[0]) < 0.10
            and abs(fork_pos[1] - self.target_pose[1]) < 0.10
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
