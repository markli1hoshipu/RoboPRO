from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import math
import numpy as np
from envs._GLOBAL_CONFIGS import *
import glob


class put_spoon_in_sink_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.spoon_id = 0
        spoon_pose = self._safe_rand_pose(
            xlim=[-0.35, 0.35],
            ylim=[-0.20, -0.05],
            zlim=[0.743 + self.table_z_bias],
            qpos=[0.5, 0.5, -0.5, -0.5],
            rotate_rand=True,
            rotate_lim=[0, 0, math.pi / 4],
            obj_radius=0.06,
        )
        self.spoon = create_actor(
            scene=self,
            pose=spoon_pose,
            modelname="134_spoon",
            convex=True,
            model_id=self.spoon_id,
        )
        self.spoon.set_mass(0.05)
        self.add_prohibit_area(self.spoon, padding=0.03, area="table")
        self.collision_list.append((
            self.spoon,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/134_spoon/collision/base{self.spoon_id}.glb",
            self.spoon.scale,
        ))

    def play_once(self):
        # Pick arm based on spoon position
        arm_tag = ArmTag("right" if self.spoon.get_pose().p[0] > 0 else "left")

        # 1. Grasp
        self.move(self.grasp_actor(self.spoon, arm_tag=arm_tag, pre_grasp_dis=0.08))
        if not self.plan_success:
            return self.info

        # 2. Lift
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15))
        if not self.plan_success:
            return self.info

        # 3. Attach for collision-aware transport
        self.attach_object(
            self.spoon,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/134_spoon/collision/base{self.spoon_id}.glb",
            str(arm_tag),
        )

        # 4. Place in sink
        sink_p = self.sink_pose.p
        target_pose = [sink_p[0], sink_p[1] - 0.04, sink_p[2] + 0.05, 1, 0, 0, 0]
        self.move(self.place_actor(
            self.spoon,
            arm_tag=arm_tag,
            target_pose=target_pose,
            pre_dis=0.10,
            dis=0.0,
        ))

        # 5. Detach
        self.detach_object(arms_tag=str(arm_tag))

        self.info["info"] = {
            "{A}": f"134_spoon/base{self.spoon_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        spoon_pos = self.spoon.get_pose().p
        sink_p = self.sink_pose.p
        return (
            abs(spoon_pos[0] - sink_p[0]) < 0.15
            and abs(spoon_pos[1] - sink_p[1]) < 0.22
            and spoon_pos[2] < 0.743 + self.table_z_bias
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
