from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
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

    def load_actors(self):
        # Set microwave door to fully open
        open_angle = self.microwave_joint_lower + 0.95 * self.microwave_joint_range
        limits = self.microwave.get_qlimits()
        ndof = len(limits)
        qpos = [0.0] * ndof
        qpos[0] = open_angle
        self.microwave.set_qpos(qpos)

        # Hamburger inside microwave
        mw_pos = self.microwave.get_pose().p
        self.hamburg_id = np.random.randint(0, 6)
        hamburg_pose = sapien.Pose(
            p=[mw_pos[0], mw_pos[1], mw_pos[2] + 0.05],
            q=[0.7071, 0.7071, 0, 0],
        )
        self.hamburg = create_actor(
            scene=self,
            pose=hamburg_pose,
            modelname="006_hamburg",
            convex=True,
            model_id=self.hamburg_id,
        )
        self.hamburg.set_mass(0.08)
        self.collision_list.append((
            self.hamburg,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/006_hamburg/collision/base{self.hamburg_id}.glb",
            self.hamburg.scale,
        ))

        # Target: counter center
        self.target_pose = [0.0, -0.05, 0.743 + self.table_z_bias + 0.02, 1, 0, 0, 0]

    def play_once(self):
        arm_tag = ArmTag("right")

        self.move(self.grasp_actor(self.hamburg, arm_tag=arm_tag, pre_grasp_dis=0.08))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08))

        self.attach_object(
            self.hamburg,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/006_hamburg/collision/base{self.hamburg_id}.glb",
            str(arm_tag),
        )

        self.move(
            self.place_actor(
                self.hamburg,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                constrain="align",
                pre_dis=0.07,
                dis=0.005,
            ))

        self.detach_object(arms_tag=str(arm_tag))

        self.info["info"] = {
            "{A}": f"006_hamburg/base{self.hamburg_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        hamburg_pos = self.hamburg.get_pose().p
        # Should be on counter, not in microwave
        return (
            hamburg_pos[2] < 0.85
            and abs(hamburg_pos[0] - self.target_pose[0]) < 0.10
            and abs(hamburg_pos[1] - self.target_pose[1]) < 0.10
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
