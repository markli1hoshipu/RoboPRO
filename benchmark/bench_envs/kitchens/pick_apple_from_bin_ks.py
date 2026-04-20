from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class pick_apple_from_bin_ks(KitchenS_base_task):

    # Scripted top-down grasp. 035_apple's labeled contacts all encode
    # side-grasp orientations; in a bin cavity those IK solutions are
    # narrow/infeasible. A scripted top-down descent bypasses that.
    # For aloha: input pose maps directly to end-link (gripper_bias=0.12,
    # delta_matrix=identity). This quat makes link +x point in world -z.
    TOP_DOWN_Q = [-0.5, 0.5, -0.5, -0.5]
    # TCP = link_pos + link_x * 0.12; for top-down wrist, link_z = tcp_z + 0.12.
    TCP_OFFSET = 0.12

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        bin_pose = self.rand_pose_on_counter(
            xlim=[-0.32, 0.32],
            ylim=[-0.23, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0],
            obj_padding=0.12,
        )
        # Force the bin off-center so one arm is the obvious picker.
        while abs(bin_pose.p[0]) < 0.3:
            bin_pose = self.rand_pose_on_counter(
                xlim=[-0.32, 0.32],
                ylim=[-0.23, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 4, 0],
                obj_padding=0.12,
            )

        self.bin_id = int(np.random.choice([0, 6]))
        self.bin = create_actor(
            scene=self,
            pose=bin_pose,
            modelname="063_tabletrashbin",
            convex=True,
            model_id=self.bin_id,
            scale=[0.10, 0.10, 0.10],
            is_static=True,
        )
        self.bin.set_name("bin")
        self.add_prohibit_area(self.bin, padding=0.02, area="table")

        bp = self.bin.get_pose().p
        ax = bp[0] + np.random.uniform(-0.04, 0.04)
        ay = bp[1] + np.random.uniform(-0.03, 0.03)
        # ~0.06 m above bin origin sits inside the scaled ~0.10 m tall bin.
        az = bp[2] + 0.06
        apple_pose = sapien.Pose([ax, ay, az], [0.5, 0.5, 0.5, 0.5])

        self.apple_id = int(np.random.choice([0, 1]))
        self.target_obj = create_actor(
            scene=self,
            pose=apple_pose,
            modelname="035_apple",
            convex=True,
            model_id=self.apple_id,
        )
        self.target_obj.set_mass(0.05)

        self.arm_tag = ArmTag("right" if bp[0] > 0 else "left")
        self.lift_z = bp[2] + 0.30

    def play_once(self):
        arm_tag = self.arm_tag

        # Counter cuboid must be out of the Curobo world so the wrist can
        # drop to apple height inside the bin.
        self.enable_table(enable=False)

        self.move(self.open_gripper(arm_tag, pos=1.0))

        apple_p = self.target_obj.get_pose().p
        bp = self.bin.get_pose().p

        # Hover above the bin rim, centered over the apple.
        hover_tcp_z = float(bp[2]) + 0.12
        hover_pose = [
            float(apple_p[0]),
            float(apple_p[1]),
            hover_tcp_z + self.TCP_OFFSET,
        ] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, hover_pose))

        # Descend to apple.
        grasp_tcp_z = float(apple_p[2]) + 0.005
        grasp_pose = [
            float(apple_p[0]),
            float(apple_p[1]),
            grasp_tcp_z + self.TCP_OFFSET,
        ] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, grasp_pose))

        self.move(self.close_gripper(arm_tag, pos=0.0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.25))

    def check_success(self):
        tp = self.target_obj.get_pose().p
        table_top_z = self.kitchens_info["table_height"] + self.table_z_bias
        gripper_closed = (not self.robot.is_left_gripper_open()) if self.arm_tag == "left" else (not self.robot.is_right_gripper_open())
        return tp[2] > table_top_z + 0.10 and gripper_closed
