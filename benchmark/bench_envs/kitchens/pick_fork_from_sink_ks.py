from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
import os
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

        # Counter drop pose on the same arm side.
        side_sign = 1 if self.arm_tag == "right" else -1
        target_rand_pose = self.rand_pose_on_counter(
            xlim=[0.05, 0.32] if side_sign > 0 else [-0.32, -0.05],
            ylim=[-0.20, 0.00],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            obj_padding=0.05,
        )
        self.des_obj_pose = target_rand_pose.p.tolist() + [0, 0, 0, 1]
        self.des_obj_pose[2] += 0.04

    def play_once(self):
        arm_tag = self.arm_tag

        # Fork sits inside the sink basin — disable the table obstacle so the
        # planner can reach below the counter level for a side grasp.
        self.enable_table(enable=False)

        grasp_arm, actions = self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.07)
        if grasp_arm is None:
            # grasp_actor failed to find a pre_grasp_pose for the yaw-randomized
            # fork. Abort this episode; planner will mark it failed.
            return
        arm_tag = grasp_arm
        self.move((arm_tag, actions))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.25))

        # Place on counter. place_actor(constrain="free") after side-grasp
        # trips INVALID_PARTIAL_POSE_COST_METRIC (same root cause as
        # put_bowl_in_sink_ks), so use explicit two-stage move_to_pose with
        # INIT_Q front-facing wrist, then open gripper.
        self.attach_object(
            self.target_obj,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/033_fork/collision/base{self.fork_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        INIT_Q = [0.707, 0, 0, 0.707]
        tgt_x, tgt_y, tgt_z = self.des_obj_pose[:3]
        hover_pose = [tgt_x, tgt_y, tgt_z + 0.20] + INIT_Q
        self.move(self.move_to_pose(arm_tag, hover_pose))
        drop_pose = [tgt_x, tgt_y, tgt_z + 0.06] + INIT_Q
        self.move(self.move_to_pose(arm_tag, drop_pose))
        self.move(self.open_gripper(arm_tag, pos=1.0))

    def check_success(self):
        tp = self.target_obj.get_pose().p
        table_top_z = self.kitchens_info["table_height"] + self.table_z_bias
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]
        on_counter_z = abs(tp[2] - table_top_z) < 0.08
        outside_sink = (abs(tp[0] - sink_p[0]) > sg["hole_hx"]
                        or abs(tp[1] - sink_p[1]) > sg["hole_hy"])
        return (on_counter_z
                and outside_sink
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
