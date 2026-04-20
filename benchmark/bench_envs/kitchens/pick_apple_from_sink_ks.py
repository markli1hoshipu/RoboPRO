from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class pick_apple_from_sink_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]

        # Apple spawns across much of the sink basin footprint. Side grasps
        # need the arm to dip below the counter, which is ok once the table
        # obstacle is disabled in play_once. y is biased toward the robot
        # (front of sink) since deep-reach IK fails when the apple is at the
        # far wall.
        bx = sink_p[0] + np.random.uniform(-sg["inner_hx"] * 0.45, sg["inner_hx"] * 0.45)
        by = sink_p[1] + np.random.uniform(-sg["inner_hy"] * 0.50, sg["inner_hy"] * 0.15)
        bz = sink_p[2] - 0.01
        rand_pos = sapien.Pose([bx, by, bz], [0.5, 0.5, 0.5, 0.5])

        self.apple_id = int(np.random.choice([0, 1]))
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="035_apple",
            convex=True,
            model_id=self.apple_id,
        )
        self.target_obj.set_mass(0.05)

        # Arm choice is driven by the sampled apple x (sink may straddle midline).
        self.arm_tag = ArmTag("right" if bx > 0 else "left")
        self.lift_z = sink_p[2] + 0.25

        # Pre-compute a clear counter pose on the same arm side to place the
        # apple after retrieving it from the sink. rand_pose_on_counter avoids
        # the sink + faucet prohibited zone automatically.
        side_sign = 1 if self.arm_tag == "right" else -1
        target_rand_pose = self.rand_pose_on_counter(
            xlim=[0.05 * side_sign, 0.32 * side_sign] if side_sign > 0 else [-0.32, -0.05],
            ylim=[-0.20, 0.00],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            obj_padding=0.05,
        )
        self.des_obj_pose = target_rand_pose.p.tolist() + [0, 0, 0, 1]
        # Lift target slightly so the apple is released just above the counter.
        self.des_obj_pose[2] += 0.04

    def play_once(self):
        arm_tag = self.arm_tag

        # Apple sits inside the sink basin (below counter level). The Curobo
        # world models the counter as a solid cuboid with no sink hole, so
        # any side grasp that would dip below the counter is IK-rejected.
        # Drop the table obstacle before planning so the planner can reach in.
        self.enable_table(enable=False)

        arm_tag, actions = self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.07)
        self.move((arm_tag, actions))

        # Lift clear of the sink rim before re-enabling the counter obstacle.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.25))

        # Place on the counter.
        self.attach_object(
            self.target_obj,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/035_apple/collision/base{self.apple_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose=self.des_obj_pose,
                constrain="free",
                pre_dis=0.07,
                dis=0.005,
            ))

    def check_success(self):
        tp = self.target_obj.get_pose().p
        table_top_z = self.kitchens_info["table_height"] + self.table_z_bias
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]
        # Apple is on the counter surface (not hovering, not back in the sink),
        # both grippers open.
        on_counter_z = abs(tp[2] - table_top_z) < 0.08
        outside_sink = (abs(tp[0] - sink_p[0]) > sg["hole_hx"]
                        or abs(tp[1] - sink_p[1]) > sg["hole_hy"])
        return (on_counter_z
                and outside_sink
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
