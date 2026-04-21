from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien, math, os, glob
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy


class chain_sink_bowl_bread_knife_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.bowl.get_name(), self.bread.get_name(), self.knife.get_name()}

    def load_actors(self):
        sink_p = self.sink.get_pose().p

        # Bowl starts in the sink basin. Right arm will pick it up.
        bowl_start = sapien.Pose(
            [float(sink_p[0]), float(sink_p[1]), float(sink_p[2]) - 0.01],
            [0.5, 0.5, 0.5, 0.5],
        )
        self.bowl_id = 3
        self.bowl = create_actor(
            scene=self, pose=bowl_start, modelname="002_bowl",
            convex=True, model_id=self.bowl_id,
        )
        self.bowl.set_mass(0.1)

        # Counter target for the bowl: same side as the right arm, just off
        # the sink (outside sink hole footprint) so the arm doesn't cross the
        # midline while carrying the bowl.
        bowl_target_x = 0.15
        bowl_target_y = -0.05
        table_top_z = self.kitchens_info["table_height"] + self.table_z_bias
        self.bowl_counter_pose = [bowl_target_x, bowl_target_y,
                                  table_top_z + 0.02, 0, 0, 0, 1]

        # Bread on counter — compact variants only (base3 is baguette).
        bread_pose = self.rand_pose_on_counter(
            xlim=[-0.32, 0.32], ylim=[-0.15, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, np.pi/2, 0],
            obj_padding=0.05,
        )
        while (abs(bread_pose.p[0] - bowl_target_x) < 0.12
               and abs(bread_pose.p[1] - bowl_target_y) < 0.10):
            bread_pose = self.rand_pose_on_counter(
                xlim=[-0.32, 0.32], ylim=[-0.15, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, np.pi/2, 0],
                obj_padding=0.05,
            )
        self.bread_id = int(np.random.choice([1, 2, 5]))
        self.bread = create_actor(
            scene=self, pose=bread_pose, modelname="075_bread",
            convex=True, model_id=self.bread_id,
        )
        self.bread.set_mass(0.05)
        self.add_prohibit_area(self.bread, padding=0.02, area="table")

        # Knife on counter
        knife_pose = self.rand_pose_on_counter(
            xlim=[-0.32, 0.32], ylim=[-0.15, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, np.pi, 0],
            obj_padding=0.05,
        )
        while (abs(knife_pose.p[0] - bowl_target_x) < 0.12
               and abs(knife_pose.p[1] - bowl_target_y) < 0.10):
            knife_pose = self.rand_pose_on_counter(
                xlim=[-0.32, 0.32], ylim=[-0.15, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, np.pi, 0],
                obj_padding=0.05,
            )
        self.knife = create_actor(
            scene=self, pose=knife_pose, modelname="034_knife",
            convex=True, model_id=0,
        )
        self.knife.set_mass(0.05)
        self.add_prohibit_area(self.knife, padding=0.02, area="table")

    def play_once(self):
        # Step 1: bowl from sink → counter (right arm). Bowl has IK-reachable
        # side contact points, so we can use the standard grasp_actor_from_table
        # with enable_table(False) so the wrist can dip below rim. Then move
        # to a pose above the counter target and open the gripper — avoids
        # the place_actor mid-air detour caused by side-grasp frame ambiguity.
        arm = ArmTag("right")
        self.enable_table(enable=False)
        self.grasp_actor_from_table(self.bowl, arm_tag=arm, pre_grasp_dis=0.10)
        self.move(self.move_by_displacement(arm_tag=arm, z=0.10))
        self.attach_object(self.bowl,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/002_bowl/collision/base{self.bowl_id}.glb",
            str(arm))
        self.enable_table(enable=True)

        # Use an explicit top-down wrist quat — side-grasp cur_q is usually
        # not IK-reachable at the counter drop pose.
        TOP_DOWN_Q = [-0.5, 0.5, -0.5, -0.5]
        TOP_DOWN_Q = [1,0,0,0]
        drop_pose = [
            self.bowl_counter_pose[0],
            self.bowl_counter_pose[1],
            self.bowl_counter_pose[2] + 0.16,
        ] + TOP_DOWN_Q
        self.move(self.move_to_pose(arm, drop_pose))
        self.move(self.open_gripper(arm, pos=1.0))
        self.move(self.back_to_origin(arm))

        # Refresh actual bowl pose (release may have shifted it slightly).
        bowl_p = self.bowl.get_pose().p
        bread_target = [bowl_p[0], bowl_p[1], bowl_p[2] + 0.02, 0, 0, 0, 1]

        # Step 2: bread → bowl (arm chosen by bread x).
        arm = ArmTag("right" if self.bread.get_pose().p[0] > 0 else "left")
        self.grasp_actor_from_table(self.bread, arm_tag=arm, pre_grasp_dis=0.07)
        self.move(self.move_by_displacement(arm_tag=arm, z=0.15))
        self.attach_object(self.bread,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/075_bread/collision/base{self.bread_id}.glb",
            str(arm))
        self.enable_table(enable=True)
        self.move(self.place_actor(self.bread, arm_tag=arm, target_pose=bread_target,
            constrain="align", pre_dis=0.05, dis=0.005))
        self.move(self.back_to_origin(arm))

        # Step 3: knife → beside bowl.
        arm = ArmTag("right" if self.knife.get_pose().p[0] > 0 else "left")
        side = 0.10 if arm == "right" else -0.10
        knife_target = [bowl_p[0] + side, bowl_p[1], bowl_p[2] + 0.015, 0, 0, 0, 1]
        self.grasp_actor_from_table(self.knife, arm_tag=arm, pre_grasp_dis=0.07)
        self.move(self.move_by_displacement(arm_tag=arm, z=0.15))
        self.attach_object(self.knife,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/034_knife/collision/base0.glb",
            str(arm))
        self.enable_table(enable=True)
        self.move(self.place_actor(self.knife, arm_tag=arm, target_pose=knife_target,
            constrain="align", pre_dis=0.05, dis=0.005))

    def check_success(self):
        bowl_p = self.bowl.get_pose().p
        bread_p = self.bread.get_pose().p
        knife_p = self.knife.get_pose().p
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]

        bowl_out_of_sink = (abs(bowl_p[0] - sink_p[0]) > sg["hole_hx"]
                            or abs(bowl_p[1] - sink_p[1]) > sg["hole_hy"])
        bread_on_bowl = (abs(bread_p[0] - bowl_p[0]) < 0.06
                         and abs(bread_p[1] - bowl_p[1]) < 0.06)
        knife_dx = abs(knife_p[0] - bowl_p[0])
        knife_dy = abs(knife_p[1] - bowl_p[1])
        knife_beside = (0.05 <= knife_dx <= 0.15 and knife_dy <= 0.10)

        return (bowl_out_of_sink and bread_on_bowl and knife_beside
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
