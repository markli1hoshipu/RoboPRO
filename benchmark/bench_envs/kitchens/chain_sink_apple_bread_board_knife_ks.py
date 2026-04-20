from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien, math, os, glob
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy


class chain_sink_apple_bread_board_knife_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.apple.get_name(), self.bread.get_name(), self.knife.get_name()}

    def load_actors(self):
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]

        # Apple inside sink basin
        self.apple_id = int(np.random.choice([0, 1]))
        ax = sink_p[0] + np.random.uniform(-sg["inner_hx"] * 0.4, sg["inner_hx"] * 0.4)
        ay = sink_p[1] + np.random.uniform(-sg["inner_hy"] * 0.4, sg["inner_hy"] * 0.4)
        az = sink_p[2] - sg["depth"] * 0.5
        apple_pose = sapien.Pose([ax, ay, az], [0.5, 0.5, 0.5, 0.5])
        self.apple = create_actor(
            scene=self, pose=apple_pose, modelname="035_apple",
            convex=True, model_id=self.apple_id,
        )
        self.apple.set_mass(0.05)

        # Plate on counter (static destination for apple)
        plate_pose = self.rand_pose_on_counter(
            xlim=[-0.32, 0.32], ylim=[-0.15, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=False,
            obj_padding=0.08,
        )
        self.plate = create_actor(
            scene=self, pose=plate_pose, modelname="003_plate",
            convex=True, model_id=0, is_static=True,
        )
        self.plate.set_mass(0.1)
        self.add_prohibit_area(self.plate, padding=0.02, area="table")

        # Board on counter (static destination for bread)
        self.board_id = int(np.random.choice([0, 1, 2, 3]))
        board_pose = self.rand_pose_on_counter(
            xlim=[-0.32, 0.32], ylim=[-0.23, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, np.pi/4, 0],
            obj_padding=0.10,
        )
        self.board = create_actor(
            scene=self, pose=board_pose, modelname="104_board",
            convex=True, model_id=self.board_id,
            scale=[0.10, 0.10, 0.10], is_static=True,
        )
        self.board.set_name("board")
        self.add_prohibit_area(self.board, padding=0.02, area="table")

        # Bread on counter
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
        self.knife = create_actor(
            scene=self, pose=knife_pose, modelname="034_knife",
            convex=True, model_id=0,
        )
        self.knife.set_mass(0.05)
        self.add_prohibit_area(self.knife, padding=0.02, area="table")

    def play_once(self):
        # Step 1: apple from sink -> plate
        sink_p = self.sink.get_pose().p
        arm = ArmTag("right" if sink_p[0] > 0 else "left")
        self.grasp_actor_from_table(self.apple, arm_tag=arm, pre_grasp_dis=0.07)
        self.move(self.move_by_displacement(arm_tag=arm, z=0.25))
        self.attach_object(self.apple,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/035_apple/collision/base{self.apple_id}.glb",
            str(arm))
        self.enable_table(enable=True)
        plate_p = self.plate.get_pose().p
        plate_target = [plate_p[0], plate_p[1], plate_p[2] + 0.02, 0, 0, 0, 1]
        self.move(self.place_actor(self.apple, arm_tag=arm, target_pose=plate_target,
            constrain="align", pre_dis=0.08, dis=0.005))
        self.move(self.back_to_origin(arm))

        # Step 2: bread -> board
        arm = ArmTag("right" if self.bread.get_pose().p[0] > 0 else "left")
        self.grasp_actor_from_table(self.bread, arm_tag=arm, pre_grasp_dis=0.07)
        self.move(self.move_by_displacement(arm_tag=arm, z=0.1))
        self.attach_object(self.bread,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/075_bread/collision/base{self.bread_id}.glb",
            str(arm))
        self.enable_table(enable=True)
        board_p = self.board.get_pose().p
        board_target = [board_p[0], board_p[1], board_p[2] + 0.02, 0, 0, 0, 1]
        self.move(self.place_actor(self.bread, arm_tag=arm, target_pose=board_target,
            constrain="align", pre_dis=0.05, dis=0.005))
        self.move(self.back_to_origin(arm))

        # Step 3: knife -> next to board
        arm = ArmTag("right" if self.knife.get_pose().p[0] > 0 else "left")
        side = 0.12 if arm == "right" else -0.12
        knife_target = [board_p[0] + side, board_p[1], board_p[2] + 0.015, 0, 0, 0, 1]
        self.grasp_actor_from_table(self.knife, arm_tag=arm, pre_grasp_dis=0.07)
        self.move(self.move_by_displacement(arm_tag=arm, z=0.1))
        self.attach_object(self.knife,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/034_knife/collision/base0.glb",
            str(arm))
        self.enable_table(enable=True)
        self.move(self.place_actor(self.knife, arm_tag=arm, target_pose=knife_target,
            constrain="align", pre_dis=0.05, dis=0.005))

    def check_success(self):
        apple_p = self.apple.get_pose().p
        plate_p = self.plate.get_pose().p
        bread_p = self.bread.get_pose().p
        board_p = self.board.get_pose().p
        knife_p = self.knife.get_pose().p

        apple_on_plate = (abs(apple_p[0] - plate_p[0]) < 0.06
                          and abs(apple_p[1] - plate_p[1]) < 0.06
                          and apple_p[2] > plate_p[2] - 0.02)
        bread_on_board = (abs(bread_p[0] - board_p[0]) < 0.08
                          and abs(bread_p[1] - board_p[1]) < 0.08
                          and bread_p[2] > board_p[2] - 0.02)
        knife_dx = abs(knife_p[0] - board_p[0])
        knife_dy = abs(knife_p[1] - board_p[1])
        knife_beside = (0.06 <= knife_dx <= 0.18 and knife_dy <= 0.12)

        return (apple_on_plate and bread_on_board and knife_beside
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
