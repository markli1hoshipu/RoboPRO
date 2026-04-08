from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import math
import numpy as np
from envs._GLOBAL_CONFIGS import *
import glob


class apple_sink_bread_board_knife_ks(KitchenS_base_task):
    """Mid-range: pick apple from sink to plate, put bread on board, move knife next to board."""

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
        self._attached = False

    def load_actors(self):
        # Apple in sink
        sink_p = self.sink_pose.p
        self.apple_id = 0
        apple_pose = rand_pose(
            xlim=[sink_p[0] - 0.05, sink_p[0] + 0.05],
            ylim=[sink_p[1] - 0.04, sink_p[1] + 0.04],
            zlim=[sink_p[2] + 0.02],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, math.pi / 4],
        )
        self.apple = create_actor(
            scene=self, pose=apple_pose, modelname="035_apple",
            convex=True, model_id=self.apple_id,
        )
        self.apple.set_mass(0.15)
        self.collision_list.append((
            self.apple,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/035_apple/collision/base{self.apple_id}.glb",
            self.apple.scale,
        ))

        # Plate on counter
        self.plate_id = 0
        plate_pose = self._safe_rand_pose(
            xlim=[-0.10, 0.05],
            ylim=[-0.15, 0.0],
            zlim=[0.743 + self.table_z_bias],
            qpos=[0.7071, 0.7071, 0, 0],
            rotate_rand=False,
        )
        self.plate = create_actor(
            scene=self, pose=plate_pose, modelname="003_plate",
            convex=True, model_id=self.plate_id, is_static=True,
        )
        self.add_prohibit_area(self.plate, padding=0.06, area="table")

        # Bread on counter
        self.bread_id = np.random.randint(0, 7)
        bread_pose = self._safe_rand_pose(
            xlim=[-0.25, -0.10],
            ylim=[-0.20, -0.05],
            zlim=[0.743 + self.table_z_bias],
            qpos=[0.7071, 0.7071, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, math.pi / 4],
        )
        self.bread = create_actor(
            scene=self, pose=bread_pose, modelname="075_bread",
            convex=True, model_id=self.bread_id,
        )
        self.bread.set_mass(0.10)
        self.add_prohibit_area(self.bread, padding=0.04, area="table")
        self.collision_list.append((
            self.bread,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/075_bread/collision/base{self.bread_id}.glb",
            self.bread.scale,
        ))

        # Knife on counter
        self.knife_id = 0
        knife_pose = self._safe_rand_pose(
            xlim=[0.15, 0.30],
            ylim=[-0.15, 0.0],
            zlim=[0.743 + self.table_z_bias],
            qpos=[0.7071, 0, -0.7071, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, math.pi / 4],
        )
        self.knife = create_actor(
            scene=self, pose=knife_pose, modelname="034_knife",
            convex=True, model_id=self.knife_id,
        )
        self.knife.set_mass(0.05)
        self.add_prohibit_area(self.knife, padding=0.04, area="table")
        self.collision_list.append((
            self.knife,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/034_knife/collision/base{self.knife_id}.glb",
            self.knife.scale,
        ))
        self.stabilize_object(self.knife)

    def play_once(self):
        # Step 1: Pick apple from sink, place on plate
        arm1 = ArmTag("left")
        self.move(self._top_down_grasp(self.apple, arm1, grasp_z=0.03))
        if not self.plan_success:
            return self.info

        self._start_kinematic_attach(self.apple, arm1)
        self.attach_object(
            self.apple,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/035_apple/collision/base{self.apple_id}.glb",
            str(arm1),
        )

        # Lift out of sink
        self._kinematic_move(arm1, [
            Action(arm1, "move", target_pose=[
                self.apple.get_pose().p[0], self.apple.get_pose().p[1],
                0.743 + self.table_z_bias + 0.25, 0.707, 0, 0.707, 0
            ]),
        ])

        # Place on plate
        plate_p = self.plate.get_pose().p.tolist()
        apple_target = plate_p + [1, 0, 0, 0]
        apple_target[2] += 0.02
        self._kinematic_move(arm1, [Action(arm1, "move", target_pose=apple_target)])
        self.move((arm1, [Action(arm1, "open", target_gripper_pos=1.0)]))
        self._stop_kinematic_attach()
        for _ in range(200):
            self.scene.step()
        self.detach_object(arms_tag=str(arm1))

        if not self.plan_success:
            return self.info

        # Step 2: Put bread on board
        board_p = self.board_pose.p.tolist()
        bread_target = board_p + [1, 0, 0, 0]
        bread_target[2] += 0.02

        arm2 = ArmTag("right" if self.bread.get_pose().p[0] > 0 else "left")
        self.move(self.grasp_actor(self.bread, arm_tag=arm2, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm2, z=0.1))
        self.attach_object(
            self.bread,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/075_bread/collision/base{self.bread_id}.glb",
            str(arm2),
        )
        self.move(self.place_actor(
            self.bread, arm_tag=arm2, target_pose=bread_target,
            constrain="align", pre_dis=0.07, dis=0.005,
        ))
        self.detach_object(arms_tag=str(arm2))

        if not self.plan_success:
            return self.info

        # Step 3: Move knife next to board
        board_p2 = self.board_pose.p
        knife_target = [board_p2[0] + 0.10, board_p2[1], 0.743 + self.table_z_bias + 0.02, 1, 0, 0, 0]

        arm3 = ArmTag("right")
        self.move(self.grasp_actor(self.knife, arm_tag=arm3, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm3, z=0.1))
        self.attach_object(
            self.knife,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/034_knife/collision/base{self.knife_id}.glb",
            str(arm3),
        )
        self.move(self.place_actor(
            self.knife, arm_tag=arm3, target_pose=knife_target,
            constrain="align", pre_dis=0.07, dis=0.005,
        ))
        self.detach_object(arms_tag=str(arm3))

        self.info["info"] = {
            "{A}": f"035_apple/base{self.apple_id}",
            "{B}": f"075_bread/base{self.bread_id}",
            "{C}": f"034_knife/base{self.knife_id}",
            "{a}": "left",
        }
        return self.info

    def check_success(self):
        apple_pos = self.apple.get_pose().p
        plate_pos = self.plate.get_pose().p
        bread_pos = self.bread.get_pose().p
        board_pos = self.board_pose.p
        knife_pos = self.knife.get_pose().p

        apple_on_plate = np.all(np.abs(apple_pos[:2] - plate_pos[:2]) < 0.06) and apple_pos[2] > 0.70
        bread_on_board = np.all(np.abs(bread_pos[:2] - board_pos[:2]) < 0.06)
        knife_near_board = np.all(np.abs(knife_pos[:2] - board_pos[:2]) < 0.15)

        return (
            apple_on_plate and bread_on_board and knife_near_board
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
