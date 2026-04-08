from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class put_bread_on_board_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        # Cutting board (static target) — scale set in model_data0.json
        self.board_model_id = 0
        board_pose = self._safe_rand_pose(
            xlim=[-0.35, 0.35],
            ylim=[-0.20, -0.05],
            zlim=[0.743 + self.table_z_bias],
            qpos=[0.7071, 0.7071, 0, 0],
            rotate_rand=False,
            obj_radius=0.10,
        )
        self.board = create_actor(
            scene=self,
            pose=board_pose,
            modelname="104_board",
            convex=True,
            model_id=self.board_model_id,
            is_static=True,
        )
        self.board_pose = self.board.get_pose()
        self.add_prohibit_area(self.board, padding=0.03, area="table")

        # Bread on counter — near board so one arm can reach both
        board_x = self.board_pose.p[0]
        bread_xlim = [max(-0.35, board_x - 0.40), min(0.35, board_x + 0.40)]

        self.bread_id = np.random.randint(0, 7)
        bread_pose = self._safe_rand_pose(
            xlim=bread_xlim,
            ylim=[-0.20, 0.05],
            zlim=[0.743 + self.table_z_bias],
            qpos=[0.7071, 0.7071, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, math.pi / 4],
        )
        self.bread = create_actor(
            scene=self,
            pose=bread_pose,
            modelname="075_bread",
            convex=True,
            model_id=self.bread_id,
        )
        self.bread.set_mass(0.10)
        self.add_prohibit_area(self.bread, padding=0.03, area="table")
        self.collision_list.append((
            self.bread,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/075_bread/collision/base{self.bread_id}.glb",
            self.bread.scale,
        ))

        # Target: board position
        board_p = self.board_pose.p.tolist()
        self.target_pose = board_p + [1, 0, 0, 0]
        self.target_pose[2] += 0.02

    def play_once(self):
        # Select arm — both objects on same side
        mid_x = (self.bread.get_pose().p[0] + self.board_pose.p[0]) / 2
        arm_tag = ArmTag("right" if mid_x > 0 else "left")

        # 1. Grasp bread
        self.move(self.grasp_actor(self.bread, arm_tag=arm_tag, pre_grasp_dis=0.1))
        if not self.plan_success:
            return self.info

        # 2. Lift
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))
        if not self.plan_success:
            return self.info

        # 3. Attach for collision-aware transport
        self.attach_object(
            self.bread,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/075_bread/collision/base{self.bread_id}.glb",
            str(arm_tag),
        )

        # 4. Place on board
        self.move(
            self.place_actor(
                self.bread,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                constrain="align",
                pre_dis=0.07,
                dis=0.005,
            ))

        # 5. Detach
        self.detach_object(arms_tag=str(arm_tag))

        self.info["info"] = {
            "{A}": f"075_bread/base{self.bread_id}",
            "{B}": f"104_board/base{self.board_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        bread_pos = self.bread.get_pose().p
        board_pos = self.board_pose.p
        eps = 0.05
        return (
            np.all(np.abs(bread_pos[:2] - board_pos[:2]) < eps)
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
