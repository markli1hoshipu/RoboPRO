from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class put_spoon_on_plate_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        # Plate on counter (static target)
        self.plate_id = 0
        plate_pose = self._safe_rand_pose(
            xlim=[-0.35, 0.35],
            ylim=[-0.20, -0.05],
            zlim=[0.743 + self.table_z_bias],
            qpos=[0.7071, 0.7071, 0, 0],
            rotate_rand=False,
            obj_radius=0.08,
        )
        self.plate = create_actor(
            scene=self,
            pose=plate_pose,
            modelname="003_plate",
            convex=True,
            model_id=self.plate_id,
            is_static=True,
        )
        self.add_prohibit_area(self.plate, padding=0.12, area="table")

        # Spoon on counter — near plate so one arm can reach both
        self.spoon_id = 0
        plate_x = self.plate.get_pose().p[0]
        spoon_pose = self._safe_rand_pose(
            xlim=[max(-0.35, plate_x - 0.40), min(0.35, plate_x + 0.40)],
            ylim=[-0.20, -0.05],
            zlim=[0.743 + self.table_z_bias],
            qpos=[0.5, 0.5, -0.5, -0.5],
            rotate_rand=False,
            obj_radius=0.10,
        )
        self.spoon = create_actor(
            scene=self,
            pose=spoon_pose,
            modelname="134_spoon",
            convex=True,
            model_id=self.spoon_id,
        )
        self.spoon.set_mass(0.05)
        self.add_prohibit_area(self.spoon, padding=0.06, area="table")
        self.collision_list.append((
            self.spoon,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/134_spoon/collision/base{self.spoon_id}.glb",
            self.spoon.scale,
        ))

        plate_p = self.plate.get_pose().p.tolist()
        self.target_pose = plate_p + [1, 0, 0, 0]
        self.target_pose[2] += 0.02

    def play_once(self):
        # Select arm — both objects nearby
        mid_x = (self.spoon.get_pose().p[0] + self.plate.get_pose().p[0]) / 2
        arm_tag = ArmTag("right" if mid_x > 0 else "left")

        # 1. Grasp
        self.move(self.grasp_actor(self.spoon, arm_tag=arm_tag, pre_grasp_dis=0.08))
        if not self.plan_success:
            return self.info

        # 2. Lift
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))
        if not self.plan_success:
            return self.info

        # 3. Attach
        self.attach_object(
            self.spoon,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/134_spoon/collision/base{self.spoon_id}.glb",
            str(arm_tag),
        )

        # 4. Place on plate
        self.move(
            self.place_actor(
                self.spoon,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                constrain="align",
                pre_dis=0.07,
                dis=0.005,
            ))

        # 5. Detach
        self.detach_object(arms_tag=str(arm_tag))

        self.info["info"] = {
            "{A}": f"134_spoon/base{self.spoon_id}",
            "{B}": f"003_plate/base{self.plate_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        spoon_pos = self.spoon.get_pose().p
        plate_pos = self.plate.get_pose().p
        eps = 0.05
        return (
            np.all(np.abs(spoon_pos[:2] - plate_pos[:2]) < eps)
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
