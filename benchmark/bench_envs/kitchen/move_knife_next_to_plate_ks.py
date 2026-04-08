from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class move_knife_next_to_plate_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        # Knife on counter
        self.knife_id = 0
        knife_pose = self._safe_rand_pose(
            xlim=[-0.25, -0.05],
            ylim=[-0.20, 0.05],
            zlim=[0.743 + self.table_z_bias],
            qpos=[0.7071, 0, -0.7071, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, math.pi / 4],
        )
        self.knife = create_actor(
            scene=self,
            pose=knife_pose,
            modelname="034_knife",
            convex=True,
            model_id=self.knife_id,
        )
        self.knife.set_mass(0.05)
        self.add_prohibit_area(self.knife, padding=0.04, area="table")
        self.collision_list.append((
            self.knife,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/034_knife/collision/base{self.knife_id}.glb",
            self.knife.scale,
        ))
        self.stabilize_object(self.knife)

        # Plate on counter
        self.plate_id = 0
        plate_pose = self._safe_rand_pose(
            xlim=[0.05, 0.25],
            ylim=[-0.15, 0.05],
            zlim=[0.743 + self.table_z_bias],
            qpos=[0.7071, 0.7071, 0, 0],
            rotate_rand=False,
        )
        self.plate = create_actor(
            scene=self,
            pose=plate_pose,
            modelname="003_plate",
            convex=True,
            model_id=self.plate_id,
            is_static=True,
        )
        self.add_prohibit_area(self.plate, padding=0.06, area="table")

        # Target: right side of plate
        plate_p = self.plate.get_pose().p
        self.target_pose = [plate_p[0] + 0.08, plate_p[1], 0.743 + self.table_z_bias + 0.02, 1, 0, 0, 0]

    def play_once(self):
        arm_tag = ArmTag("right" if self.knife.get_pose().p[0] > 0 else "left")

        self.move(self.grasp_actor(self.knife, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))

        self.attach_object(
            self.knife,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/034_knife/collision/base{self.knife_id}.glb",
            str(arm_tag),
        )

        self.move(
            self.place_actor(
                self.knife,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                constrain="align",
                pre_dis=0.07,
                dis=0.005,
            ))

        self.detach_object(arms_tag=str(arm_tag))

        self.info["info"] = {
            "{A}": f"034_knife/base{self.knife_id}",
            "{B}": f"003_plate/base{self.plate_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        knife_pos = self.knife.get_pose().p
        plate_pos = self.plate.get_pose().p
        eps = 0.12
        return (
            np.all(np.abs(knife_pos[:2] - plate_pos[:2]) < eps)
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
