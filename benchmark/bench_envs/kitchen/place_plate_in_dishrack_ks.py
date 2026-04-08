from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class place_plate_in_dishrack_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        # Plate on counter
        self.plate_id = 0
        plate_pose = self._safe_rand_pose(
            xlim=[-0.15, 0.15],
            ylim=[-0.20, 0.0],
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
        )
        self.plate.set_mass(0.1)
        self.add_prohibit_area(self.plate, padding=0.06, area="table")
        self.collision_list.append((
            self.plate,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/003_plate/collision/base{self.plate_id}.glb",
            self.plate.scale,
        ))

        # Target: dish rack position
        rack_pos = self.dish_rack.get_pose().p
        self.target_pose = [rack_pos[0], rack_pos[1], rack_pos[2] + 0.03, 1, 0, 0, 0]

    def play_once(self):
        arm_tag = ArmTag("right" if self.plate.get_pose().p[0] > 0 else "left")

        self.move(self.grasp_actor(self.plate, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))

        self.attach_object(
            self.plate,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/003_plate/collision/base{self.plate_id}.glb",
            str(arm_tag),
        )

        self.move(
            self.place_actor(
                self.plate,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                constrain="align",
                pre_dis=0.07,
                dis=0.005,
            ))

        self.detach_object(arms_tag=str(arm_tag))

        self.info["info"] = {
            "{A}": f"003_plate/base{self.plate_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        plate_pos = self.plate.get_pose().p
        rack_pos = self.dish_rack.get_pose().p
        eps = 0.06
        return (
            np.all(np.abs(plate_pos[:2] - rack_pos[:2]) < eps)
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
