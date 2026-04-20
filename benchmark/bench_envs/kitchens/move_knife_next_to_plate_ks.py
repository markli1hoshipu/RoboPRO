from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
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

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        rand_pos = self.rand_pose_on_counter(
            xlim=[-0.45, 0.45],
            ylim=[-0.23, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
            obj_padding=0.04,
        )
        while abs(rand_pos.p[0]) < 0.3:
            rand_pos = self.rand_pose_on_counter(
                xlim=[-0.45, 0.45],
                ylim=[-0.23, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
                obj_padding=0.04,
            )

        self.knife_id = 0
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="034_knife",
            convex=True,
            model_id=self.knife_id,
        )
        self.target_obj.set_mass(0.05)

        target_rand_pose = self.rand_pose_on_counter(
            xlim=[0],
            ylim=[-0.23, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            obj_padding=0.08,
        )

        self.plate_id = 0
        self.des_obj = create_actor(
            scene=self,
            pose=target_rand_pose,
            modelname="003_plate",
            convex=True,
            model_id=self.plate_id,
            is_static=True,
        )
        self.add_prohibit_area(self.des_obj, padding=0.01, area="table")
        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")

        # Target pose is offset laterally from the plate — knife should sit
        # beside the plate (right if carried by right arm, left otherwise),
        # not on top of it. Side chosen in play_once from knife's x sign.
        self.des_obj_pose = self.des_obj.get_pose().p.tolist() + [0, 0, 0, 1]
        self.des_obj_pose[2] += 0.02

    def play_once(self):
        arm_tag = ArmTag("right" if self.target_obj.get_pose().p[0] > 0 else "left")

        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.1)

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))

        self.attach_object(
            self.target_obj,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/034_knife/collision/base{self.knife_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        # Place beside the plate: +x of plate if right arm, -x if left arm.
        lateral = 0.10 if str(arm_tag) == "right" else -0.10
        plate_p = self.des_obj.get_pose().p
        self.target_pose = [
            float(plate_p[0]) + lateral,
            float(plate_p[1]),
            self.kitchens_info["table_height"] + self.table_z_bias + 0.015,
            0, 0, 0, 1,
        ]

        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                constrain="align",
                pre_dis=0.05,
                dis=0.005,
            ))

    def check_success(self):
        knife_p = self.target_obj.get_pose().p
        plate_p = self.des_obj.get_pose().p
        dx = abs(knife_p[0] - plate_p[0])
        dy = abs(knife_p[1] - plate_p[1])

        # Knife must sit beside the plate on the counter — x offset in the
        # adjacent band (not overlapping, not far), y nearly aligned.
        return (0.04 <= dx <= 0.12
                and dy <= 0.08
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
