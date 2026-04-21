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

    def check_stable(self):
        is_stable, unstable_list = super().check_stable()
        unstable_list = [n for n in unstable_list if "034_knife" not in n]
        return len(unstable_list) == 0, unstable_list

    def load_actors(self):
        # Plate: place somewhere along the counter (not pinned to the middle).
        # The arm chosen in play_once is based on knife x-sign, so plate can
        # be anywhere as long as the knife has clearance to its side.
        target_rand_pose = self.rand_pose_on_counter(
            xlim=[-0.20, 0.20],
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

        # Knife: spawn on either side of the plate so either arm can be used.
        plate_x = float(self.des_obj.get_pose().p[0])
        # Pick a side at random; 60% chance spawn to the side with more room.
        if np.random.rand() < 0.5:
            knife_xlim = [plate_x + 0.14, plate_x + 0.22]
        else:
            knife_xlim = [plate_x - 0.22, plate_x - 0.14]
        # Clamp to counter lims so we don't fall off the edge.
        knife_xlim = [max(-0.32, knife_xlim[0]), min(0.32, knife_xlim[1])]
        rand_pos = self.rand_pose_on_counter(
            xlim=knife_xlim,
            ylim=[-0.15, 0.0],
            qpos=[0.7071068, 0.0, -0.7071068, 0.0],
            rotate_rand=False,
            obj_padding=0.08,
        )

        self.knife_id = 0
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="034_knife",
            convex=True,
            model_id=self.knife_id,
            scale=0.150,
        )
        self.target_obj.set_mass(0.05)

        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")

        self.des_obj_pose = self.des_obj.get_pose().p.tolist() + [0, 0, 0, 1]
        self.des_obj_pose[2] += 0.02

    def play_once(self):
        arm_side = "right" if self.target_obj.get_pose().p[0] > 0 else "left"
        print(arm_side)
        arm_tag = ArmTag(arm_side)

        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.1)

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))

        self.attach_object(
            self.target_obj,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/034_knife/collision/base{self.knife_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        side_sign = 1 if str(arm_tag) == "right" else -1
        plate_p = self.des_obj.get_pose().p
        counter_z = self.kitchens_info["table_height"] + self.table_z_bias

        # Move gripper to target position, keeping its current (top-down) quat.
        cur_tcp = (self.robot.get_right_ee_pose() if str(arm_tag) == "right"
                   else self.robot.get_left_ee_pose())
        cur_q = list(cur_tcp[3:7])
        cur_q = [0.7071068, 0.0, -0.7071068, 0.0]  # Override to top-down for better IK
        cur_q = [1,0,0,0]

        target_tcp_xyz = [
            float(plate_p[0]) + 0.15 * side_sign,
            float(plate_p[1]),
            counter_z+0.20,
        ]

        self.move(self.move_to_pose(arm_tag, target_tcp_xyz + cur_q))
        self.move(self.open_gripper(arm_tag, pos=1.0))

    def check_success(self):
        knife_p = self.target_obj.get_pose().p
        plate_p = self.des_obj.get_pose().p
        dx = abs(knife_p[0] - plate_p[0])
        dy = abs(knife_p[1] - plate_p[1])

        return (0.04 <= dx <= 0.14
                and dy <= 0.10
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
