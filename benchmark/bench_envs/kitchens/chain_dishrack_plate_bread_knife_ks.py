from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien, math, os, glob
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy


class chain_dishrack_plate_bread_knife_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.plate.get_name(), self.bread.get_name(), self.knife.get_name()}

    def load_actors(self):
        # Plate on the dishrack
        rack_p = self.dishrack.get_pose().p
        plate_start = sapien.Pose(
            [rack_p[0], rack_p[1] - 0.09, rack_p[2] + 0.06],
            [0.5, 0.5, 0.5, 0.5],
        )
        self.plate = create_actor(
            scene=self, pose=plate_start, modelname="003_plate",
            convex=True, model_id=0,
        )
        self.plate.set_mass(0.1)

        # Pick a target counter position to drop the plate onto, on the side
        # opposite the dishrack (so the arm has a clean approach).
        plate_target_x = 0.25 if rack_p[0] < 0 else -0.25
        plate_target_y = -0.05
        table_top_z = self.kitchens_info["table_height"] + self.table_z_bias
        self.plate_counter_pose = [plate_target_x, plate_target_y, table_top_z + 0.02, 0, 0, 0, 1]

        # Bread on counter — keep away from plate_target_xy
        bread_pose = self.rand_pose_on_counter(
            xlim=[-0.32, 0.32], ylim=[-0.15, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, np.pi/2, 0],
            obj_padding=0.05,
        )
        while (abs(bread_pose.p[0] - plate_target_x) < 0.12
               and abs(bread_pose.p[1] - plate_target_y) < 0.10):
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
        while (abs(knife_pose.p[0] - plate_target_x) < 0.12
               and abs(knife_pose.p[1] - plate_target_y) < 0.10):
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
        # Step 1: plate from dishrack → counter
        arm = ArmTag("right" if self.dishrack.get_pose().p[0] < 0 else "left")
        # Actually use the target position to pick the arm:
        arm = ArmTag("right" if self.plate_counter_pose[0] > 0 else "left")
        self.grasp_actor_from_table(self.plate, arm_tag=arm, pre_grasp_dis=0.10)
        self.move(self.move_by_displacement(arm_tag=arm, z=0.20))
        self.attach_object(self.plate,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/003_plate/collision/base0.glb",
            str(arm))
        self.enable_table(enable=True)
        self.move(self.place_actor(self.plate, arm_tag=arm, target_pose=self.plate_counter_pose,
            constrain="align", pre_dis=0.08, dis=0.005))
        self.move(self.back_to_origin(arm))

        # Refresh actual plate pose (the place may have shifted slightly)
        plate_p = self.plate.get_pose().p
        bread_target = [plate_p[0], plate_p[1], plate_p[2] + 0.02, 0, 0, 0, 1]

        # Step 2: bread → plate
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

        # Step 3: knife → beside plate (on the side opposite the arm that carried
        # plate; offset by ±0.10 in x)
        arm = ArmTag("right" if self.knife.get_pose().p[0] > 0 else "left")
        side = 0.10 if arm == "right" else -0.10
        knife_target = [plate_p[0] + side, plate_p[1], plate_p[2] + 0.015, 0, 0, 0, 1]
        self.grasp_actor_from_table(self.knife, arm_tag=arm, pre_grasp_dis=0.07)
        self.move(self.move_by_displacement(arm_tag=arm, z=0.15))
        self.attach_object(self.knife,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/034_knife/collision/base0.glb",
            str(arm))
        self.enable_table(enable=True)
        self.move(self.place_actor(self.knife, arm_tag=arm, target_pose=knife_target,
            constrain="align", pre_dis=0.05, dis=0.005))

    def check_success(self):
        plate_p = self.plate.get_pose().p
        bread_p = self.bread.get_pose().p
        knife_p = self.knife.get_pose().p
        rack_p = self.dishrack.get_pose().p

        # Plate is OFF the rack (not in rack footprint)
        plate_off_rack = (abs(plate_p[0] - rack_p[0]) > 0.12
                          or abs(plate_p[1] - (rack_p[1] - 0.09)) > 0.12)
        # Bread on plate
        bread_on_plate = (abs(bread_p[0] - plate_p[0]) < 0.06
                          and abs(bread_p[1] - plate_p[1]) < 0.06)
        # Knife beside plate (not ON it)
        knife_dx = abs(knife_p[0] - plate_p[0])
        knife_dy = abs(knife_p[1] - plate_p[1])
        knife_beside = (0.05 <= knife_dx <= 0.15 and knife_dy <= 0.10)

        return (plate_off_rack and bread_on_plate and knife_beside
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
