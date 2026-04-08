from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class plate_bread_knife_ks(KitchenS_base_task):
    """Mid-range: plate from dishrack to counter, bread on plate, knife next to plate."""

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        # Plate starts in/on dish rack
        rack_pos = self.dish_rack.get_pose().p
        self.plate_id = 0
        plate_pose = sapien.Pose(
            p=[rack_pos[0], rack_pos[1], rack_pos[2] + 0.03],
            q=[0.7071, 0.7071, 0, 0],
        )
        self.plate = create_actor(
            scene=self, pose=plate_pose, modelname="003_plate",
            convex=True, model_id=self.plate_id,
        )
        self.plate.set_mass(0.1)
        self.collision_list.append((
            self.plate,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/003_plate/collision/base{self.plate_id}.glb",
            self.plate.scale,
        ))

        # Bread on counter
        self.bread_id = np.random.randint(0, 7)
        bread_pose = self._safe_rand_pose(
            xlim=[-0.25, -0.10],
            ylim=[-0.20, 0.0],
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

        # Target position for plate on counter center
        self.plate_target = [0.0, -0.10, 0.743 + self.table_z_bias + 0.02, 1, 0, 0, 0]

    def play_once(self):
        # Step 1: Move plate from dishrack to counter
        arm1 = ArmTag("left")
        self.move(self.grasp_actor(self.plate, arm_tag=arm1, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm1, z=0.1))
        self.attach_object(
            self.plate,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/003_plate/collision/base{self.plate_id}.glb",
            str(arm1),
        )
        self.move(self.place_actor(
            self.plate, arm_tag=arm1, target_pose=self.plate_target,
            constrain="align", pre_dis=0.07, dis=0.005,
        ))
        self.detach_object(arms_tag=str(arm1))
        if not self.plan_success:
            return self.info

        # Step 2: Put bread on plate
        plate_p = self.plate.get_pose().p.tolist()
        bread_target = plate_p + [1, 0, 0, 0]
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

        # Step 3: Move knife next to plate
        plate_p2 = self.plate.get_pose().p
        knife_target = [plate_p2[0] + 0.08, plate_p2[1], 0.743 + self.table_z_bias + 0.02, 1, 0, 0, 0]

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
            "{A}": f"003_plate/base{self.plate_id}",
            "{B}": f"075_bread/base{self.bread_id}",
            "{C}": f"034_knife/base{self.knife_id}",
            "{a}": "left",
        }
        return self.info

    def check_success(self):
        plate_pos = self.plate.get_pose().p
        bread_pos = self.bread.get_pose().p
        knife_pos = self.knife.get_pose().p

        # Plate on counter (not in rack)
        plate_on_counter = plate_pos[2] < 0.85 and abs(plate_pos[1]) < 0.20
        # Bread on plate
        bread_on_plate = np.all(np.abs(bread_pos[:2] - plate_pos[:2]) < 0.06)
        # Knife near plate
        knife_near_plate = np.all(np.abs(knife_pos[:2] - plate_pos[:2]) < 0.12)

        return (
            plate_on_counter and bread_on_plate and knife_near_plate
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
