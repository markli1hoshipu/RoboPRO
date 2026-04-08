from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import math
import numpy as np
from envs._GLOBAL_CONFIGS import *
import glob


class apple_bin_plate_knife_sink_ks(KitchenS_base_task):
    """Mid-range: drop apple in bin, then put plate and knife in sink."""

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
        self._attached = False

    def load_actors(self):
        # Apple on counter (left side)
        self.apple_id = 0
        apple_pose = self._safe_rand_pose(
            xlim=[-0.20, -0.05],
            ylim=[-0.20, -0.05],
            zlim=[0.743 + self.table_z_bias],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, math.pi / 4],
        )
        self.apple = create_actor(
            scene=self, pose=apple_pose, modelname="035_apple",
            convex=True, model_id=self.apple_id,
        )
        self.apple.set_mass(0.15)
        self.add_prohibit_area(self.apple, padding=0.04, area="table")
        self.collision_list.append((
            self.apple,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/035_apple/collision/base{self.apple_id}.glb",
            self.apple.scale,
        ))

        # Plate on counter (center)
        self.plate_id = 0
        plate_pose = self._safe_rand_pose(
            xlim=[0.0, 0.15],
            ylim=[-0.15, 0.0],
            zlim=[0.743 + self.table_z_bias],
            qpos=[0.7071, 0.7071, 0, 0],
            rotate_rand=False,
        )
        self.plate = create_actor(
            scene=self, pose=plate_pose, modelname="003_plate",
            convex=True, model_id=self.plate_id,
        )
        self.plate.set_mass(0.1)
        self.add_prohibit_area(self.plate, padding=0.06, area="table")
        self.collision_list.append((
            self.plate,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/003_plate/collision/base{self.plate_id}.glb",
            self.plate.scale,
        ))
        self.stabilize_object(self.plate)

        # Knife on counter (right side)
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

    def _drop_in_bin(self, obj, obj_name, obj_id, arm_tag):
        """Grasp object, kinematic move above bin, drop."""
        self.move(self.grasp_actor(obj, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))
        if not self.plan_success:
            return

        collision_path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{obj_name}/collision/base{obj_id}.glb"
        self._start_kinematic_attach(obj, arm_tag)
        self.attach_object(obj, collision_path, str(arm_tag))

        bin_p = self.bin_pose.p
        self._kinematic_move(arm_tag, [
            Action(arm_tag, "move", target_pose=[bin_p[0], bin_p[1], bin_p[2] + 0.25, 1, 0, 0, 0]),
        ])

        self.move((arm_tag, [Action(arm_tag, "open", target_gripper_pos=1.0)]))
        self._stop_kinematic_attach()
        for _ in range(200):
            self.scene.step()
        self.detach_object(arms_tag=str(arm_tag))

    def _put_in_sink(self, obj, obj_name, obj_id, arm_tag):
        """Top-down grasp object, kinematic move above sink, drop."""
        self.move(self._top_down_grasp(obj, arm_tag, grasp_z=0.05))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15))
        if not self.plan_success:
            return

        collision_path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{obj_name}/collision/base{obj_id}.glb"
        self._start_kinematic_attach(obj, arm_tag)
        self.attach_object(obj, collision_path, str(arm_tag))

        sink_p = self.sink_pose.p
        self._kinematic_move(arm_tag, [
            Action(arm_tag, "move", target_pose=[sink_p[0], sink_p[1], sink_p[2] + 0.25, 0.707, 0, 0, 0.707]),
        ])

        self.move((arm_tag, [Action(arm_tag, "open", target_gripper_pos=1.0)]))
        self._stop_kinematic_attach()
        for _ in range(200):
            self.scene.step()
        self.detach_object(arms_tag=str(arm_tag))

    def play_once(self):
        # Step 1: drop apple in bin
        self._drop_in_bin(self.apple, "035_apple", self.apple_id, ArmTag("right"))
        if not self.plan_success:
            return self.info

        # Step 2: put plate in sink
        self._put_in_sink(self.plate, "003_plate", self.plate_id,
                          ArmTag("right" if self.plate.get_pose().p[0] > 0 else "left"))
        if not self.plan_success:
            return self.info

        # Step 3: put knife in sink
        self._put_in_sink(self.knife, "034_knife", self.knife_id,
                          ArmTag("right" if self.knife.get_pose().p[0] > 0 else "left"))

        self.info["info"] = {
            "{A}": f"035_apple/base{self.apple_id}",
            "{B}": f"003_plate/base{self.plate_id}",
            "{C}": f"034_knife/base{self.knife_id}",
            "{a}": "right",
        }
        return self.info

    def check_success(self):
        apple_pos = self.apple.get_pose().p
        plate_pos = self.plate.get_pose().p
        knife_pos = self.knife.get_pose().p
        bin_p = self.bin_pose.p
        sink_p = self.sink_pose.p

        apple_in_bin = abs(apple_pos[0] - bin_p[0]) < 0.15 and abs(apple_pos[1] - bin_p[1]) < 0.15 and apple_pos[2] < 0.5
        plate_in_sink = abs(plate_pos[0] - sink_p[0]) < 0.15 and abs(plate_pos[1] - sink_p[1]) < 0.15
        knife_in_sink = abs(knife_pos[0] - sink_p[0]) < 0.15 and abs(knife_pos[1] - sink_p[1]) < 0.15

        return (
            apple_in_bin and plate_in_sink and knife_in_sink
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
