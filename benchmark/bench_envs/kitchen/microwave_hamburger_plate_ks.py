from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class microwave_hamburger_plate_ks(KitchenS_base_task):
    """Mid-range: open microwave, take hamburger from inside, place on plate, close microwave."""

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.joint_lower = self.microwave_joint_lower
        self.joint_upper = self.microwave_joint_upper
        self.joint_range = self.microwave_joint_range

        # Microwave starts closed
        mw_pos = self.microwave.get_pose().p
        self.microwave_interior = [mw_pos[0], mw_pos[1], mw_pos[2] + 0.05]

        # Hamburger inside microwave (pre-set door open temporarily for placement, then close)
        open_angle = self.joint_lower + 0.95 * self.joint_range
        limits = self.microwave.get_qlimits()
        ndof = len(limits)
        qpos = [0.0] * ndof
        qpos[0] = open_angle
        self.microwave.set_qpos(qpos)

        self.hamburg_id = np.random.randint(0, 6)
        hamburg_pose = sapien.Pose(
            p=[mw_pos[0], mw_pos[1], mw_pos[2] + 0.05],
            q=[0.7071, 0.7071, 0, 0],
        )
        self.hamburg = create_actor(
            scene=self, pose=hamburg_pose, modelname="006_hamburg",
            convex=True, model_id=self.hamburg_id,
        )
        self.hamburg.set_mass(0.08)
        self.collision_list.append((
            self.hamburg,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/006_hamburg/collision/base{self.hamburg_id}.glb",
            self.hamburg.scale,
        ))

        # Close door after placing hamburger
        qpos[0] = 0.0
        self.microwave.set_qpos(qpos)

        # Plate on counter
        self.plate_id = 0
        plate_pose = self._safe_rand_pose(
            xlim=[-0.15, 0.05],
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

    def play_once(self):
        # Step 1: Open microwave (left arm) — same pattern as open_microwave_ks
        open_arm = ArmTag("left")
        self.move(self.grasp_actor(self.microwave, arm_tag=open_arm, pre_grasp_dis=0.08, contact_point_id=0))
        start_qpos = self.microwave.get_qpos()[0]
        for _ in range(50):
            self.move(self.grasp_actor(
                self.microwave, arm_tag=open_arm,
                pre_grasp_dis=0.0, grasp_dis=0.0, contact_point_id=4,
            ))
            new_qpos = self.microwave.get_qpos()[0]
            if new_qpos - start_qpos <= 0.001:
                break
            start_qpos = new_qpos
            if not self.plan_success:
                break
            if self.microwave.get_qpos()[0] >= self.joint_upper * 0.8:
                break

        if self.microwave.get_qpos()[0] < self.joint_upper * 0.8:
            self.plan_success = True
            self.move(self.open_gripper(arm_tag=open_arm))
            self.move(self.move_by_displacement(arm_tag=open_arm, y=-0.05, z=0.05))
            self.move(self.grasp_actor(self.microwave, arm_tag=open_arm, contact_point_id=1))
            self.move(self.grasp_actor(self.microwave, arm_tag=open_arm, pre_grasp_dis=0.02, contact_point_id=1))
            start_qpos = self.microwave.get_qpos()[0]
            for _ in range(30):
                self.move(self.grasp_actor(
                    self.microwave, arm_tag=open_arm,
                    pre_grasp_dis=0.0, grasp_dis=0.0, contact_point_id=2,
                ))
                new_qpos = self.microwave.get_qpos()[0]
                if new_qpos - start_qpos <= 0.001:
                    break
                start_qpos = new_qpos
                if not self.plan_success:
                    break
                if self.microwave.get_qpos()[0] >= self.joint_upper * 0.8:
                    break

        if not self.plan_success:
            return self.info

        self.move(self.open_gripper(arm_tag=open_arm))
        self.move(self.move_by_displacement(arm_tag=open_arm, y=-0.10, z=0.05))

        # Step 2: Take hamburger from microwave, place on plate (right arm)
        place_arm = ArmTag("right")
        self.move(self.grasp_actor(self.hamburg, arm_tag=place_arm, pre_grasp_dis=0.08))
        self.move(self.move_by_displacement(arm_tag=place_arm, z=0.08))

        self.attach_object(
            self.hamburg,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/006_hamburg/collision/base{self.hamburg_id}.glb",
            str(place_arm),
        )

        plate_p = self.plate.get_pose().p.tolist()
        plate_target = plate_p + [1, 0, 0, 0]
        plate_target[2] += 0.02

        self.move(self.place_actor(
            self.hamburg, arm_tag=place_arm, target_pose=plate_target,
            constrain="align", pre_dis=0.07, dis=0.005,
        ))
        self.detach_object(arms_tag=str(place_arm))

        if not self.plan_success:
            return self.info

        # Step 3: Close microwave (right arm)
        close_arm = ArmTag("right")
        self.move(self.grasp_actor(self.microwave, arm_tag=close_arm, pre_grasp_dis=0.10))
        self.move(self.move_by_displacement(arm_tag=close_arm, y=0.15))
        self.detach_object(arms_tag=str(close_arm))

        self.info["info"] = {
            "{A}": f"006_hamburg/base{self.hamburg_id}",
            "{B}": f"003_plate/base{self.plate_id}",
            "{C}": f"044_microwave/base{self.microwave_model_id}",
            "{a}": "right",
        }
        return self.info

    def check_success(self):
        hamburg_pos = self.hamburg.get_pose().p
        plate_pos = self.plate.get_pose().p
        hamburg_on_plate = np.all(np.abs(hamburg_pos[:2] - plate_pos[:2]) < 0.06)
        door_closed = self.microwave.get_qpos()[0] < self.joint_lower + 0.2 * self.joint_range
        return (
            hamburg_on_plate and door_closed
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
