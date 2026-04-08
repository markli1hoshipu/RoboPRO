from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class microwave_hamburger_ks(KitchenS_base_task):
    """Mid-range: open microwave, put hamburger inside, close microwave."""

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        # Microwave starts closed (default)
        self.joint_lower = self.microwave_joint_lower
        self.joint_upper = self.microwave_joint_upper
        self.joint_range = self.microwave_joint_range

        # Microwave cavity center
        mw_pos = self.microwave.get_pose().p
        self.microwave_interior = [mw_pos[0], mw_pos[1], mw_pos[2] + 0.05]

        # Hamburger on counter (use _safe_rand_pose to respect prohibited areas)
        self.hamburg_id = np.random.randint(0, 6)
        hamburg_pose = self._safe_rand_pose(
            xlim=[-0.25, -0.10],
            ylim=[-0.15, 0.0],
            zlim=[0.743 + self.table_z_bias],
            rotate_rand=True,
            rotate_lim=[0, 0, math.pi],
            qpos=[0.7071, 0.7071, 0, 0],
        )
        self.hamburg = create_actor(
            scene=self,
            pose=hamburg_pose,
            modelname="006_hamburg",
            convex=True,
            model_id=self.hamburg_id,
            is_static=False,
        )
        self.hamburg.set_mass(0.08)
        self.add_prohibit_area(self.hamburg, padding=0.03, area="table")
        self.collision_list.append((
            self.hamburg,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/006_hamburg/collision/base{self.hamburg_id}.glb",
            self.hamburg.scale,
        ))

    def play_once(self):
        # Step 1: Open microwave (left arm)
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

        # Release open arm
        self.move(self.open_gripper(arm_tag=open_arm))
        self.move(self.move_by_displacement(arm_tag=open_arm, y=-0.10, z=0.05))

        # Step 2: Put hamburger in microwave (right arm)
        place_arm = ArmTag("right")
        self.move(self.grasp_actor(self.hamburg, arm_tag=place_arm, pre_grasp_dis=0.08))
        self.move(self.move_by_displacement(arm_tag=place_arm, z=0.10))

        self.attach_object(
            self.hamburg,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/006_hamburg/collision/base{self.hamburg_id}.glb",
            str(place_arm),
        )

        target_pose = self.microwave_interior + [1, 0, 0, 0]
        self.move(self.place_actor(
            self.hamburg, arm_tag=place_arm, target_pose=target_pose,
            constrain="align", pre_dis=0.08, dis=0.005,
        ))
        self.detach_object(arms_tag=str(place_arm))
        self.move(self.move_by_displacement(arm_tag=place_arm, x=-0.08))

        if not self.plan_success:
            return self.info

        # Step 3: Close microwave (right arm)
        close_arm = ArmTag("right")
        self.move(self.grasp_actor(self.microwave, arm_tag=close_arm, pre_grasp_dis=0.10))
        self.move(self.move_by_displacement(arm_tag=close_arm, y=0.15))
        self.detach_object(arms_tag=str(close_arm))

        self.info["info"] = {
            "{A}": f"006_hamburg/base{self.hamburg_id}",
            "{B}": f"044_microwave/base{self.microwave_model_id}",
            "{a}": "right",
        }
        return self.info

    def check_success(self):
        hamburg_pos = self.hamburg.get_pose().p
        mw_center = np.array(self.microwave_interior)
        inside = (
            abs(hamburg_pos[0] - mw_center[0]) < 0.12
            and abs(hamburg_pos[1] - mw_center[1]) < 0.10
            and abs(hamburg_pos[2] - mw_center[2]) < 0.08
        )
        door_closed = self.microwave.get_qpos()[0] < self.joint_lower + 0.2 * self.joint_range
        return inside and door_closed
