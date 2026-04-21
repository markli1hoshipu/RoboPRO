from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
import numpy as np
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class pick_hamburger_from_microwave_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def _microwave_mesh_name(self) -> str:
        """Reproduce the key used by Bench_base_task.update_world() so we
        can toggle the microwave in the Curobo collision world."""
        pose = self.microwave.get_pose()
        np_pose = np.concatenate([pose.p, pose.q]).tolist()
        return f"{self.microwave.get_name()}_{np_pose}_{self.seed}"

    def _disable_microwave_obstacle(self):
        """Remove the microwave from Curobo's collision list so the arm
        can plan inside the mouth without being rejected."""
        try:
            self.collision_list = [
                e for e in self.collision_list if e.get("actor") is not self.microwave
            ]
            self.update_world()
        except Exception:
            pass

    def load_actors(self):
        # Force microwave door fully open so the interior is accessible.
        # Same recipe as put_hamburger_in_microwave_ks (which is proven).
        limits = self.microwave.get_qlimits()
        qpos_mw = self.microwave.get_qpos()
        qpos_mw[0] = limits[0][1] * 0.95
        self.microwave.set_qpos(qpos_mw)

        mw_p = self.microwave.get_pose().p
        mw_x = float(mw_p[0])
        mw_y = float(mw_p[1])
        mw_z = float(mw_p[2])

        # Geometry (per put_hamburger_in_microwave_ks notes):
        # Microwave is yaw +π/2. Door face at world +y (toward robot).
        # Mouth plane ≈ mw_y + 0.18. Back wall ≈ mw_y - 0.11.
        # A target just inside the mouth sits at y ≈ mw_y + 0.08..0.12.
        #
        # Spawn hamburger just inside the mouth so the arm does not have
        # to reach deep into the cavity (poor IK inside the enclosure).
        self.hamburger_id = int(np.random.choice([0, 1, 2]))
        spawn_x = mw_x + float(np.random.uniform(-0.02, 0.02))
        spawn_y = mw_y + float(np.random.uniform(0.08, 0.12))
        spawn_z = mw_z + 0.01  # just above the microwave floor surface

        hpose = sapien.Pose(
            [spawn_x, spawn_y, spawn_z],
            [0.5, 0.5, 0.5, 0.5],
        )
        self.target_obj = create_actor(
            scene=self,
            pose=hpose,
            modelname="006_hamburg",
            convex=True,
            model_id=self.hamburger_id,
        )
        self.target_obj.set_mass(0.05)

        # Pick the arm on the microwave's side. mw at x=-0.32 → left arm.
        self.arm_tag = ArmTag("right" if mw_x > 0 else "left")

    def play_once(self):
        arm_tag = self.arm_tag
        mw_p = self.microwave.get_pose().p
        mw_y = float(mw_p[1])
        mw_z = float(mw_p[2])

        # Pop the microwave from the Curobo collision world so the arm can
        # plan into the mouth to reach the hamburger. The arm is small
        # relative to the cavity at 1.5× scale, so no real collision risk
        # given the jittered spawn sits just inside the mouth plane.
        self._disable_microwave_obstacle()

        # Use the base-class helper, which also disables the table during
        # the grasp — already validated by move_hamburger_onto_plate_ks
        # (100/100) and put_hamburger_in_microwave_ks.
        self.grasp_actor_from_table(
            self.target_obj,
            arm_tag=arm_tag,
            pre_grasp_dis=0.06,
        )

        if not self.plan_success:
            return

        # Attach the hamburger for Curobo so the retreat plan accounts
        # for it being held.
        self.attach_object(
            self.target_obj,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/006_hamburg/collision/base{self.hamburger_id}.glb",
            str(arm_tag),
        )

        # Lift slightly first to clear the microwave floor, then retreat
        # in +y (mouth direction → toward the robot) to clear the
        # microwave interior, then raise up above the counter.
        # Splitting the motion keeps Curobo from having to solve a long
        # diagonal path from inside the enclosure.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=0.20))

        # Safe to re-enable table now that we have left the microwave.
        self.enable_table(enable=True)
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

    def check_success(self):
        tp = self.target_obj.get_pose().p
        mw_p = self.microwave.get_pose().p
        mw_y = float(mw_p[1])
        table_top_z = self.kitchens_info["table_height"] + self.table_z_bias
        gripper_closed = (
            (not self.robot.is_left_gripper_open())
            if self.arm_tag == "left"
            else (not self.robot.is_right_gripper_open())
        )
        # Success: hamburger lifted above the counter, gripper still
        # holding it, and the hamburger has been drawn past the mouth
        # plane (mw_y + 0.18) out of the interior in +y.
        return (
            tp[2] > table_top_z + 0.08
            and gripper_closed
            and tp[1] > mw_y + 0.18
        )
