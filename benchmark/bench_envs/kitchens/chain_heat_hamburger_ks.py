from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien, math, os, glob
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy


class chain_heat_hamburger_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.hamburger.get_name()}

    def load_actors(self):
        # Hamburger on counter
        h_pose = self.rand_pose_on_counter(
            xlim=[-0.32, 0.32], ylim=[-0.15, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, np.pi, 0],
            obj_padding=0.06,
        )
        while abs(h_pose.p[0]) < 0.3:
            h_pose = self.rand_pose_on_counter(
                xlim=[-0.4, 0.4], ylim=[-0.15, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, np.pi, 0],
                obj_padding=0.06,
            )
        self.hamburger_id = int(np.random.choice([0, 1, 2]))
        self.hamburger = create_actor(
            scene=self, pose=h_pose, modelname="006_hamburg",
            convex=True, model_id=self.hamburger_id,
        )
        self.hamburger.set_mass(0.05)
        self.add_prohibit_area(self.hamburger, padding=0.02, area="table")

        mw_p = self.microwave.get_pose().p
        self.mw_target_pose = [mw_p[0], mw_p[1] + 0.05, mw_p[2] + 0.05, 0, 0, 0, 1]

    def play_once(self):
        # Step 1: open microwave door (left arm is canonical in open_microwave_ks)
        door_arm = ArmTag("left")
        self.move(self.grasp_actor(self.microwave, arm_tag=door_arm, pre_grasp_dis=0.08, contact_point_id=0))
        start_qpos = self.microwave.get_qpos()[0]
        for _ in range(50):
            self.move(self.grasp_actor(self.microwave, arm_tag=door_arm,
                pre_grasp_dis=0.0, grasp_dis=0.0, contact_point_id=4))
            new_qpos = self.microwave.get_qpos()[0]
            if new_qpos - start_qpos <= 0.001:
                break
            start_qpos = new_qpos
            if not self.plan_success:
                break
            limits = self.microwave.get_qlimits()
            if self.microwave.get_qpos()[0] >= limits[0][1] * 0.7:
                break
        # Open gripper and retreat the door arm
        self.move(self.open_gripper(arm_tag=door_arm))
        self.move(self.move_by_displacement(arm_tag=door_arm, z=0.1))
        self.move(self.back_to_origin(door_arm))

        # Step 2: hamburger → microwave (use arm opposite to hamburger x)
        arm = ArmTag("right" if self.hamburger.get_pose().p[0] > 0 else "left")
        self.grasp_actor_from_table(self.hamburger, arm_tag=arm, pre_grasp_dis=0.08)
        self.move(self.move_by_displacement(arm_tag=arm, z=0.20))
        self.attach_object(self.hamburger,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/006_hamburg/collision/base{self.hamburger_id}.glb",
            str(arm))
        self.enable_table(enable=True)
        self.move(self.place_actor(self.hamburger, arm_tag=arm,
            target_pose=self.mw_target_pose,
            constrain="align", pre_dis=0.08, dis=0.02))
        self.move(self.back_to_origin(arm))

        # Step 3: close the microwave door (follow close_microwave_ks: grasp handle,
        # push toward body until qpos <= 0.1 * limit)
        door_arm = ArmTag("left")
        self.move(self.grasp_actor(self.microwave, arm_tag=door_arm, pre_grasp_dis=0.08, contact_point_id=0))
        limits = self.microwave.get_qlimits()
        for _ in range(15):
            self.move(self.move_by_displacement(arm_tag=door_arm, x=0.02))
            if self.microwave.get_qpos()[0] <= limits[0][1] * 0.1:
                break
            if not self.plan_success:
                break

    def check_success(self):
        hp = self.hamburger.get_pose().p
        mw_p = self.microwave.get_pose().p
        limits = self.microwave.get_qlimits()
        burger_in_mw = (abs(hp[0] - mw_p[0]) < 0.10
                        and abs(hp[1] - mw_p[1]) < 0.15
                        and hp[2] < mw_p[2] + 0.15
                        and hp[2] > mw_p[2] - 0.05)
        door_closed = self.microwave.get_qpos()[0] <= limits[0][1] * 0.2
        return (burger_in_mw and door_closed
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
