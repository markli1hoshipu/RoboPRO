from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class put_hamburger_in_microwave_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        # Force microwave door fully open so the interior is accessible.
        limits = self.microwave.get_qlimits()
        qpos_mw = self.microwave.get_qpos()
        qpos_mw[0] = limits[0][1] * 0.9
        self.microwave.set_qpos(qpos_mw)

        # Spawn hamburger somewhere on the counter, away from the center where
        # the microwave sits.
        rand_pos = self.rand_pose_on_counter(
            xlim=[-0.32, 0.32],
            ylim=[-0.15, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi, 0],
            obj_padding=0.06,
        )
        while abs(rand_pos.p[0]) < 0.3:
            rand_pos = self.rand_pose_on_counter(
                xlim=[-0.4, 0.4],
                ylim=[-0.15, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi, 0],
                obj_padding=0.06,
            )

        # 006_hamburg has three compact variants that all grasp reliably.
        self.hamburger_id = int(np.random.choice([0, 1, 2]))
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="006_hamburg",
            convex=True,
            model_id=self.hamburger_id,
        )
        self.target_obj.set_mass(0.05)

        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")

        # Drop point: just inside the microwave opening, a few cm above the
        # interior floor. The microwave is rotated +90° about z, so the door
        # opens toward +y in world — push the target a bit in +y so the wrist
        # can enter without colliding with the back wall.
        mw_p = self.microwave.get_pose().p
        self.des_obj_pose = [
            float(mw_p[0]),
            float(mw_p[1]) + 0.05,
            float(mw_p[2]) + 0.05,
            0, 0, 0, 1,
        ]

    def play_once(self):
        arm_tag = ArmTag("right" if self.target_obj.get_pose().p[0] > 0 else "left")

        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.08)

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.20))

        self.attach_object(
            self.target_obj,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/006_hamburg/collision/base{self.hamburger_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose=self.des_obj_pose,
                constrain="align",
                pre_dis=0.08,
                dis=0.02,
            ))

    def check_success(self):
        tp = self.target_obj.get_pose().p
        mw_p = self.microwave.get_pose().p
        return (abs(tp[0] - mw_p[0]) < 0.10
                and abs(tp[1] - mw_p[1]) < 0.15
                and tp[2] < mw_p[2] + 0.15
                and tp[2] > mw_p[2] - 0.05
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
