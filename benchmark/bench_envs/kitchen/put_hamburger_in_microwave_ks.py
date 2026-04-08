from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
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

    def load_actors(self):
        # Set microwave door to fully open
        open_angle = self.microwave_joint_lower + 0.95 * self.microwave_joint_range
        limits = self.microwave.get_qlimits()
        ndof = len(limits)
        qpos = [0.0] * ndof
        qpos[0] = open_angle
        self.microwave.set_qpos(qpos)

        # Microwave cavity center
        mw_pos = self.microwave.get_pose().p
        self.microwave_interior = [mw_pos[0], mw_pos[1], mw_pos[2] + 0.05]
        self.mw_half_x = 0.12
        self.mw_half_y = 0.10
        self.mw_half_z = 0.08

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

        self.target_pose = self.microwave_interior + [1, 0, 0, 0]

    def play_once(self):
        arm_tag = ArmTag("right")

        self.move(self.grasp_actor(self.hamburg, arm_tag=arm_tag, pre_grasp_dis=0.08))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        self.attach_object(
            self.hamburg,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/006_hamburg/collision/base{self.hamburg_id}.glb",
            str(arm_tag),
        )

        self.move(
            self.place_actor(
                self.hamburg,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                constrain="align",
                pre_dis=0.08,
                dis=0.005,
            ))

        self.detach_object(arms_tag=str(arm_tag))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=-0.08))

        self.info["info"] = {
            "{A}": f"006_hamburg/base{self.hamburg_id}",
            "{B}": f"044_microwave/base{self.microwave_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        hamburg_pos = self.hamburg.get_pose().p
        mw_center = np.array(self.microwave_interior)
        return (
            abs(hamburg_pos[0] - mw_center[0]) < self.mw_half_x
            and abs(hamburg_pos[1] - mw_center[1]) < self.mw_half_y
            and abs(hamburg_pos[2] - mw_center[2]) < self.mw_half_z
        )
