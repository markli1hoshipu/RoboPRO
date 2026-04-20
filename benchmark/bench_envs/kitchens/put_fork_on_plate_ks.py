from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class put_fork_on_plate_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        # Narrow yaw/xlim so the single handle contact stays IK-feasible.
        rand_pos = self.rand_pose_on_counter(
            xlim=[-0.32, 0.32],
            ylim=[-0.20, 0.00],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0],
            obj_padding=0.04,
        )
        while abs(rand_pos.p[0]) < 0.3:
            rand_pos = self.rand_pose_on_counter(
                xlim=[-0.32, 0.32],
                ylim=[-0.20, 0.00],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 4, 0],
                obj_padding=0.04,
            )

        # Restrict to fork model 0: other models have collision hulls that
        # intersect the counter cuboid when attached, triggering
        # INVALID_START_STATE_WORLD_COLLISION right after the lift.
        self.fork_id = 0
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="033_fork",
            convex=True,
            model_id=self.fork_id,
            scale=self.item_info['scales']['033_fork'].get(f'{self.fork_id}', None),
        )
        self.target_obj.set_mass(0.05)

        self.plate_id = np.random.choice(self.item_info[self.sample_d]["kitchens"]["targets"]["003_plate"])
        target_rand_pose = self.rand_pose_on_counter(
            xlim=[0],
            ylim=[-0.23, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            obj_padding=0.08,
        )

        self.des_obj = create_actor(
            scene=self,
            pose=target_rand_pose,
            modelname="003_plate",
            convex=True,
            model_id=self.plate_id,
            scale=self.item_info['scales']['003_plate'].get(f'{self.plate_id}', None),
            is_static=True,
        )
        self.add_prohibit_area(self.des_obj, padding=0.01, area="table")
        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")

        self.des_obj_pose = self.des_obj.get_pose().p.tolist() + [0, 0, 0, 1]
        self.des_obj_pose[2] += 0.02

    def play_once(self):
        arm_tag = ArmTag("right" if self.target_obj.get_pose().p[0] > 0 else "left")

        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.1)

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.2))

        self.attach_object(self.target_obj, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/033_fork/collision/base{self.fork_id}.glb", str(arm_tag))
        self.enable_table(enable=False)

        # Use move_by_displacement + open_gripper instead of place_actor.
        # place_actor with this elongated fork geometry repeatedly fails
        # trajectory optimisation; a simpler carry-and-drop succeeds.
        plate_p = self.des_obj.get_pose().p
        spawn_p = self.target_obj.get_pose().p
        self.move(self.move_by_displacement(
            arm_tag=arm_tag,
            x=float(plate_p[0]) - float(spawn_p[0]),
            y=float(plate_p[1]) - float(spawn_p[1]),
            z=0.0,
        ))
        # Descend close to the plate so the fork lands within the
        # 0.02m xy tolerance instead of bouncing off from lift height.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.15))
        self.move(self.open_gripper(arm_tag, pos=1.0))

    def check_success(self):
        end_pose_actual = self.target_obj.get_pose().p
        end_pose_desired = self.des_obj.get_pose().p
        eps1 = 0.02
        eps2 = 0.02

        return (np.all(abs(end_pose_actual[:2] - end_pose_desired[:2]) < np.array([eps1, eps2]))
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
