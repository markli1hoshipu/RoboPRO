# Demo for placing phone on stand on shelf.
# from envs._base_task import Base_Task
from bench_envs._office_base_task import Office_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class place_phone_shelf(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 70, "obb": 1}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        tag = np.random.randint(2)
        tag = 1
        ori_quat = [
            [0.707, 0.707, 0, 0],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, -0.5],
        ]
        if tag == 0:
            phone_x_lim = [-0.25, -0.05]
            stand_x_lim = [-0.15, 0.0]
        else:
            phone_x_lim = [0.05, 0.25]
            stand_x_lim = [0, 0.15]

        self.phone_id = np.random.choice([0, 1, 2, 4], 1)[0]
        phone_pose = rand_pose(
            xlim=phone_x_lim,
            ylim=[-0.2, 0.0],
            qpos=ori_quat[self.phone_id],
            rotate_rand=True,
            rotate_lim=[0, 0.7, 0],
        )
        self.phone = create_actor(
            scene=self,
            pose=phone_pose,
            modelname="077_phone",
            convex=True,
            model_id=self.phone_id,
        )
        self.phone.set_mass(0.01)

        stand_pose = rand_pose(
                xlim=[0.88,0.94],
                ylim=[-0.55,-0.4],
                zlim=[0.95],
                qpos=[0.5, 0.5, -0.5, -0.5],
                rotate_rand=False,
            )
        # while np.sqrt(np.sum((phone_pose.p[:2] - stand_pose.p[:2])**2)) < 0.15:
        #     stand_pose = rand_pose(
        #         xlim=stand_x_lim,
        #         ylim=[0, 0.2],
        #         qpos=[0.707, 0.707, 0, 0],
        #         rotate_rand=False,
        #     )

        self.stand_id = np.random.choice([1, 2], 1)[0]
        # stand_pose.p = [0, 5, 0.95]
        self.stand = create_actor(
            scene=self,
            pose=stand_pose,
            modelname="078_phonestand",
            convex=True,
            model_id=self.stand_id,
            is_static=False,
        )
        self.stand.set_mass(0.1)
        # self.collision_list.append((self.stand, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/078_phonestand/collision/base0.glb", [0.065, 0.065, 0.065]))
        self.add_prohibit_area(self.phone, padding=0.15)
        self.add_prohibit_area(self.stand, padding=0.15)

        # self.shampoo = create_actor(
        #     scene=self,
        #     pose=sapien.Pose(p=[0.5, -0.35, 0.4], q=[0.5, 0.5, 0.5, 0.5]),
        #     modelname="049_shampoo",
        #     convex=True,
        #     model_id=1,
        #     is_static=True,
        # )
        # self.collision_list.append((self.shampoo, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/049_shampoo/collision/base1.glb", [0.1, 0.5, 0.1]))

    def play_once(self):
        # Determine which arm to use based on phone's position (left if phone is on left side, else right)
        arm_tag = ArmTag("left" if self.phone.get_pose().p[0] < 0 else "right")

        # Grasp the phone with specified arm
        self.move(self.grasp_actor(self.phone, arm_tag=arm_tag, pre_grasp_dis=0.08))

        # Get stand's functional point as target for placement
        stand_func_pose = self.stand.get_functional_point(0)
        
        # self.move(self.back_to_origin(arm_tag=arm_tag))
        
        # Place the phone onto the stand's functional point with alignment constraint
        self.move(
            self.place_actor(
                self.phone,
                arm_tag=arm_tag,
                target_pose=stand_func_pose,
                functional_point_id=0,
                dis=0,
                constrain="align",
            ))
        # self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.4))
        # self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.4))


        self.info["info"] = {
            "{A}": f"077_phone/base{self.phone_id}",
            "{B}": f"078_phonestand/base{self.stand_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        phone_func_pose = np.array(self.phone.get_functional_point(0))
        stand_func_pose = np.array(self.stand.get_functional_point(0))
        eps = np.array([0.045, 0.04, 0.04])
        return (np.all(np.abs(phone_func_pose - stand_func_pose)[:3] < eps) and self.is_left_gripper_open()
                and self.is_right_gripper_open())
