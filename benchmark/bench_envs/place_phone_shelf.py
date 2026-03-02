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
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        shelf_level = 0
        ori_quat = [
            [0.707, 0.707, 0, 0],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, -0.5],
        ]

        self.phone_id = np.random.choice([0, 1, 2, 4], 1)[0]
        phone_pose = rand_pose(
            # xlim = [-0.0],
            xlim = [0,0.35],
            # ylim=[0.05],
            ylim=[-0.23, 0.05],
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
                xlim=[self.shelf.get_pose().p[0]-0.1],
                ylim=[-0.6,-0.4],
                zlim=[self.shelf_heights[shelf_level]+0.01],
                qpos=[0.5, 0.5, -0.5, -0.5],
                rotate_rand=False,
            )

        self.stand_id = np.random.choice([1, 2], 1)[0]
        self.stand = create_actor(
            scene=self,
            pose=stand_pose,
            modelname="078_phonestand",
            convex=True,
            model_id=self.stand_id,
            is_static=False,
        )
        self.stand.set_mass(1)
        self.add_prohibit_area(self.phone, padding=0.08, area="table")
        self.add_prohibit_area(self.stand, padding=0.06, area=f"shelf{shelf_level}")

        #  ---------------------------------------------------------------------
        self.id_list = [i for i in range(20)]
        self.bottle_id = np.random.choice(self.id_list)
        self.bottle = rand_create_actor(
            self,
            xlim=[self.phone.get_pose().p[0]+0.15],
            ylim=[self.phone.get_pose().p[1]-0.01],
            modelname="001_bottle",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            qpos=[0.66, 0.66, -0.25, -0.25],
            convex=True,
            model_id=self.bottle_id,
            scale = [0.14, 0.14, 0.14],
        )
        
        self.bottle.set_mass(1)
        rb = self.bottle.actor.components[1]
        rb.set_linear_damping(5.0)
        rb.set_angular_damping(20.0)
        self.add_prohibit_area(self.bottle, padding=0.04, area="table")
        self.collision_list.append((self.bottle, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/001_bottle/collision/base{self.bottle_id}.glb", [0.14, 0.14, 0.14]))

    def play_once(self):
        # Determine which arm to use based on phone's position (left if phone is on left side, else right)
        arm_tag = ArmTag("right")

        # Grasp the phone with specified arm
        self.move(self.grasp_actor(self.phone, arm_tag=arm_tag, pre_grasp_dis=0.08))

        # Get stand's functional point as target for placement
        stand_func_pose = self.stand.get_functional_point(0)
        
        self.attach_object(self.phone, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/077_phone/collision/base{self.phone_id}.glb", str(arm_tag))

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
