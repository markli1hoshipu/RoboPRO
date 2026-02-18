# from envs._base_task import Base_Task
from bench_envs._office_base_task import Office_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class pencup_on_pad(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.shelf_level = np.random.randint(0, 2)
        zlim = self.shelf_heights[self.shelf_level]
        rand_pos = rand_pose(
                xlim=[0.74,0.8],
                ylim=[-0.55,-0.4],
                zlim=[zlim],
                qpos=[0.5, 0.5, -0.5, -0.5],
                rotate_rand=False,
            )

        self.pencup_id = np.random.choice([0, 1, 2, 3, 5], 1)[0]
        # [3]
        # print(f"pencup_id: {self.pencup_id}")
        # 1,6 might be too small

        self.coaster = create_actor(
            scene=self,
            pose=sapien.Pose(p=rand_pos.p, q=[0.5, 0.5, 0.5, 0.5]),
            modelname="019_coaster",
            convex=True,
            model_id=0,
            is_static=False,
        )
        # self.collision_list.append((self.coaster, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/019_coaster/collision/base0.glb", self.coaster.scale))
        
        p = rand_pos.p
        p[2] += 0.025
        rand_pos.set_p(p) # cup above the coaster
        self.pencup = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="059_pencup",
            convex=True,
            model_id=self.pencup_id,
            is_static=False,
        )
        self.pencup.set_mass(0.1)
        self.add_prohibit_area(self.pencup, padding=0.08)

        target_rand_pose = rand_pose(
            xlim=[0.05,0.38],
            # xlim=[0.4],
            # ylim=[0.05],
            ylim=[-0.05,0.2],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        colors = {
            "Red": (1, 0, 0),
            "Green": (0, 1, 0),
            "Blue": (0, 0, 1),
            "Yellow": (1, 1, 0),
            "Cyan": (0, 1, 1),
            "Magenta": (1, 0, 1),
            "Black": (0, 0, 0),
            "Gray": (0.5, 0.5, 0.5),
        }

        color_items = list(colors.items())
        color_index = np.random.choice(len(color_items))
        self.color_name, self.color_value = color_items[color_index]

        half_size = [0.04, 0.04, 0.0005]
        self.target = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=self.color_value,
            name="box",
            is_static=True,
        )
        self.add_prohibit_area(self.target, padding=0.08)
        # Construct target pose with position from target object and identity orientation
        self.target_pose = self.target.get_pose().p.tolist() + [1,0,0,0]   # wxyz
        self.target_pose[2]+=0.05

        # ------------------------------------------------------------
        self.id_list = [i for i in range(20)]
        self.bottle_id = np.random.choice(self.id_list)
        self.bottle = rand_create_actor(
            self,
            xlim=[self.target_pose[0]+0.18],
            ylim=[self.target_pose[1]-0.1],
            modelname="001_bottle",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            qpos=[0.66, 0.66, -0.25, -0.25],
            convex=True,
            model_id=self.bottle_id,
            scale = [0.14, 0.14, 0.14],
        )
        
        self.bottle.set_mass(0.3)
        self.add_prohibit_area(self.bottle, padding=0.04)
        self.collision_list.append((self.bottle, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/001_bottle/collision/base{self.bottle_id}.glb", self.bottle.scale))

    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag("right")

        # Grasp the mouse with the selected arm
        self.move(self.grasp_actor(self.pencup, arm_tag=arm_tag, pre_grasp_dis=0.08, contact_point_id=[0,1,2,5,6,7]))

        # Lift the mouse upward by 0.1 meters in z-direction
        # self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))

        # Place the mouse at the target location with alignment constraint
        self.move(
            self.place_actor(
                self.pencup,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                constrain="align",
                pre_dis=0.01,
                dis=0.005,
            ))

        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"059_pencup/base{self.pencup_id}",
            "{B}": f"{self.color_name}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        mouse_pose = self.pencup.get_pose().p
        mouse_qpose = np.abs(self.pencup.get_pose().q)
        target_pos = self.target.get_pose().p
        eps1 = 0.015
        eps2 = 0.012

        return (np.all(abs(mouse_pose[:2] - target_pos[:2]) < np.array([eps1, eps2]))
                and (np.abs(mouse_qpose[2] * mouse_qpose[3] - 0.49) < eps1
                     or np.abs(mouse_qpose[0] * mouse_qpose[1] - 0.49) < eps1) and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
