# from envs._base_task import Base_Task
from bench_envs.office._office_base_task import Office_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class milktea_to_laptop(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.milktea.get_name()}

    def load_actors(self):
        laptop_id = np.random.choice([9748,9912,9960,9968,9992,9996,10040,10098,10101,10125,10211])
        self.laptop: ArticulationActor = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="015_laptop",
            modelid=laptop_id,
            xlim=[-self.office_info["table_area"][0]/2+0.1, 0.1],
            ylim=[-self.office_info["table_area"][1]/2, self.office_info["table_area"][1]/2-0.3],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 3],
            qpos=[0.7, 0, 0, 0.7],
            fix_root_link=True,
        )
        self.laptop.set_mass(0.1)
        limit = self.laptop.get_qlimits()[0]
        self.laptop.set_qpos([limit[0] + (limit[1] - limit[0]) * 0.9])
        self.add_prohibit_area(self.laptop, padding=0.01)
        
        milktea_id = np.random.choice(self.item_info[self.sample_d]["office"]["targets"]["101_milk-tea"])
        success, self.milktea = rand_create_cluttered_actor(
            scene=self.scene,
            xlim=[-self.office_info["table_area"][0]/2, self.office_info["table_area"][0]/2],
            ylim=[-self.office_info["table_area"][1]/2, self.office_info["table_area"][1]/2-0.2],
            zlim=[self.office_info["table_height"]],
            modelname="101_milk-tea",
            modelid=milktea_id,
            modeltype="glb",
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
            qpos=[0.66, 0.66, -0.25, -0.25],
            obj_radius=0.03,
            z_offset=0,
            z_max=0.1,
            prohibited_area=self.prohibited_area["table"],
            constrained=False,
            is_static=False,
            size_dict=dict(),
            scale = self.item_info["scales"]["101_milk-tea"]
        )
        if not success:
            raise RuntimeError("Failed to load laptop")

    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag("right" if self.mouse.get_pose().p[0] > 0 else "left")

        # Grasp the mouse with the selected arm
        self.move(self.grasp_actor(self.mouse, arm_tag=arm_tag, pre_grasp_dis=0.1))

        # Lift the mouse upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))

        self.attach_object(self.mouse, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/047_mouse/collision/base{self.mouse_id}.glb", str(arm_tag))

        # Place the mouse at the target location with alignment constraint
        self.move(
            self.place_actor(
                self.mouse,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                constrain="align",
                pre_dis=0.07,
                dis=0.005,
            ))

        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"047_mouse/base{self.mouse_id}",
            "{B}": f"{self.color_name}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        mouse_pose = self.mouse.get_pose().p
        mouse_qpose = np.abs(self.mouse.get_pose().q)
        target_pos = self.target.get_pose().p
        eps1 = 0.04
        eps2 = 0.04

        return (np.all(abs(mouse_pose[:2] - target_pos[:2]) < np.array([eps1, eps2]))
                and (np.abs(mouse_qpose[2] * mouse_qpose[3] - 0.49) < eps1
                     or np.abs(mouse_qpose[0] * mouse_qpose[1] - 0.49) < eps1) and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
