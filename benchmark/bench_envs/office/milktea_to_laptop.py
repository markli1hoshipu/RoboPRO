# from envs._base_task import Base_Task
from bench_envs.office._office_base_task import Office_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob
from transforms3d.euler import euler2quat

class milktea_to_laptop(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.milktea.get_name()}

    def load_actors(self):
        # add table as collision
        self.cuboid_collision_list.append(("table", [2, 2, 0.002], [0,0,0.74,1,0,0,0]))
        self.side = np.random.choice(["left", "right"])
        laptop_id = np.random.choice([9748,9912,9960,9968,9992,9996,10040,10098,10101,10125,10211])
        laptop_id = 9912
        self.milktea_id = np.random.choice(self.item_info[self.sample_d]["office"]["targets"]["101_milk-tea"])

        if self.side == "left":
            xlim1 = [self.office_info["table_lims"][0]+0.08, 0]
            xlim2 = [self.office_info["table_lims"][0]+self.target_objects_info["101_milk-tea"]["params"][f"{self.milktea_id}"]["radius"], 0.1]
        else:
            xlim1 = [-0.3, self.office_info["table_lims"][2]-0.27]
            xlim2 = [-0.1, self.office_info["table_lims"][2]-self.target_objects_info["101_milk-tea"]["params"][f"{self.milktea_id}"]["radius"]]
        ylim1 = [self.office_info["table_lims"][1]+0.2, self.office_info["shelf_lims"][1]-0.1]
        ylim2 = [self.office_info["table_lims"][1] + 0.2, self.office_info["shelf_lims"][1]-0.05]
        

        self.laptop: ArticulationActor = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="015_laptop",
            modelid=laptop_id,
            xlim=xlim1,
            ylim=ylim1,
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 6],
            qpos=[0.7, 0, 0, 0.7],
            fix_root_link=True,
        )
        self.laptop.set_mass(0.1)
        limit = self.laptop.get_qlimits()[0]
        self.laptop.set_qpos([limit[0] + (limit[1] - limit[0]) * 0.9])
        self.add_prohibit_area(self.laptop, padding=0.01)
        cuboid_pose = self.laptop.get_pose().p.tolist() + [1, 0, 0, 0]
        cuboid_pose[1] += 0.04
        cuboid_pose[2] = self.office_info["table_height"] + 0.07
        self.cuboid_collision_list.append(("015_laptop", [0.2, 0.07, 0.14], cuboid_pose))
        # self.laptop_collision_box = create_box(
        #     scene=self,
        #     pose=sapien.Pose(p=cuboid_pose[:3], q=cuboid_pose[3:]),
        #     half_size=[0.1, 0.025, 0.07],
        #     color=(1, 0, 0),
        #     name="laptop_collision_box",
        #     is_static=True,
        # )


        half_size = [0.03, 0.03, 0.0005]
        p = self.laptop.get_pose().p.tolist()
        p[0] += 0.15
        p[1] -= 0.05
        p[2] = self.office_info["table_height"] - 0.001
        target_pose = sapien.Pose(p=p, q=[1, 0, 0, 0])
        self.target = create_box(
            scene=self,
            pose=target_pose,
            half_size=half_size,
            color=(0, 0, 1),
            name="box",
            is_static=True,
        )
        self.add_prohibit_area(self.target, padding=0.06, area="table")
        
        success, self.milktea = rand_create_cluttered_actor(
            scene=self.scene,
            xlim=xlim2,
            ylim=ylim2,
            zlim=[self.office_info["table_height"]],
            modelname="101_milk-tea",
            modelid=self.milktea_id,
            modeltype="glb",
            rotate_rand=False,
            rotate_lim=[0, 1, 0],
            qpos=euler2quat(np.pi/2,0, 0, axes='sxyz'),
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
            raise RuntimeError("Failed to load milktea")
        self.milktea.set_mass(0.06)
        self.add_prohibit_area(self.milktea, padding=0.01, area="table")

        self.target_pose = target_pose.p.tolist() + self.milktea.get_pose().q.tolist()
        self.target_pose[2] += 0.02
        # p = self.milktea.get_pose().p.tolist()
        # p[1]-=0.1
        # self.target_pose = p + self.milktea.get_pose().q.tolist()

    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side)

        # Grasp the mouse with the selected arm
        action = self.grasp_actor(self.milktea, arm_tag=arm_tag, pre_grasp_dis=0.1, grasp_dis=0.02, contact_point_id=2)
        action[1][0].target_pose[2] += 0.04
        action[1][1].target_pose[2] += 0.04
        self.move(action)

        # Lift the mouse upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.01))

        self.attach_object(self.milktea, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/101_milk-tea/collision/base{self.milktea_id}.glb", str(arm_tag))

        # Place the mouse at the target location with alignment constraint
        action = self.place_actor(
                self.milktea,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                constrain="free",
                pre_dis=0.0,
                dis=0.0,
                local_up_axis=[0,0,1]
            )
        action[1][0].target_pose[2] += 0.03
        self.move(action)

        # # Record information about the objects and arm used in the task
        # self.info["info"] = {
        #     "{A}": f"047_mouse/base{self.mouse_id}",
        #     "{B}": f"{self.color_name}",
        #     "{a}": str(arm_tag),
        # }
        # return self.info

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
