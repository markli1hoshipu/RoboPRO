# from envs._base_task import Base_Task
from bench_envs.office._office_base_task import Office_base_task
from envs.utils import *
from bench_envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob
from transforms3d.euler import euler2quat


class cleaning_2(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return set()

    def load_actors(self):
        # target3 ------------------------------------------------------------
        target_rand_pose = rand_pose(
            xlim=[self.office_info["shelf_lims"][0]+0.04, self.office_info["shelf_lims"][2]-0.04],
            ylim=[self.office_info["shelf_lims"][1] + 0.03],
            zlim = [self.office_info["shelf_heights"][0]],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        half_size = [0.025, 0.025, 0.0005]
        self.target3_box = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=(0, 0, 1), # blue
            name="target",
            is_static=True,
        )
        self.target3 = self.target3_box.get_pose().p.tolist()
        self.target3[2] += 0.02 # raise target 0.02 meters
        self.add_prohibit_area(self.target3_box, padding=0.05, area=f"shelf0")

        # milktea ------------------------------------------------------------
        self.milktea_id = np.random.choice(self.item_info[self.sample_d]["office"]["targets"]["101_milk-tea"])
        if self.target3_box.get_pose().p[0] < 0:
            xlim1 = [self.office_info["table_lims"][0]+self.target_objects_info["101_milk-tea"]["params"][f"{self.milktea_id}"]["radius"], 0.1]
        else:
            xlim1 = [-0.1, self.office_info["table_lims"][2]-self.target_objects_info["101_milk-tea"]["params"][f"{self.milktea_id}"]["radius"]]
        ylim1 = [self.office_info["table_lims"][1] + 0.3, self.office_info["shelf_lims"][1]-0.05]

        self.milktea = rand_create_actor(
            self,
            xlim=xlim1,
            ylim=ylim1,
            modelname="101_milk-tea",
            rotate_rand=False,
            rotate_lim=[0, 1, 0],
            qpos=euler2quat(np.pi/2,0, 0, axes='sxyz'),
            convex=True,
            model_id=self.milktea_id, 
        )
        self.milktea.set_mass(0.06)
        self.add_prohibit_area(self.milktea, padding=0.01)
        self.target3 += self.milktea.get_pose().q.tolist()

        # mouse ------------------------------------------------------------
        if self.target3_box.get_pose().p[0] < 0:
            xlim = [0.3, 0.45]
        else:
            xlim = [-0.45, -0.3]
        rand_pos = rand_pose(
            xlim=xlim,
            ylim=[-0.23, self.office_info["shelf_lims"][1]-0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
        )
        self.mouse_id = np.random.choice(self.item_info[self.sample_d]["office"]["targets"]["047_mouse"])
        self.mouse = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="047_mouse",
            convex=True,
            model_id=self.mouse_id,
            scale=self.item_info['scales']['047_mouse'].get(f'{self.mouse_id}',None),
        )
        self.mouse.set_mass(0.05)
        self.add_prohibit_area(self.mouse, padding=0.03, area="table")

        # target1 ------------------------------------------------------------
        target_rand_pose = rand_pose(
            xlim=[0],
            ylim=[-0.23, self.office_info["shelf_lims"][1]-0.05],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        colors = {
            # "Red": (1, 0, 0),
            # "Green": (0, 1, 0),
            "Blue": (0, 0, 1),
            # "Yellow": (1, 1, 0),
            "Cyan": (0, 1, 1),
            # "Magenta": (1, 0, 1),
            "Black": (0, 0, 0),
            "Gray": (0.5, 0.5, 0.5),
        }

        color_items = list(colors.items())
        color_index = np.random.choice(len(color_items))
        self.color_name, self.color_value = color_items[color_index]
        half_size = [0.035, 0.065, 0.0005]
        self.target1_box = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=self.color_value,
            name="target1_box",
            is_static=True,
        )
        self.add_prohibit_area(self.target1_box, padding=0.04, area="table")
        self.target1 = self.target1_box.get_pose().p.tolist() + [0, 0, 0, 1]

        # target2 ------------------------------------------------------------
        center_x = (self.mouse.get_pose().p[0] + self.target1_box.get_pose().p[0]) / 2
        center_y = (self.mouse.get_pose().p[1] + self.target1_box.get_pose().p[1]) / 2
        self.stand_id = np.random.choice([1, 2], 1)[0]
        self.stand = rand_create_actor(
            scene=self,
            xlim=[center_x],
            ylim=[center_y],
            zlim=[self.office_info["table_height"]],
            qpos=[0.7071, 0.7071, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 6, 0],
            modelname="078_phonestand",
            convex=True,
            model_id=self.stand_id,
            is_static=False,
        )
        self.stand.set_mass(2)
        self.collision_list.append({
            "actor": self.stand,
            "collision_path": f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/078_phonestand/collision/base{self.stand_id}.glb",
        })
        self.add_prohibit_area(self.stand, padding=0.06, area="table")

        # phone ------------------------------------------------------------
        ori_quat = [
            [0.707, 0.707, 0, 0],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, -0.5],
        ]
        if self.target3_box.get_pose().p[0] < 0:
            xlim = [-0.1,self.office_info["table_lims"][2]-0.05]
        else:
            xlim = [self.office_info["table_lims"][0]+0.05, 0.1]
        self.phone_id = np.random.choice(self.item_info[self.sample_d]["office"]["targets"]["077_phone"])
        success, self.phone = rand_create_cluttered_actor(
            scene=self.scene,
            xlim=xlim,
            ylim=[self.office_info["table_lims"][1]+0.05, self.office_info["shelf_lims"][1]-0.05],
            zlim=[self.office_info["table_height"]],
            modelname="077_phone",
            modelid=self.phone_id,
            modeltype="glb",
            rotate_rand=True,
            qpos=ori_quat[self.phone_id],
            rotate_lim=[0, 0.7, 0],
            obj_radius=0.04,
            z_offset=0,
            z_max=0.02,
            prohibited_area=self.prohibited_area["table"],
            constrained=False,
            is_static=False,
            size_dict=dict(),
        )
        if not success:
            raise RuntimeError("Failed to load phone")
        self.phone.set_mass(0.01)
        self.add_prohibit_area(self.phone, padding=0.01, area="table")


    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        armL = ArmTag("left")
        armR = ArmTag("right")
        arms = [
            armL if self.mouse.get_pose().p[0] < 0 else armR,
            armL if self.stand.get_pose().p[0] < 0 else armR,
            armL if self.target3_box.get_pose().p[0] < 0 else armR,
        ]

        # mouse --------------------------------------------------
        self.move(self.grasp_actor(self.mouse, arm_tag=arms[0], pre_grasp_dis=0.1))

        # Lift the mouse upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arms[0], z=0.1))

        self.attach_object(self.mouse, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/047_mouse/collision/base{self.mouse_id}.glb", str(arms[0]))

        # Place the mouse at the target location with alignment constraint
        self.move(
            self.place_actor(
                self.mouse,
                arm_tag=arms[0],
                target_pose=self.target1,
                constrain="align",
                pre_dis=0.07,
                dis=0.005,
            ))
        self.detach_object(arms_tag=str(arms[0]))

        if arms[0] != arms[1]:
            self.move(self.back_to_origin(arms[0]))

        # phone --------------------------------------------------
        self.move(self.grasp_actor(self.phone, arm_tag=arms[1], pre_grasp_dis=0.08))

        # Get stand's functional point as target for placement
        stand_func_pose = self.stand.get_functional_point(0)
        
        self.attach_object(self.phone, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/077_phone/collision/base{self.phone_id}.glb", str(arms[1]))

        # Place the phone onto the stand's functional point with alignment constraint
        self.move(
            self.place_actor(
                self.phone,
                arm_tag=arms[1],
                target_pose=stand_func_pose,
                functional_point_id=0,
                dis=0,
                constrain="align",
                pre_dis=0.05,
            ))
        self.detach_object(arms_tag=str(arms[1]))
        self.move(self.move_by_displacement(arm_tag=arms[1], y=-0.04))

        if arms[1] != arms[2]:
            self.move(self.back_to_origin(arms[1]))
    
        # milktea --------------------------------------------------
        action = self.grasp_actor(self.milktea, arm_tag=arms[2], pre_grasp_dis=0.1, grasp_dis=0.02, contact_point_id=2)
        action[1][0].target_pose[2] += 0.04
        action[1][1].target_pose[2] += 0.04
        self.move(action)

        # Lift the mouse upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arms[2], z=0.01))

        self.attach_object(self.milktea, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/101_milk-tea/collision/base{self.milktea_id}.glb", str(arms[2]))

        # Place the mouse at the target location with alignment constraint
        action = self.place_actor(
                self.milktea,
                arm_tag=arms[2],
                target_pose=self.target3,
                constrain="free",
                pre_dis=0.0,
                dis=0.0,
                local_up_axis=[0,0,1]
            )
        # action[1][0].target_pose[2] += 0.03
        self.move(action)

        # Record information about the objects and arm used in the task
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
        eps1 = 0.015
        eps2 = 0.012

        return (np.all(abs(mouse_pose[:2] - target_pos[:2]) < np.array([eps1, eps2]))
                and (np.abs(mouse_qpose[2] * mouse_qpose[3] - 0.49) < eps1
                     or np.abs(mouse_qpose[0] * mouse_qpose[1] - 0.49) < eps1) and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
