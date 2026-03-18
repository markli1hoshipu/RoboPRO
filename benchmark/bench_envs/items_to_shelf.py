# from envs._base_task import Base_Task
from bench_envs._office_base_task import Office_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class items_to_shelf(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.tea_box.get_name(), self.cube.get_name()}

    def load_actors(self):
        rand_pos = rand_pose(
            xlim=[0.1,0.4],
            ylim=[-0.15,0],
            qpos=[0, 0, 0.7071, 0.7071],
            rotate_rand=False,
            rotate_lim=[0, 3.14, 0],
        )
        scale = [0.18,0.14,0.17]
        self.wooden_box = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="042_wooden_box",
            convex=True,
            model_id=0,
            scale=scale,
            is_static=False,
        )
        self.wooden_box.set_mass(1)
        self.add_prohibit_area(self.wooden_box, padding=0.02, area="table")
        self.collision_list.append((self.wooden_box, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/042_wooden_box/collision/base0.glb", scale))

        # ------------------------------------------------------------
        self.cube_id = 0
        center = self.wooden_box.get_pose().p
        self.cube = rand_create_actor(
            self,
            xlim=[center[0]-0.09,center[0]-0.02],
            ylim=[center[1]-0.03,center[1]+0.03],
            zlim=[center[2]+0.05],
            modelname="073_rubikscube",
            rotate_rand=True,
            rotate_lim=[0, 0.5, 0],
            qpos=[0, 0, 0.7071, 0.7071],
            convex=True,
            model_id=self.cube_id,
            is_static=False,
        )
        self.cube.set_mass(0.1)

        # ------------------------------------------------------------
        self.tea_box_id = 0
        self.tea_box = rand_create_actor(
            self,
            xlim=[center[0]+0.09,center[0]+0.02],
            ylim=[center[1]-0.03,center[1]+0.03],
            zlim=[center[2]+0.095],
            modelname="112_tea-box",
            rotate_rand=True,
            rotate_lim=[0.5, 0, 0],
            qpos=[-0.7071, 0, 0.7071, 0],
            convex=True,
            model_id=self.tea_box_id,
            is_static=False,
        )
        self.tea_box.set_mass(0.1)

        # placement targets --------------------------------------------------
        # target 1, tea box placement target
        target_rand_pose = rand_pose(
            xlim=[0.785],
            ylim=[self.shelf.get_pose().p[1]-0.16,self.shelf.get_pose().p[1]+0.16],
            zlim = [self.shelf_heights[1]-0.015],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        half_size = [0.05, 0.05, 0.0005]
        self.target1 = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=(0, 0, 1), # blue
            name="target1",
            is_static=True,
        )
        self.target1_pose = self.target1.get_pose().p.tolist() + [0, 0, 0, 1]
        self.target1_pose[2] += 0.04 # raise target 0.02 meters
        self.add_prohibit_area(self.target1, padding=0.05, area="shelf1")

        # target 2, rubics cube placement target
        target_rand_pose = rand_pose(
            xlim=[0.785],
            ylim=[self.shelf.get_pose().p[1]-0.24,self.shelf.get_pose().p[1]+0.24],
            zlim = [self.shelf_heights[0]-0.02],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        half_size = [0.05, 0.05, 0.0005]
        self.target2 = create_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=(0, 0, 1), # blue
            name="target2",
            is_static=True,
        )
        self.target2_pose = self.target2.get_pose().p.tolist() + [-0.5, 0.5, 0.5, -0.5]
        self.target2_pose[2] += 0.09 # raise target 0.02 meters
        self.add_prohibit_area(self.target2, padding=0.05, area=f"shelf0")

    def play_once(self):
        arm_tag = ArmTag("right")

        # tea box --------------------------------------------------
        action = self.grasp_actor(self.tea_box, arm_tag=arm_tag, pre_grasp_dis=0.1, contact_point_id=5)
        if action:
            action[1][1].target_pose[2] += 0.04 # grasp center of box
        self.move(action)

        # Lift the box upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.06))

        self.attach_object(self.tea_box, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/112_tea-box/collision/base{self.tea_box_id}.glb", str(arm_tag))

        self.move(
            self.place_actor(
                self.tea_box,
                arm_tag=arm_tag,
                target_pose=self.target1_pose,
                constrain="align",
                pre_dis=0.08,
                dis=0.005,
            ))
        self.detach_object(arms_tag=str(arm_tag))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=-0.08))

        # rubics cube --------------------------------------------------
        action = self.grasp_actor(self.cube, arm_tag=arm_tag, pre_grasp_dis=0.1, contact_point_id=3)
        if action:
            action[1][1].target_pose[2] += 0.04 # grasp center of box
        self.move(action)

        # Lift the box upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.06))

        self.attach_object(self.cube, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/073_rubikscube/collision/base{self.cube_id}.glb", str(arm_tag))

        self.move(
            self.place_actor(
                self.cube,
                arm_tag=arm_tag,
                target_pose=self.target2_pose,
                constrain="align",
                pre_dis=0.08,
                dis=-0.02,
            ))

        self.detach_object(arms_tag=str(arm_tag))

        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"112_tea-box/base{self.tea_box_id}",
            "{B}": f"073_rubikscube/base{self.cube_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        tea_box_pose = self.tea_box.get_pose().p
        rubics_cube_pose = self.cube.get_pose().p
        return tea_box_pose[2]+0.03 > self.shelf_heights[1] and rubics_cube_pose[2] > self.shelf_heights[0]
