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


class place_notebook(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 200, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return set()

    def load_actors(self):
        pose = self.file_holder.get_pose().p
        pose[2] = 1.05
        self.book = create_actor(
            scene=self,
            pose=sapien.Pose(p=pose, q=euler2quat(np.pi, 0, np.pi/2, axes='sxyz')),
            modelname="043_book",
            convex=True,
            model_id=0,
            is_static=True,
            scale = 0.5
        )


    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag("left")

        self.move(self.grasp_actor(self.book, arm_tag=arm_tag, pre_grasp_dis=0.04, grasp_dis=0.02))
        # self.attach_object(self.book, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/043_book/collision/base0.glb", str(arm_tag))

        self.target_pose = self.file_holder.get_pose().p.tolist() + euler2quat(np.pi, 0, 0, axes='sxyz')
        self.target_pose[2]+=0.1
        self.target_pose[1]-=0.1

        self.move(
            self.place_actor(
                self.book,
                arm_tag=arm_tag,
                target_pose=self.target_pose,
                constrain="free",
                pre_dis=0.01,
                dis=0.005,
            ))

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
