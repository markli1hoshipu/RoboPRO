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


class grab_battery(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 200, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return set()

    def load_actors(self):
        self.side =  "right"
        limit = self.cabinet.get_qlimits()[0]
        self.cabinet.set_qpos([limit[1],0,0])
        self.collision_list.append({
            "actor": self.cabinet,
            "collision_path": f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/036_cabinet/46653/textured_objs/",
            "link": "link_1",
            "files": ["original-23.obj", "original-24.obj"],
        })

        self.stapler_id = np.random.choice([0, 1, 2, 3, 4, 5, 6])
        self.stapler = create_actor(
            scene=self,
            pose=sapien.Pose(p=[0,0,0.743], q=euler2quat(np.pi/2,0, np.pi, axes='sxyz')),
            modelname="048_stapler",
            convex=True,
            model_id=self.stapler_id,
            is_static=False,
        )

    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side)

        self.move(self.grasp_actor(self.stapler, arm_tag=arm_tag, pre_grasp_dis=0.05, contact_point_id=[0,1]))
        # self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))
        target_pose = self.cabinet.get_functional_point(0)
        # target_pose[3:] = euler2quat(np.pi/2,0, np.pi, axes='sxyz')
        print(target_pose)
        self.move(self.place_actor(
            self.stapler,
            arm_tag=arm_tag,
            target_pose=target_pose,
            pre_dis=0.05,
            dis=0.05,
            constrain="free",
            # local_up_axis=[0,1,0],
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
