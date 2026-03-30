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


class pick_mouse(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return set()

    def load_actors(self):
        self.side =  "left" if self.arr_v == 2 else "right"

        # set up cabinet
        limit = self.cabinet.get_qlimits()[0]
        self.cabinet.set_qpos([limit[1],0,0])
        self.collision_list.append({
            "actor": self.cabinet,
            "collision_path": f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/036_cabinet/46653/textured_objs/",
            "link": "link_1",
            "files": ["original-23.obj", "original-24.obj", "original-18.obj"],
        })
        self.collision_list.append({
            "actor": self.cabinet,
            "collision_path": f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/036_cabinet/46653/textured_objs/",
            "link": "link_2",
            "files": ["original-34.obj", "original-41.obj"],
        })
        cabinet_pose = self.cabinet.get_pose().p
        cabinet_pose[1]-= 0.19  
        self.prohibited_area["table"].append([cabinet_pose[0]-0.11, cabinet_pose[1]-0.1, cabinet_pose[0]+0.11, cabinet_pose[1]+0.1])

        # set up stapler --------------------------------------------------
        self.mouse_id = np.random.choice(self.item_info[self.sample_d]["office"]["targets"]["047_mouse"])
        pose = self.cabinet.get_pose().p
        pose[1]-= 0.2
        pose[2] = self.office_info["table_height"]+0.03
        self.mouse = rand_create_actor(
            scene=self,
            xlim = [pose[0]],
            ylim = [pose[1]],
            zlim = [pose[2]],
            rotate_rand = True,
            qpos = euler2quat(np.pi/2, 0, np.pi, axes='sxyz'),
            rotate_lim = [0, np.pi/8, 0],
            modelname="047_mouse",
            scale=self.item_info['scales']['047_mouse'].get(f'{self.mouse_id}',None),
            convex=True,
            model_id=self.mouse_id,
            is_static=False,
        )

    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side)
        self.

        self.move(self.grasp_actor(self.mouse, arm_tag=arm_tag, pre_grasp_dis=0.05,grasp_dis=0.02))
        self.attach_object(self.mouse, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/047_mouse/collision/base{self.mouse_id}.glb", str(arm_tag))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, y=-0.25))

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
