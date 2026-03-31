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


class lift_book(Office_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
    
    def _get_target_object_names(self) -> set[str]:
        return set()

    def load_actors(self):
        level = np.random.choice([0,1])
        model_id = np.random.choice(self.item_info[self.sample_d]["office"]["targets"]["043_book"])
        ylim = [self.shelf.get_pose().p[1]-0.07]
        zlim = [self.office_info["shelf_heights"][level]+0.1]
        xlims = [[self.office_info["shelf_lims"][0]+0.04, 0], [0, self.office_info["shelf_lims"][2]-0.04], [0, self.office_info["shelf_lims"][2]-0.04]]
        self.target_obj = rand_create_actor(
            self,
            xlim=xlims[self.arr_v],
            ylim=ylim,
            zlim=zlim,
            modelname="043_book",
            rotate_rand=False,
            qpos=euler2quat(np.pi, 0, np.pi/2, axes='sxyz'),
            convex=True,
            model_id=model_id,
            is_static=False,
            scale = self.item_info['scales']['043_book'][f'{model_id}']
        )
        self.target_obj.set_mass(0.1)
        self.stabilize_object(self.target_obj)
        center_x = self.target_obj.get_pose().p[0] + 0.02
        center_y = self.target_obj.get_pose().p[1]
        self.prohibited_area[f"shelf{level}"].append([center_x-0.025, center_y-0.06, center_x+0.025, center_y+0.06])
        self.target_obj_pose = self.target_obj.get_pose().p.tolist()


    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag("left") if self.target_obj.get_pose().p[0] < 0 else ArmTag("right")

        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.04, grasp_dis=0.02))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.2))

        # Record information about the objects and arm used in the task
        # self.info["info"] = {
        #     "{A}": f"047_mouse/base{self.mouse_id}",
        #     "{B}": f"{self.color_name}",
        #     "{a}": str(arm_tag),
        # }
        # return self.info

    def check_success(self):
        end_pose_actual = self.target_obj.get_pose().p
        end_pose_desired = self.target_obj_pose
        end_pose_desired[1] -= 0.2
        eps1 = 0.04
        eps2 = 0.04
        eps3 = 0.04

        return np.all(abs(end_pose_actual[:3] - end_pose_desired[:3]) < np.array([eps1, eps2, eps3]))
