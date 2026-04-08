from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import math
import numpy as np
from envs._GLOBAL_CONFIGS import *
import glob


class put_plate_in_sink_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
        self._attached = False

    def load_actors(self):
        self.plate_id = 0
        plate_pose = self._safe_rand_pose(
            xlim=[-0.15, 0.15],
            ylim=[-0.20, 0.0],
            zlim=[0.743 + self.table_z_bias],
            qpos=[0.7071, 0.7071, 0, 0],
            rotate_rand=False,
        )
        self.plate = create_actor(
            scene=self,
            pose=plate_pose,
            modelname="003_plate",
            convex=True,
            model_id=self.plate_id,
        )
        self.plate.set_mass(0.1)
        self.add_prohibit_area(self.plate, padding=0.06, area="table")
        self.collision_list.append((
            self.plate,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/003_plate/collision/base{self.plate_id}.glb",
            self.plate.scale,
        ))

    def play_once(self):
        arm_tag = ArmTag("right" if self.plate.get_pose().p[0] > 0 else "left")

        self.move(self._top_down_grasp(self.plate, arm_tag, grasp_z=0.05))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15))
        if not self.plan_success:
            return self.info

        collision_path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/003_plate/collision/base{self.plate_id}.glb"
        self._start_kinematic_attach(self.plate, arm_tag)
        self.attach_object(self.plate, collision_path, str(arm_tag))

        # Move above sink
        sink_p = self.sink_pose.p
        if self.need_plan:
            plan_func = self.robot.left_plan_path if arm_tag == "left" else self.robot.right_plan_path
            quat = [0.707, 0, 0, 0.707]
            placed = False
            for z_off in [0.25, 0.20, 0.15, 0.30]:
                ee_pose = self._compute_container_ee_pose(sink_p, quat, z_off)
                if plan_func(ee_pose)['status'] == 'Success':
                    self._kinematic_move(arm_tag, [Action(arm_tag, "move", target_pose=ee_pose)])
                    if self.plan_success:
                        placed = True
                        break
            if not placed:
                self._stop_kinematic_attach()
                self.plan_success = False
                return self.info
        else:
            self._kinematic_move(arm_tag, [Action(arm_tag, "move", target_pose=[0,0,0,0,0,0,0])])

        self.move((arm_tag, [Action(arm_tag, "open", target_gripper_pos=1.0)]))
        self._stop_kinematic_attach()
        for _ in range(200):
            self.scene.step()
        self.detach_object(arms_tag=str(arm_tag))

        self.info["info"] = {
            "{A}": f"003_plate/base{self.plate_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        plate_pos = self.plate.get_pose().p
        sink_p = self.sink_pose.p
        eps = 0.15
        return (
            abs(plate_pos[0] - sink_p[0]) < eps
            and abs(plate_pos[1] - sink_p[1]) < eps
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
