# Demo for transferring object on shelf to drawer. Right now the drawer motion planning is not able to find a solution, or some other issue.
# from envs._base_task import Base_Task
from bench_envs._office_base_task import Office_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class transfer_object(Office_base_task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags, table_static=False)

    def load_actors(self):
        rand_pos = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.2, -0.1],
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 3, 0],
        )
        while abs(rand_pos.p[0]) < 0.2:
            rand_pos = rand_pose(
                xlim=[-0.32, 0.32],
                ylim=[-0.2, -0.1],
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 3, 0],
            )

        # rand_pos = sapien.Pose(p=[1, -0.75, 0.95], q=[0.5, 0.5, 0.5, 0.5])
        rand_pos = sapien.Pose(p=[0.2, 0, 0.8], q=[0.5, 0.5, 0.5, 0.5])
        
        def get_available_model_ids(modelname):
            asset_path = os.path.join("assets/objects", modelname)
            json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))
            available_ids = []
            for file in json_files:
                base = os.path.basename(file)
                try:
                    idx = int(base.replace("model_data", "").replace(".json", ""))
                    available_ids.append(idx)
                except ValueError:
                    continue
            return available_ids

        object_list = [
            # "047_mouse",
            # "048_stapler",
            # "057_toycar",
            "073_rubikscube",
            # "075_bread",
            # "077_phone",
            # "081_playingcards",
            # "112_tea-box",
            # "113_coffee-box",
            # "107_soap",
            # "059_pencup",
        ]
        self.selected_modelname = np.random.choice(object_list)
        available_model_ids = get_available_model_ids(self.selected_modelname)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname}")
        self.selected_model_id = np.random.choice(available_model_ids)
        self.object = create_actor(
            scene=self,
            pose=rand_pos,
            modelname=self.selected_modelname,
            convex=True,
            model_id=self.selected_model_id,
        )
        self.object.set_mass(0.01)
        self.add_prohibit_area(self.object, padding=0.01)
        self.add_prohibit_area(self.cabinet, padding=0.01)
        self.prohibited_area.append([-0.15, -0.3, 0.15, 0.3])

    def play_once(self):
        arm_tag = ArmTag("right")
        self.arm_tag = arm_tag
        self.origin_z = self.object.get_pose().p[2]
        # breakpoint()
        # Grasp the drawer bar
        self.move(self.grasp_actor(self.cabinet, arm_tag=arm_tag.opposite, pre_grasp_dis=0.05))

        # Pull the drawer
        for _ in range(4):
            self.move(self.move_by_displacement(arm_tag=arm_tag.opposite, y=-0.04))
        # Grasp the object
        self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.15))
        # # Move the object to cabinet
        self.move(self.back_to_origin(arm_tag=arm_tag))
        # target_pose = self.cabinet.get_pose().p + [0, 0, 0.3]
        # self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=target_pose))

        # Place the object into the cabinet
        target_pose = self.cabinet.get_functional_point(0)
        self.move(self.place_actor(
            self.object,
            arm_tag=arm_tag,
            target_pose=target_pose,
            pre_dis=0.13,
            dis=0.1,
        ))

        self.info["info"] = {
            "{A}": f"{self.selected_modelname}/base{self.selected_model_id}",
            "{B}": f"036_cabinet/base{0}",
            "{a}": str(arm_tag),
            "{b}": str(arm_tag.opposite),
        }
        return self.info

    def check_success(self):
        object_pose = self.object.get_pose().p
        target_pose = self.cabinet.get_functional_point(0)
        tag = np.all(abs(object_pose[:2] - target_pose[:2]) < np.array([0.05, 0.05]))
        return ((object_pose[2] - self.origin_z) > 0.007 and (object_pose[2] - self.origin_z) < 0.12 and tag
                and (self.robot.is_left_gripper_open() if self.arm_tag == "left" else self.robot.is_right_gripper_open()))
