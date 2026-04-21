from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from bench_envs.utils.scene_gen_utils import get_obj_new_pose
from envs.utils import *
import sapien, math, os, glob
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
from transforms3d.euler import euler2quat


class chain_apple_bin_plate_knife_sink_ks(KitchenS_base_task):
    TOP_DOWN_Q = [-0.5, 0.5, -0.5, -0.5]
    TCP_OFFSET = 0.12
    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.apple.get_name(), self.plate.get_name(), self.knife.get_name()}

    def load_actors(self):
        # 1) Bin (static destination for apple) — spawn on the LEFT side so
        #    apple+bin live on the left and knife can live on the right.
        bin_pose = self.rand_pose_on_counter(
            xlim=[-0.42, -0.35], ylim=[-0.23, 0.0],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, np.pi/4, 0],
            obj_padding=0.12,
        )
        self.bin_id = int(np.random.choice([0, 6]))
        self.bin = create_actor(
            scene=self, pose=bin_pose, modelname="063_tabletrashbin",
            convex=True, model_id=self.bin_id,
            scale=[0.10, 0.10, 0.10], is_static=True,
        )
        self.bin.set_name("bin")
        self.add_prohibit_area(self.bin, padding=0.02, area="table")


        # 3) Plate on counter (→ sink) — keep in center/left band so it doesn't
        #    crowd the knife's right-side band
        plate_pose = self.rand_pose_on_counter(
            xlim=[0.1,0.15], ylim=[-0.15, -0.08],
            qpos=euler2quat(math.pi/2, 0,0), rotate_rand=False, rotate_lim=[0, np.pi, 0],
            obj_padding=0.08,
        )
        self.plate = create_actor(
            scene=self, pose=plate_pose, modelname="003_plate",
            convex=True, model_id=0
        )
        self.plate.set_mass(0.1)
        self.add_prohibit_area(self.plate, padding=0.02, area="table")


        # 2) Apple on counter (must grasp → drop in bin).
        #    Bin is on the left, so apple lives on the left too.
        # apple_pose = self.rand_pose_on_counter(
        #     xlim=[-0.30, -0.05], ylim=[-0.15, 0.05],
        #     qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, np.pi, 0],
        #     obj_padding=0.05,
        # )


        app_p = [self.plate.get_pose().p[0] -.03, # np.random.uniform(-0.03, 0.03),
                 self.plate.get_pose().p[1] + np.random.uniform(-0.02, 0.02),
                 self.plate.get_pose().p[2] + 0.02]
        apple_pose =  sapien.Pose(app_p , [0.5, 0.5, 0.5, 0.5])
        self.apple_id = int(np.random.choice([0, 1]))
        self.apple = create_actor(
            scene=self, pose=apple_pose, modelname="035_apple",
            convex=True, model_id=self.apple_id)
        self.apple.set_mass(0.05)
        self.add_prohibit_area(self.apple, padding=0.02, area="table")
       
       
       
        # get_obj_new_pose(self.apple, ele=0.02)

        # 4) Knife on counter (→ sink) — spawn on the RIGHT side, opposite the bin
        # knife_pose = self.rand_pose_on_counter(
        #     xlim=[0.05, 0.30], ylim=[-0.15, 0.05],
        #     qpos=[0.7071068, 0.0, -0.7071068, 0.0], rotate_rand=False,
        #     obj_padding=0.08,
        # )
        # self.knife = create_actor(
        #     scene=self, pose=knife_pose, modelname="034_knife",
        #     convex=True, model_id=0, scale=0.155,
        # )
        # self.knife.set_mass(0.05)
        # get_obj_new_pose(self.knife)
        
        # self.add_prohibit_area(self.knife, padding=0.02, area="table")

        # Targets
        bin_p = self.bin.get_pose().p
        self.bin_target_pose = [bin_p[0], bin_p[1], bin_p[2] + 0.08, 0, 0, 0, 1]
        sink_p = self.sink.get_pose().p
        self.sink_target_pose = [sink_p[0], sink_p[1], sink_p[2] + 0.05, 0, 0, 0, 1]

    def grasp_plate(self, target_obj, arm_tag="right"):
        # arm_tag = self.arm_tag
        side_sign = 1 if arm_tag == "left" else -1

        # Counter cuboid must be out of the Curobo world so the wrist can
        # dip below rim z when placing plate into the sink basin.
        self.enable_table(enable=False)

        self.move(self.open_gripper(arm_tag, pos=1.0))

        plate_p = target_obj.get_pose().p
        # Rim point nearest the robot-arm base: inner side of plate along x.
        rim_x = float(plate_p[0]) - 0.085 * side_sign
        rim_y = float(plate_p[1])
        rim_z_top = float(plate_p[2]) + 0.012  # rim sits ~1.2 cm above base

        hover_tcp_z = rim_z_top + 0.10
        hover_pose = [rim_x, rim_y, hover_tcp_z + self.TCP_OFFSET] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, hover_pose))

        grasp_tcp_z = rim_z_top - 0.002
        grasp_pose = [rim_x, rim_y, grasp_tcp_z + self.TCP_OFFSET] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, grasp_pose))

        self.move(self.close_gripper(arm_tag, pos=0.0))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.25))

    def play_once(self):
        # # Step 1: apple → bin
        arm = ArmTag("left")
        self.grasp_actor_from_table(self.apple, arm_tag=arm, pre_grasp_dis=0.07)
        self.move(self.move_by_displacement(arm_tag=arm, z=0.15))
        self.attach_object(self.apple,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/035_apple/collision/base{self.apple_id}.glb",
            str(arm))
        self.enable_table(enable=True)
        self.move(self.place_actor(self.apple, arm_tag=arm, target_pose=self.bin_target_pose,
            constrain="align", pre_dis=0.08, dis=0.01))

        # Step 2: plate → sink
        arm = ArmTag("right")

        # self.grasp_actor_from_table(self.plate, arm_tag=arm, pre_grasp_dis=0.07)
        # self.move(self.move_by_displacement(arm_tag=arm, z=0.15))
        self.grasp_plate(self.plate)
        self.attach_object(self.plate,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/003_plate/collision/base0.glb",
            str(arm))
        self.enable_table(enable=True)
        self.move(self.place_actor(self.plate, arm_tag=arm, target_pose=self.sink_target_pose,
            constrain="align", pre_dis=0.08, dis=0.01))

        # # Step 3: knife → sink
        # arm = ArmTag("right" if self.knife.get_pose().p[0] > 0 else "left")
        # self.grasp_actor_from_table(self.knife, arm_tag=arm, pre_grasp_dis=0.07)
        # self.move(self.move_by_displacement(arm_tag=arm, z=0.15))
        # self.attach_object(self.knife,
        #     f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/034_knife/collision/base0.glb",
        #     str(arm))
        # self.enable_table(enable=True)
        # self.move(self.place_actor(self.knife, arm_tag=arm, target_pose=self.sink_target_pose,
        #     constrain="align", pre_dis=0.08, dis=0.01))

    def check_success(self):
        sg = self.kitchens_info["sink_geom"]
        sink_p = self.sink.get_pose().p
        bin_p = self.bin.get_pose().p

        ap = self.apple.get_pose().p
        pp = self.plate.get_pose().p
        # kp = self.knife.get_pose().p

        apple_in_bin = (abs(ap[0] - bin_p[0]) < 0.08 and abs(ap[1] - bin_p[1]) < 0.07
                        and ap[2] < bin_p[2] + 0.10)
        plate_in_sink = (abs(pp[0] - sink_p[0]) < sg["hole_hx"]
                         and abs(pp[1] - sink_p[1]) < sg["hole_hy"]
                         and pp[2] < sink_p[2] + 0.01)
        # knife_in_sink = (abs(kp[0] - sink_p[0]) < sg["hole_hx"]
        #                  and abs(kp[1] - sink_p[1]) < sg["hole_hy"]
        #                  and kp[2] < sink_p[2] + 0.01)
        return (apple_in_bin and plate_in_sink #and knife_in_sink
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
