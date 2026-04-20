from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class place_plate_in_dishrack_ks(KitchenS_base_task):

    # Scripted top-down rim pinch (same recipe as the working
    # put_plate_in_sink_ks) plus scene-aware arm selection so that when
    # the dishrack is on the left of the counter (scene 2) the left arm
    # grasps and places, avoiding a cross-body carry.
    TOP_DOWN_Q = [-0.5, 0.5, -0.5, -0.5]
    TCP_OFFSET = 0.12

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        rack_p = self.dishrack.get_pose().p
        # Scene 2 has rack at x=-0.32: use the left arm so the grasp +
        # carry stays on the same side of the midline.
        self.arm_tag = ArmTag("right" if rack_p[0] > 0 else "left")
        side_sign = 1 if self.arm_tag == "right" else -1

        if side_sign > 0:
            x_range = [0.30, 0.45]
        else:
            x_range = [-0.45, -0.30]

        rand_pos = self.rand_pose_on_counter(
            xlim=x_range,
            ylim=[-0.15, -0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            obj_padding=0.10,
        )

        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="003_plate",
            convex=True,
            model_id=0,
        )
        self.target_obj.set_mass(0.1)

        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")

    def play_once(self):
        arm_tag = self.arm_tag
        side_sign = 1 if arm_tag == "right" else -1

        # Drop the counter and the dishrack from Curobo so the wrist can
        # swing a large plate over the rack without registering as in
        # collision with the prongs.
        self.enable_table(enable=False)
        _rack_pose = self.dishrack.get_pose()
        _rack_np_pose = np.concatenate([_rack_pose.p, _rack_pose.q]).tolist()
        _rack_mesh_name = f"dishrack_{_rack_np_pose}_{self.seed}"
        self.enable_obstacle(False, mesh_names=[_rack_mesh_name])

        self.move(self.open_gripper(arm_tag, pos=1.0))

        plate_p = self.target_obj.get_pose().p
        rim_x = float(plate_p[0]) - 0.085 * side_sign
        rim_y = float(plate_p[1])
        rim_z_top = float(plate_p[2]) + 0.012

        hover_tcp_z = rim_z_top + 0.10
        hover_pose = [rim_x, rim_y, hover_tcp_z + self.TCP_OFFSET] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, hover_pose))

        grasp_tcp_z = rim_z_top - 0.002
        grasp_pose = [rim_x, rim_y, grasp_tcp_z + self.TCP_OFFSET] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, grasp_pose))

        self.move(self.close_gripper(arm_tag, pos=0.0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.25))

        self.attach_object(
            self.target_obj,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/003_plate/collision/base0.glb",
            str(arm_tag),
        )

        rack_p = self.dishrack.get_pose().p
        drop_tcp_z = float(rack_p[2]) + 0.25
        drop_pose = [
            float(rack_p[0]) - 0.085 * side_sign,
            float(rack_p[1]) - 0.09,
            drop_tcp_z + self.TCP_OFFSET,
        ] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, drop_pose))

        self.move(self.open_gripper(arm_tag, pos=1.0))

    def check_success(self):
        tp = self.target_obj.get_pose().p
        rack_p = self.dishrack.get_pose().p
        eps_x, eps_y = 0.12, 0.12
        return (abs(tp[0] - rack_p[0]) < eps_x
                and abs(tp[1] - (rack_p[1] - 0.09)) < eps_y
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
