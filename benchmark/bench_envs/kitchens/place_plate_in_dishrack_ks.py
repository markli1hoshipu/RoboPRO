from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class place_plate_in_dishrack_ks(KitchenS_base_task):

    # Mirrors put_plate_in_sink_ks: top-down rim pinch, lift, drop above
    # dishrack. Only the drop target (rack instead of sink) differs.
    TOP_DOWN_Q = [-0.5, 0.5, -0.5, -0.5]
    # Home EE quat (gripper facing front) for aloha-agilex
    INIT_Q = [0.707, 0, 0, 0.707]
    TCP_OFFSET = 0.12

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        rack_p = self.dishrack.get_pose().p
        self.arm_tag = ArmTag("right" if rack_p[0] > 0 else "left")
        side_sign = 1 if self.arm_tag == "right" else -1

        # Plate spawn must clear both the rack and sink footprints.
        #   Rack: |x - rack_x| < (0.084 + 0.04 + plate_pad) → ~0.18.
        #   Sink: its prohibited box extends y up to faucet_y at y≈0.11,
        #         and sink y_min ≈ -0.15. Plate padded bbox must not touch.
        # Push plate y to the counter front (y ≤ -0.24) so the plate's
        # padded bbox (±0.06) never overlaps the sink y-range, freeing x to
        # be chosen purely from rack clearance + arm reachability.
        rack_x = float(rack_p[0])
        if side_sign > 0:  # right arm
            if rack_x > 0.35:
                # scene 1: rack at 0.42 → left of rack is available
                x_range = [0.10, 0.24]
            else:
                # scene 0: rack at 0.22 → right of rack (near sink x=0.42)
                x_range = [0.38, 0.48]
        else:  # left arm
            # scene 2: rack at -0.32 → right of rack
            x_range = [-0.22, -0.08]

        rand_pos = self.rand_pose_on_counter(
            xlim=x_range,
            ylim=[-0.20, -0.14],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            obj_padding=0.06,
        )

        self.plate_id = 0
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="003_plate",
            convex=True,
            model_id=self.plate_id,
        )
        self.target_obj.set_mass(0.02)

        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")

    def play_once(self):
        arm_tag = self.arm_tag
        side_sign = 1 if arm_tag == "right" else -1

        self.enable_table(enable=False)

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

        # Attach before the lift so the plate is rigidly welded to the gripper
        # and doesn't slip during the vertical move.
        self.attach_object(
            self.target_obj,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/003_plate/collision/base{self.plate_id}.glb",
            str(arm_tag),
        )

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.25))

        # Rack is in curobo collision while sink is not; pop it so the
        # planner isn't blocked by plate-rack overlap checks at the target.
        self.collision_list = [
            e for e in self.collision_list if e.get("actor") is not self.dishrack
        ]
        self.update_world()

        # Split the drop into two moves so curobo has a reachable midpoint
        # to seed from: first a hover pose above the rack basin, then the
        # drop. The plate rim is held by the wrist with an offset of
        # 0.085 * side_sign in x; we target rack basin center by drop_x +
        # 0.085*side_sign = rack_x (so the plate center sits over rack_x).
        # Drop y is rack_p[1] - 0.09 which puts the plate center at the
        # basin centroid (chain_dishrack_plate_bread_knife_ks uses the
        # same 0.09 offset for plate-on-rack placement).
        rack_p = self.dishrack.get_pose().p
        drop_x = float(rack_p[0]) 
        drop_y = float(rack_p[1]) - 0.35

        hover_drop_pose = [drop_x, drop_y-0.20, 1.20] + self.INIT_Q
        self.move(self.move_to_pose(arm_tag, hover_drop_pose))

        # Rack top z ≈ table_height + 0.091 ≈ 0.831. Plate needs to hover a
        # few cm above rack top before release, so plate center z ≈ 0.88 →
        # TCP z = plate_z + TCP_OFFSET = 1.00.
        drop_tcp_z = 0.88
        drop_pose = [drop_x, drop_y, drop_tcp_z + self.TCP_OFFSET] + self.INIT_Q
        self.move(self.move_to_pose(arm_tag, drop_pose))

        self.move(self.open_gripper(arm_tag, pos=1.0))

    def check_success(self):
        rack_p = self.dishrack.get_pose().p
        tp = self.target_obj.get_pose().p
        eps_x, eps_y = 0.12, 0.15
        return (abs(tp[0] - rack_p[0]) < eps_x
                and abs(tp[1] - rack_p[1]) < eps_y
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
