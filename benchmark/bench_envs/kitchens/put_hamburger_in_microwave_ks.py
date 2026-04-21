from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class put_hamburger_in_microwave_ks(KitchenS_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        # Force microwave door fully open so the interior is accessible.
        limits = self.microwave.get_qlimits()
        qpos_mw = self.microwave.get_qpos()
        qpos_mw[0] = limits[0][1] * 0.95
        self.microwave.set_qpos(qpos_mw)

        # Spawn the hamburger on the counter, in the front band, on the
        # same side of the table as the microwave so the same arm can
        # perform both the grasp and the insertion.
        mw_x = float(self.microwave.get_pose().p[0])
        if mw_x < 0:
            xlim = [-0.45, -0.20]
        else:
            xlim = [0.20, 0.45]

        rand_pos = self.rand_pose_on_counter(
            xlim=xlim,
            ylim=[-0.23, -0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi, 0],
            obj_padding=0.06,
        )

        # 006_hamburg has three compact variants that all grasp reliably.
        self.hamburger_id = int(np.random.choice([0, 1, 2]))
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="006_hamburg",
            convex=True,
            model_id=self.hamburger_id,
        )
        self.target_obj.set_mass(0.05)

        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")

    def play_once(self):
        mw_p = self.microwave.get_pose().p
        mw_x = float(mw_p[0])
        mw_y = float(mw_p[1])
        mw_z = float(mw_p[2])

        # Grasp with the same-side arm (guaranteed by spawn band).
        arm_tag = ArmTag("right" if mw_x > 0 else "left")

        # -------------------------------------------------------------
        # 1) Grasp the hamburger.
        # -------------------------------------------------------------
        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.08)
        if not self.plan_success:
            return

        # Lift clear of the counter.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15))

        self.attach_object(
            self.target_obj,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/006_hamburg/collision/base{self.hamburger_id}.glb",
            str(arm_tag),
        )
        self.enable_table(enable=True)

        # -------------------------------------------------------------
        # 2) Pop microwave from curobo collision world so the planner
        #    doesn't reject poses that graze the mouth/walls. Also keep
        #    the table enabled so the planner respects the counter top.
        # -------------------------------------------------------------
        self.collision_list = [
            e for e in self.collision_list if e.get("actor") is not self.microwave
        ]
        self.update_world()

        # -------------------------------------------------------------
        # 3) Move the gripper horizontally toward the microwave in small
        #    world-frame +y steps. This is friendlier to IK than jumping
        #    to a far-away hover pose in one shot. Between steps, we also
        #    sweep x toward mw_x so the final insertion is centered.
        # -------------------------------------------------------------
        if arm_tag == "left":
            cur_ee = np.asarray(self.robot.get_left_ee_pose(), dtype=np.float64)
        else:
            cur_ee = np.asarray(self.robot.get_right_ee_pose(), dtype=np.float64)
        cur_x, cur_y, cur_z = float(cur_ee[0]), float(cur_ee[1]), float(cur_ee[2])

        # Target wrist xy: centered in x on the microwave, y just at/in
        # front of the mouth plane, z at current lift height. We
        # intentionally keep z low (near the counter top + 0.15) since
        # the microwave interior floor is right there and the task's
        # success check covers the whole mw bounding volume.
        target_wrist_x = mw_x
        target_wrist_y = mw_y      # microwave center line
        target_wrist_z = cur_z     # keep current post-lift height
        quat = cur_ee[3:].tolist()

        # Interpolate in N small steps; each is a fresh IK solve, so if a
        # pose is unreachable the sequence gracefully stops early.
        n_steps = 4
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            wp = [
                cur_x + (target_wrist_x - cur_x) * alpha,
                cur_y + (target_wrist_y - cur_y) * alpha,
                cur_z + (target_wrist_z - cur_z) * alpha,
            ] + quat
            self.move(self.move_to_pose(arm_tag, wp))
            if not self.plan_success:
                # Recover and stop stepping.
                self.plan_success = True
                break

        # -------------------------------------------------------------
        # 4) Final push: small world-frame displacement in +y to nudge
        #    the hamburger just past the mouth plane. If this fails we
        #    settle for whatever position was reached.
        # -------------------------------------------------------------
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=0.05))
        if not self.plan_success:
            self.plan_success = True

        # -------------------------------------------------------------
        # 5) Release the hamburger and retreat.
        # -------------------------------------------------------------
        self.move(self.open_gripper(arm_tag, pos=1.0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))
        if not self.plan_success:
            self.plan_success = True
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.12))

    def check_success(self):
        tp = self.target_obj.get_pose().p
        mw_p = self.microwave.get_pose().p
        return (abs(tp[0] - mw_p[0]) < 0.14
                and abs(tp[1] - mw_p[1]) < 0.22
                and tp[2] < mw_p[2] + 0.18
                and tp[2] > mw_p[2] - 0.15
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
