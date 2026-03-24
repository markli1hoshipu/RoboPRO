from bench_envs._kitchen_base_large import Kitchen_base_large
from envs.utils import *
import sapien
import math
import numpy as np
import transforms3d as t3d
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class open_cabinet(Kitchen_base_large):

    def setup_demo(self, is_test: bool = False, **kwargs):
        # Match collision-cache usage from other benchmark tasks
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
        self._boost_gripper_handle_friction()

    def _boost_gripper_handle_friction(self) -> None:
        """
        Increase contact friction specifically between robot gripper links and
        cabinet doors/handles by assigning a high-friction physical material to
        their collision shapes.
        """
        if not hasattr(self, "scene") or self.scene is None:
            return

        # Keep restitution at 0 to avoid bounce.
        high_friction_mat = self.scene.create_physical_material(2.5, 2.5, 0.0)

        def _set_link_shapes_material(link_obj) -> None:
            if link_obj is None:
                return
            try:
                if hasattr(link_obj, "get_collision_shapes"):
                    shapes = link_obj.get_collision_shapes()
                elif hasattr(link_obj, "collision_shapes"):
                    cs = getattr(link_obj, "collision_shapes")
                    shapes = cs() if callable(cs) else cs
                else:
                    return
            except Exception:
                return

            for shape in list(shapes):
                try:
                    if hasattr(shape, "set_physical_material"):
                        shape.set_physical_material(high_friction_mat)
                except Exception:
                    continue

        # Robot gripper links (names are pre-collected in robot.gripper_name).
        gripper_names = set(getattr(self.robot, "gripper_name", []))
        for entity_name in ("left_entity", "right_entity"):
            entity = getattr(self.robot, entity_name, None)
            if entity is None:
                continue
            try:
                for link in entity.get_links():
                    if link.get_name() in gripper_names:
                        _set_link_shapes_material(link)
            except Exception:
                continue

        # Cabinet door links where the handle geometry lives.
        if hasattr(self, "cabinet") and self.cabinet is not None:
            for door_link_name in ("rightdoor", "leftdoor"):
                door_link = self.cabinet.link_dict.get(door_link_name)
                _set_link_shapes_material(door_link)

    def load_actors(self):
        # No additional movable actors are needed for opening the cabinet.
        # Record right-door-only cabinet state for success checking.
        if hasattr(self, "cabinet") and self.cabinet is not None:
            self._init_cabinet_states()
        else:
            self.cabinet_closed_qpos = None

    def pull_door_circularly(self, arm_tag, door_radius, total_open_angle=45.0, num_steps=30):
        """
        Execute a circular pull trajectory for a hinged cabinet door.
        """
        d_angle = np.deg2rad(total_open_angle) / num_steps
        current_angle = 0.0

        # Keep orientation anchored to the grasp-time TCP orientation.
        start_pose = np.array(self.get_arm_pose(arm_tag), dtype=float)
        start_q = start_pose[3:]

        for _ in range(num_steps):
            next_angle = current_angle + d_angle

            # Arc displacement in x-y plane.
            x_curr = door_radius * (1.0 - np.cos(current_angle))
            y_curr = -door_radius * np.sin(current_angle)
            x_next = door_radius * (1.0 - np.cos(next_angle))
            y_next = -door_radius * np.sin(next_angle)

            dx = x_next - x_curr
            dy = y_next - y_curr

            # Rotate TCP with the pulling arc around world z-axis.
            dq_step = np.array(t3d.quaternions.axangle2quat([0.0, 0.0, 1.0], next_angle), dtype=float)
            target_q = np.array(t3d.quaternions.qmult(dq_step, start_q), dtype=float)
            target_q = target_q / max(np.linalg.norm(target_q), 1e-9)

            self.move(
                self.move_by_displacement(
                    arm_tag=arm_tag,
                    x=dx,
                    y=dy,
                    quat=target_q.tolist(),
                )
            )

            current_angle = next_angle

    def play_once(self):
        # Provide a simple info mapping for downstream use
        arm_tag = ArmTag("right")

        # Keep the initial TCP pose for downstream stages.
        initial_tcp_pose = np.array(self.get_arm_pose(arm_tag), dtype=float)
        initial_tcp_pos = initial_tcp_pose[:3].tolist()
        initial_tcp_quat = initial_tcp_pose[3:]

        # Rotate TCP by +90 degrees around the local/world y-axis before moving.
        q_rot_y_90 = np.array(t3d.quaternions.axangle2quat([0.0, 1.0, 0.0], np.pi / 2.0), dtype=float)
        rotated_tcp_quat = np.array(t3d.quaternions.qmult(q_rot_y_90, initial_tcp_quat), dtype=float)
        rotated_tcp_quat = rotated_tcp_quat / max(np.linalg.norm(rotated_tcp_quat), 1e-9)

        # Move to the initial TCP pose and rotate by +90 degrees around the y-axis.
        self.move(self.move_by_displacement(arm_tag=arm_tag, quat=rotated_tcp_quat.tolist()))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=-0.17, y=0.2, z=0.30))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=0.054))
        self.move(self.close_gripper(arm_tag=arm_tag, pos=-0.1))

        # Pull the cabinet door with a circular trajectory (same style as open_fridge).
        self.pull_door_circularly(
            arm_tag=arm_tag,
            door_radius=0.21,
            total_open_angle=45,
            num_steps=30,
        )

        # Move back to the initial TCP pose and rotate by -90 degrees around the y-axis.
        self.move(self.close_gripper(arm_tag=arm_tag, pos=1.0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.05, y=-0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=-0.25))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=0.2))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.25, y=-0.2))

        self.info["info"] = {
            "{A}": "122_cabinet_nkrgez",
            "{tcp_init_pos}": str(initial_tcp_pos),
        }
        return self.info

    def check_success(self):
        # Success: only right-door joint has moved away from closed.
        return self.is_cabinet_open(threshold=0.02)

