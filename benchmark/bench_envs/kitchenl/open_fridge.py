from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from envs.utils import *
import numpy as np
import transforms3d as t3d


class open_fridge(Kitchen_base_large):

    def setup_demo(self, is_test: bool = False, **kwargs):
        # Match collision cache usage from other benchmark tasks
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

        # For an open-fridge task, the environment should start with the fridge closed.
        self._init_fridge_states()
        self.set_fridge_closed()

    def load_actors(self):
        # No additional movable actors are required for simply opening the fridge.
        pass

    def pull_door_circularly(self, arm_tag, door_radius, total_open_angle=45.0, num_steps=30):
        """
        Executes a circular trajectory to pull open a hinged door.
        
        Args:
            arm_tag: The identifier for the robotic arm.
            door_radius (float): The estimated distance from the hinge to the handle in meters.
            total_open_angle (float): Total degrees to swing the door open.
            num_steps (int): Number of intermediate waypoints. Higher = smoother.
        """
        d_angle = np.deg2rad(total_open_angle) / num_steps
        current_angle = 0.0
        
        # Capture the orientation immediately after grasping. 
        # This acts as our mathematical anchor to prevent quaternion drift.
        start_pose = np.array(self.get_arm_pose(arm_tag), dtype=float)
        start_q = start_pose[3:]

        for i in range(num_steps):
            next_angle = current_angle + d_angle
            
            # Compute Position Deltas (dx, dy)
            x_curr = door_radius * (1 - np.cos(current_angle))
            y_curr = -door_radius * np.sin(current_angle)
            
            x_next = door_radius * (1 - np.cos(next_angle))
            y_next = -door_radius * np.sin(next_angle)
            
            dx = x_next - x_curr
            dy = y_next - y_curr
            
            # Compute New Orientation Target
            dq_step = np.array(t3d.quaternions.axangle2quat([0.0, 0.0, 1.0], next_angle), dtype=float)
            target_q = np.array(t3d.quaternions.qmult(dq_step, start_q), dtype=float)
            target_q = target_q / max(np.linalg.norm(target_q), 1e-9) # Normalize
            
            # Execute the incremental step
            self.move(self.move_by_displacement(
                arm_tag=arm_tag, 
                x=dx, 
                y=dy, 
                quat=target_q.tolist()
            ))
            
            current_angle = next_angle

    def play_once(self):
        arm_tag = ArmTag("right")
        initial_tcp_quat = np.array(self.get_arm_pose(arm_tag), dtype=float)[3:].tolist()

        # 1. Move straight to the handle and close the gripper
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=-0.05, y=0.255))
        self.move(self.close_gripper(arm_tag=arm_tag, pos=0.0))

        # 2. Execute the circular pull
        # Tune the door_radius here to match the physical fridge
        self.pull_door_circularly(
            arm_tag=arm_tag, 
            door_radius=0.23,
            total_open_angle=45.0, 
            num_steps=30
        )
        self.move(self.close_gripper(arm_tag=arm_tag, pos=1.0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.03, y=-0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=-0.2))
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=0.2))
        self.move(self.move_by_displacement(arm_tag=arm_tag, quat=initial_tcp_quat))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=0.21, y=-0.2))
 
        self.info["info"] = {
            "{A}": "124_fridge_hivvdf",
        }
        return self.info

    def check_success(self):
        # Success if the fridge door is at (or very near) the canonical fully-open pose
        # (now capped by the URDF joint limit, i.e., 90 degrees max).
        return self.is_fridge_fully_open()

