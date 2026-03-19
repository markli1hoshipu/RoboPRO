from bench_envs._kitchen_base_large import Kitchen_base_large
from envs.utils import *
import numpy as np


class close_fridge(Kitchen_base_large):

    def setup_demo(self, is_test: bool = False, **kwargs):
        # Match collision cache usage from other benchmark tasks
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

        # For a close-fridge task, start from an open configuration.
        self._init_fridge_states()
        # Initialize at a fixed 180-degree opening (fully open).
        self.fridge_start_open_angle_deg = 180.0
        self.set_fridge_open_angle_deg(self.fridge_start_open_angle_deg, open_span_deg=180.0)

    def load_actors(self):
        # No additional movable actors are required for simply closing the fridge.
        pass

    def play_once(self):
        # For now, do not execute any robot motion; we only care about
        # the initial articulation state for visualization.
        self.info["info"] = {
            "{A}": "124_fridge_hivvdf",
        }
        return self.info

    def check_success(self):
        # Success if the fridge door is at (or very near) the canonical closed pose.
        return self.is_fridge_closed()

