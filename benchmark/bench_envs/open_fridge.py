from bench_envs._kitchen_base_large import Kitchen_base_large
from envs.utils import *
import numpy as np


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

    def play_once(self):
        # For now, do not execute any robot motion; we only care about
        # the initial articulation state for visualization.
        self.info["info"] = {
            "{A}": "124_fridge_hivvdf",
        }
        return self.info

    def check_success(self):
        # Success if the fridge is in an open configuration relative to its
        # canonical closed state.
        return self.is_fridge_open()

