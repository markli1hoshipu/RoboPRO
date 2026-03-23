from bench_envs._kitchen_base_large import Kitchen_base_large
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob
import numpy as np


class close_cabinet(Kitchen_base_large):
    """
    Task: start with the cabinet doors open and close them.
    """

    def setup_demo(self, is_test: bool = False, **kwargs):
        # Match collision-cache usage from other benchmark tasks
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

        # Capture the closed configuration before opening
        if hasattr(self, "cabinet") and self.cabinet is not None:
            self._init_cabinet_states()
            # Start close-cabinet task from right-door-open state only.
            self.set_cabinet_open()
        else:
            self.cabinet_closed_qpos = None

    def load_actors(self):
        # Nothing extra to load; the cabinet qpos has already been set in setup_demo.
        pass

    def play_once(self):
        # Provide a simple info mapping for downstream use
        self.info["info"] = {
            "{A}": "122_cabinet_nkrgez",
        }
        return self.info

    def check_success(self):
        # Success: only right-door joint is back to closed.
        return self.is_cabinet_closed(threshold=0.02)

