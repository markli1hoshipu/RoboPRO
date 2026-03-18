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
            closed_qpos = np.array(self.cabinet.get_qpos(), dtype=float)
            self.cabinet_closed_qpos = closed_qpos.copy()

            # Compute an open configuration by moving each finite-interval joint to its upper limit
            try:
                qlimits = np.array(self.cabinet.get_qlimits(), dtype=float)  # shape (dof, 2)
            except Exception:
                qlimits = None

            open_qpos = closed_qpos.copy()
            if qlimits is not None and qlimits.shape[0] >= open_qpos.shape[0]:
                for i in range(open_qpos.shape[0]):
                    low, high = qlimits[i]
                    if np.isfinite(low) and np.isfinite(high) and high > low:
                        open_qpos[i] = high

            self.cabinet.set_qpos(open_qpos)
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
        # Success: cabinet articulation has returned close to the recorded closed configuration
        if self.cabinet is None:
            return False
        if not hasattr(self, "cabinet_closed_qpos") or self.cabinet_closed_qpos is None:
            return False

        current_qpos = np.array(self.cabinet.get_qpos(), dtype=float)
        if current_qpos.shape != self.cabinet_closed_qpos.shape:
            return False

        delta = np.abs(current_qpos - self.cabinet_closed_qpos)
        return np.max(delta) < 0.02

