import sapien
from bench_envs.kitchen._kitchens_base_task import KitchenS_base_task
from envs._GLOBAL_CONFIGS import *


class base_scene_ks(KitchenS_base_task):
    """Base KitchenS scene — structural fixtures only, no decor/task objects."""

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)
        # Remove decor objects placed by the base task
        HIDE = sapien.Pose(p=[0, 0, -10])
        for attr in ("static_board", "breadbasket", "trash_bin"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.actor.set_pose(HIDE)

    def load_actors(self):
        pass  # No task objects

    def play_once(self):
        return self.info

    def check_success(self):
        return True
