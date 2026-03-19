# from envs._base_task import Base_Task
from bench_envs.study._study_base_task import Study_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob
import yaml
import os

def get_actor_boundingbox(entity):
    all_points = []
    try:
        actor = entity.actor
    except:
        actor = entity
    
    entity_mat = actor.pose.to_transformation_matrix()
    for comp in actor.get_components():
        # Check if component has collision shapes (RigidBodyComponent)
        if hasattr(comp, 'get_collision_shapes'):
            for shape in comp.get_collision_shapes():
                # 1. Get local vertices [N, 3]
                try:
                    local_v = shape.get_vertices() 
                except:
                    hs = shape.half_size  # e.g., [0.1, 0.1, 0.1]

                    # 2. Define the 8 corners in the shape's local frame
                    local_v = np.array([[x, y, z] for x in [-hs[0], hs[0]] 
                                for y in [-hs[1], hs[1]] 
                                for z in [-hs[2], hs[2]]])
                try:
                    local_v *= shape.scale
                except:
                    print("table does not have scale")
                # 2. Get shape's offset relative to the entity center
                shape_mat = shape.get_local_pose().to_transformation_matrix()
                
                # 3. Combine: Entity World * Shape Local
                world_mat = entity_mat @ shape_mat
                
                # 4. Transform points to world space
                homo_v = np.pad(local_v, ((0, 0), (0, 1)), constant_values=1)
                world_v = (world_mat @ homo_v.T).T[:, :3]
                all_points.append(world_v)

    if not all_points:
        return None, None

    points_cloud = np.vstack(all_points)
    return points_cloud.min(axis=0), points_cloud.max(axis=0)

def get_position_limits(surface_obj, boundary_thr = 0.15, robot_reach_thr = 0.6,
                       arm_x_pose = 0.15, side=None):
    # Get generation limits
    # Assumption is that the robot is centered with respect to the the surface area
    try:
        actor = surface_obj.actor
    except:
        actor = surface_obj
    table_bb = get_actor_boundingbox(actor)
    print(f"surface bb {table_bb}")

    side_to_place = side or np.random.choice(["left", "right"])

    if side_to_place == "left":
        xmin = max((table_bb[0][0] + boundary_thr),  (-arm_x_pose - robot_reach_thr))
        xmax = min((-arm_x_pose + robot_reach_thr), arm_x_pose)
        xlim=[xmin, xmax ] # 0.9 is the range of reach for the robot arm
    else:
        xmax = min((table_bb[1][0] - boundary_thr),  (arm_x_pose + robot_reach_thr))
        xmin = max((arm_x_pose - robot_reach_thr), -arm_x_pose)
        xlim=[xmin, xmax] # 0.9 is the range of reach for the robot arm
    ylim= [table_bb[0][1] + boundary_thr, table_bb[1][1]- boundary_thr]

    return xlim, ylim, side_to_place


def get_random_valid_placement(table_bounds, object_bounds, new_w, new_h):
    tx1, tx2 = table_bounds[0]
    ty1,ty2 = table_bounds[1]
    
    # 1. Generate candidate X and Y coordinates from all edges
    # We also add a small buffer (0.01) if you want objects not to touch exactly
    x_coords = sorted(list(set([tx1, tx2 - new_w] + [o for o in object_bounds if o <= tx2 - new_w] + [o for o in object_bounds if o <= tx2 - new_w])))
    y_coords = sorted(list(set([ty1, ty2 - new_h] + [o for o in object_bounds if o <= ty2 - new_h] + [o for o in object_bounds if o <= ty2 - new_h])))

    valid_spots = []

    # 2. Collect ALL valid (x, y) coordinates
    for x in x_coords:
        for y in y_coords:
            nx1, ny1, nx2, ny2 = x, y, x + new_w, y + new_h
            
            # Boundary check
            if nx2 > tx2 or ny2 > ty2 or nx1 < tx1 or ny1 < ty1:
                continue
                
            # Collision check against all existing objects
            collision = False
            for ox1, oy1, ox2, oy2 in object_bounds:
                if not (nx2 <= ox1 or nx1 >= ox2 or ny2 <= oy1 or ny1 >= oy2):
                    collision = True
                    break
            
            if not collision:
                valid_spots.append((nx1, ny1, nx2, ny2))

    # 3. Pick one at random
    if not valid_spots:
        return None
    
    return np.random.choice(valid_spots)


def get_collison_with_objs(object_bounds, obj_pose, x_thr, y_thr = None):

    y_thr = y_thr or x_thr
    for ob in object_bounds:
        if (obj_pose.p[0] > ob[0][0] - x_thr and \
            obj_pose.p[0] < ob[0][1] + x_thr) and \
            (obj_pose.p[1] > ob[1][0] - y_thr and \
                obj_pose.p[1] < ob[1][1] + y_thr):
            return True
    return False

class put_cup_in_box(Study_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r") as f:
            task_objs = yaml.safe_load(f)
        
        xlim, ylim, self.side_to_place = get_position_limits(self.table, boundary_thr=0.20, side="left")
       
        print(xlim, ylim, self.side_to_place)
        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]

        for bb, o in zip(object_bounds, self.scene_objs):
            print(o.get_name(), bb)
        # Threshold between the objects
        col_thr = 0.15


        while True:
            tar_obj_rand_pos = rand_pose(
                xlim=xlim,
                ylim=ylim,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                # rotate_lim=[0, 3.14, 0],
            )
            if not get_collison_with_objs(object_bounds, tar_obj_rand_pos, col_thr):
                break
                    
            # if abs(tar_obj_rand_pos.p[0]) > 0.3:
            #     break
      
        self.target_name = "021_cup"# np.random.choice(list(task_objs['train']['study']['targets'].keys()))
        self.target_id = np.random.choice(task_objs['objects']['study']['targets'][self.target_name])
        
        print(f"Generating {self.target_name} with id {self.target_id} at position {tar_obj_rand_pos}")

        self.target_obj = create_actor(
            scene=self,
            pose=tar_obj_rand_pos,
            modelname=self.target_name,
            convex=True,
            model_id= self.target_id ,
            scale= None if task_objs['scales'].get(self.target_name) is None else  task_objs['scales'][self.target_name].get(str(self.target_id)) 
        )
        self.target_obj.set_mass(0.1)
        # Create destination object
     
        self.des_obj = self.box

        des_bb = get_actor_boundingbox(self.des_obj.actor)
        p = self.des_obj.get_pose().p.tolist() 
        p[-1] = des_bb[1][-1]
        self.des_obj_pose = p + [1, 0, 0, 0]
        print(f"Placement destination pose {self.des_obj_pose}")


        self.add_prohibit_area(self.target_obj, padding=0.12, area="table")
        self.add_prohibit_area(self.des_obj, padding=0.12, area="table")

     
      
    def play_once(self, z = 0.1, pre_dis= 0.07, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side_to_place ) #("right" if self.target_obj.get_pose().p[0] > 0 else "left")

        # Grasp the mouse with the selected arm
        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))

        # Lift the mouse upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=z))

        self.attach_object(self.target_obj, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{self.target_name}/collision/base{self.target_id}.glb", str(arm_tag))

        # Place the mouse at the target location with alignment constraint
        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose= self.des_obj_pose,
                constrain="auto",
                pre_dis=pre_dis,
                dis=dis,
            ))

        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"{self.target_name}/base{self.target_id}",
            "{B}": f"red",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        target_pose = self.target_obj.get_pose().p
        target_qpose = np.abs(self.target_obj.get_pose().q)
        target_des_pos = self.target_obj.get_pose().p
        eps1 = 0.015
        eps2 = 0.012

        return (np.all(abs(target_pose[:2] - target_des_pos[:2]) < np.array([eps1, eps2]))
                and (np.abs(target_qpose[2] * target_qpose[3] - 0.49) < eps1
                     or np.abs(target_qpose[0] * target_qpose[1] - 0.49) < eps1) and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
