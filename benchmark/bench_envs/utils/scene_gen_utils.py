import numpy as np
import sapien
from transforms3d.euler import euler2quat
from envs.utils.create_actor import create_actor
from envs.utils.rand_create_actor import rand_pose

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
                    pass
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
        if  (ob[0][0] - x_thr <= obj_pose.p[0] <= ob[1][0] + x_thr) and \
            (ob[0][1] - y_thr <= obj_pose.p[1] <= ob[1][1] + y_thr):
            return True
    return False

def place_actor(obj_name, scene, task_objs, col_thr=0.15, object_bounds=None, 
                obj_id = None, mass = 0.1,  xlim=None, ylim=None, obj_pose=None, 
                qpos=(0,0,0), rotation=False, rotate_lim = (0,0,0)):
    
    if obj_pose is None:
        # Threshold between the objects
        while True:
            obj_pose = rand_pose(
                xlim=xlim,
                ylim=ylim,
                qpos=euler2quat(*[np.deg2rad(d) for d in qpos]), #[0.5, 0.5, 0.5, 0.5],
                rotate_rand=rotation,
                rotate_lim=rotate_lim,
            )
            if not get_collison_with_objs(object_bounds, obj_pose, col_thr):
                break
    if isinstance(obj_pose, list):
        obj_pose = sapien.Pose(obj_pose[0],
                    euler2quat(*[np.deg2rad(d) for d in obj_pose[1]]))
    obj_id = obj_id if obj_id is not None else np.random.choice(task_objs['objects']['study']['targets'][obj_name])
    
    print_c(f"Generating {obj_name} with id {obj_id} at position {obj_pose}", "BLUE")

    obj = create_actor(
            scene=scene,
            pose=obj_pose,
            modelname=obj_name,
            convex=True,
            model_id= obj_id,
            scale= None if task_objs['scales'].get(obj_name) is None else \
                task_objs['scales'][obj_name].get(str(obj_id)) 
    )
    obj.set_mass(mass)
    
    # To compensate for the pose offset
    bbox = get_actor_boundingbox(obj.actor)
    if bbox[0][-1] < obj_pose.p[-1]:
        new_p = [obj_pose.p[0], obj_pose.p[1], 
                    2*obj_pose.p[2]- bbox[0][2]]
        obj.actor.set_pose(
            sapien.Pose(
                p = new_p, 
                q = obj_pose.q)
        )
        print_c(f"Pose adjusted to {new_p}", "YELLOW")
    return obj, obj_id, obj.get_pose()


TEXT_COLOR = {"RED": '\033[31m',
            "GREEN": '\033[32m',
            "BLUE": '\033[34m',
            "RESET": '\033[0m',
            "YELLOW": '\033[33m',
            "MAGENTA": '\033[35m',
            "CYON": '\033[36m',
             "WHITE": '\033[37m'   }
def print_c(text, color="WHITE"):
    print(f"{TEXT_COLOR[color]}{text}{TEXT_COLOR['RESET']}")