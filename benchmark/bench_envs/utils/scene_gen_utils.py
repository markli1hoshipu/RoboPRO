import numpy as np
from scipy.spatial.transform import Rotation as R

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

def euler_to_quat(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw])
    q = r.as_quat()  # returns [x, y, z, w]
    q = [q[3], q[0], q[1], q[2]] # return as [w, x, y, z]
    return q
