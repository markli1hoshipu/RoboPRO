import sapien.core as sapien
import numpy as np
from pathlib import Path
import os
from .actor_utils_custom import Simple_Actor

def create_multiple_obj_actor(
    scene,
    pose: sapien.Pose,
    visual_path: str,
    collision_path: str = None,
    scale: float = 1.0,
    is_static: bool = False,
    name: str = None,
):
    """
    Create a SAPIEN actor from a GLTF visual file and multiple OBJ collision files.
    
    Args:
        scene: SAPIEN scene
        pose: Initial pose of the actor
        visual_path: Path to the GLTF/GLB file for visual mesh
        collision_path: Path to directory containing multiple OBJ files for collision meshes
        scale: Uniform scale factor (or pass [x, y, z] list for non-uniform)
        is_static: If True, creates static actor. If False, creates dynamic actor
        name: Optional name for the actor
    
    Returns:
        SAPIEN Actor
    """
    # Convert scale to list if it's a single number
    if isinstance(scale, (int, float)):
        scale = [scale, scale, scale]
    
    # Create builder
    builder = scene.create_actor_builder()
    
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")
    
    # Add visual mesh from GLTF file
    builder.add_visual_from_file(visual_path, scale=scale)
    
    # Add collision meshes from all OBJ files in the collision directory
    if collision_path is not None:
        collision_dir = Path(collision_path)
        if collision_dir.exists() and collision_dir.is_dir():
            # Find all .obj files in the directory
            obj_files = sorted(collision_dir.glob("*.obj"))
            if len(obj_files) == 0:
                raise FileNotFoundError(f"No OBJ files found in collision directory: {collision_path}")
            
            # Add each OBJ file as a convex collision mesh
            for obj_file in obj_files:
                builder.add_multiple_convex_collisions_from_file(filename=str(obj_file), scale=scale)
        else:
            raise ValueError(f"collision_path must be a valid directory: {collision_path}")
    
    # Build the actor
    if is_static:
        actor = builder.build_static()
    else:
        actor = builder.build()
    
    # Set pose and name
    actor.set_pose(pose)
    if name:
        actor.set_name(name)
    
    return Simple_Actor(actor, scale)