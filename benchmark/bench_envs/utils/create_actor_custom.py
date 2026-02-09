import sapien.core as sapien
import numpy as np

def create_sapien_gltf_actor(
    scene,
    pose: sapien.Pose,
    gltf_path: str,
    collision_path: str = None,
    scale: float = 1.0,
    is_static: bool = False,
    name: str = None,
):
    """
    Create a SAPIEN actor from a GLTF file and an optional collision mesh.
    
    Args:
        scene: SAPIEN scene
        pose: Initial pose of the actor
        gltf_path: Path to the GLTF/GLB file for visual mesh
        collision_path: Optional path to collision mesh (OBJ/GLTF). If None, uses gltf_path
        scale: Uniform scale factor (or pass [x, y, z] list for non-uniform)
        is_static: If True, creates static actor. If False, creates dynamic actor
        name: Optional name for the actor
    
    Returns:
        SAPIEN Actor
    """
    # Convert scale to list if it's a single number
    if isinstance(scale, (int, float)):
        scale = [scale, scale, scale]
    
    # Use gltf_path for collision if no separate collision mesh provided
    if collision_path is None:
        collision_path = gltf_path
    
    # Create builder
    builder = scene.create_actor_builder()
    
    # Add visual mesh
    builder.add_visual_from_file(gltf_path, scale=scale)
    
    # Add collision mesh (nonconvex for complex meshes)
    builder.add_nonconvex_collision_from_file(collision_path, scale=scale)
    
    # Build the actor
    if is_static:
        actor = builder.build_static()
    else:
        actor = builder.build()
    
    # Set pose and name
    actor.set_pose(pose)
    if name:
        actor.set_name(name)
    
    return actor