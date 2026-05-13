"""Robot placement logic."""
from typing import Dict, List, Optional, Any
from creator.robots.catalog import get_robot_info


def place_robots(
    robot_ids: List[str],
    placed_models: List[Dict[str, Any]],
    room_half_size: float = 5.0
) -> List[Dict[str, Any]]:
    """Place robots in the scene.
    
    Args:
        robot_ids: List of robot IDs to place
        placed_models: Already placed objects (furniture, etc.)
        room_half_size: Room size for bounds checking
        
    Returns:
        List of robot placements with {robot_id, pos, yaw, xml_path}
    """
    placements = []
    table_robot_count = 0  # Track how many robots placed on table
    
    for robot_id in robot_ids:
        info = get_robot_info(robot_id)
        if not info:
            continue
            
        placement_type = info.get("placement", "floor")
        
        if placement_type == "table_mounted":
            placement = _place_on_table(robot_id, info, placed_models, room_half_size, table_robot_count)
            if placement:
                table_robot_count += 1
        else:  # floor
            placement = _place_on_floor(robot_id, info, placed_models, room_half_size)
        
        if placement:
            placements.append(placement)
    
    return placements


def _place_on_table(
    robot_id: str,
    info: Dict,
    placed_models: List[Dict],
    room_half_size: float,
    robot_index: int = 0
) -> Optional[Dict]:
    """Place manipulator on a table surface."""
    # Find largest table/desk
    tables = [
        m for m in placed_models
        if any(kw in str(m.get("Model", "")).lower() for kw in ["table", "desk", "counter"])
    ]
    
    if not tables:
        # No table found - place on floor instead
        return _place_on_floor(robot_id, info, placed_models, room_half_size)
    
    # Choose largest table by surface area (X*Z)
    table = max(tables, key=lambda t: t.get("size", [1,1,1])[0] * t.get("size", [1,1,1])[2])
    table_pose = table.get("Pose", {})
    table_size = table.get("size", [1, 0.75, 1])  # [X, Y=height, Z]
    
    tx = float(table_pose.get("x", 0.0))
    ty = float(table_pose.get("y", 0.0))
    tz = float(table_pose.get("z", 0.0))
    
    # Table top Z = center_z + half_height
    table_top_z = tz + float(table_size[1]) / 2.0
    
    # Place robots at different positions to avoid overlap
    # First robot: left side, second robot: right side
    side_multiplier = 1 if robot_index % 2 == 0 else -1
    offset_x = float(table_size[0]) * 0.3 * side_multiplier
    offset_y = float(table_size[2]) * 0.2 * (robot_index // 2)  # Stagger in Y if more than 2
    
    robot_x = tx + offset_x
    robot_y = ty + offset_y
    robot_z = table_top_z
    
    xml_path = info['xml_file']
    
    return {
        "robot_id": robot_id,
        "pos": [robot_x, robot_y, robot_z],
        "yaw": 180.0 if side_multiplier > 0 else 0.0,  # Face toward table center
        "xml_path": xml_path
    }


def _place_on_floor(
    robot_id: str,
    info: Dict,
    placed_models: List[Dict],
    room_half_size: float
) -> Dict:
    """Place robot on floor (quadrupeds, mobile bases)."""
    # Place near room center with small offset to avoid exact center
    robot_x = 1.0
    robot_y = 0.0
    # Floor is at z=0, robot base sits on floor
    robot_z = 0.0
    
    xml_path = info['xml_file']
    
    return {
        "robot_id": robot_id,
        "pos": [robot_x, robot_y, robot_z],
        "yaw": 180.0,  # Face toward center
        "xml_path": xml_path
    }


def _get_volume(size: List[float]) -> float:
    """Calculate volume from size."""
    if len(size) < 3:
        return 0.0
    return float(size[0]) * float(size[1]) * float(size[2])
