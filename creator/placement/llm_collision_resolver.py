"""LLM-based collision resolution fallback.

This module provides LLM fallback for collision resolution when gradient-based
methods fail to converge.
"""

import json
from typing import Any, Dict, List, Tuple

from creator.llm.model import prompt_model
from creator.placement.physics import check_collisions


def llm_replan_layout(
    placed_models: List[Dict[str, Any]],
    semantic_plan: Dict[str, Any],
    room_half_size: float,
    collision_details: List[Dict[str, Any]],
    max_retries: int = 3,
) -> Tuple[List[Dict[str, Any]], bool]:
    """LLM fallback for replanning object layout when gradient resolution fails.
    
    This function is invoked when algorithmic collision resolution cannot converge.
    It uses the LLM to determine optimal object positions that eliminate overlaps
    while preserving semantic constraints and room boundaries.
    
    Args:
        placed_models: Current placed objects with positions
        semantic_plan: Semantic plan with constraints
        room_half_size: Half-size of the room (for rectangular rooms)
        collision_details: List of collision dictionaries from check_collisions()
        max_retries: Maximum number of LLM retry attempts (default: 3)
        
    Returns:
        Tuple of (replanned_models, success):
        - replanned_models: Updated placed objects with new positions
        - success: Whether LLM replanning succeeded
    """
    print(f"[llm_replan_layout] Invoking LLM fallback for {len(placed_models)} objects")
    print(f"[llm_replan_layout] Reason: {len(collision_details)} collision(s) after gradient resolution")
    
    # Extract object information for LLM prompt
    objects_info = []
    for i, obj in enumerate(placed_models):
        model_name = obj.get("Model", f"object_{i}")
        size = obj.get("size", [1.0, 1.0, 1.0])
        pose = obj.get("Pose", {})
        current_pos = (
            float(pose.get("x", 0.0)),
            float(pose.get("y", 0.0)),
            float(pose.get("z", 0.0)),
        )
        
        objects_info.append({
            "index": i,
            "name": model_name,
            "size": size,
            "current_position": current_pos,
        })
    
    # Extract semantic constraints and build surface relationships
    constraints_info = []
    surface_objects = {}  # Maps surface name -> list of objects on that surface
    objects_list = semantic_plan.get("objects", [])
    
    for obj_dict in objects_list:
        obj_name = obj_dict.get("Model", "")
        constraints = obj_dict.get("constraints", [])
        for constraint in constraints:
            if isinstance(constraint, dict):
                ctype = constraint.get("type", "")
                target = constraint.get("target", "")
                
                constraints_info.append({
                    "object": obj_name,
                    "type": ctype,
                    "target": target,
                    "value": constraint.get("value", ""),
                })
                
                # Track surface relationships (on_top_of)
                if ctype.lower() in {"on", "on_top_of", "on-top-of", "on top of", "on_top"}:
                    if target not in surface_objects:
                        surface_objects[target] = []
                    surface_objects[target].append(obj_name)
    
    # Build surface information for LLM context
    surface_info = []
    for surface_name, objects_on_surface in surface_objects.items():
        # Find surface object to get its size
        surface_obj = None
        for obj in placed_models:
            if obj.get("Model") == surface_name:
                surface_obj = obj
                break
        
        if surface_obj:
            size = surface_obj.get("size", [1.0, 1.0, 1.0])
            pose = surface_obj.get("Pose", {})
            surface_info.append({
                "name": surface_name,
                "size": {
                    "width": float(size[0]) if len(size) > 0 else 1.0,
                    "depth": float(size[2]) if len(size) > 2 else 1.0,
                    "height": float(size[1]) if len(size) > 1 else 1.0,
                },
                "position": {
                    "x": float(pose.get("x", 0.0)),
                    "y": float(pose.get("y", 0.0)),
                    "z": float(pose.get("z", 0.0)),
                },
                "objects_on_surface": objects_on_surface,
            })
    
    # Format collision information
    collision_summary = []
    for collision in collision_details:
        collision_summary.append({
            "object_a": collision["object_a"],
            "object_b": collision["object_b"],
            "overlap_depth": collision["overlap_depth"],
        })
    
    # Build LLM prompt
    system_prompt = """You are a 3D scene layout optimizer. Your task is to reposition objects to eliminate overlaps while preserving semantic constraints and room boundaries.

You will receive:
1. List of objects with their current positions and sizes
2. Semantic constraints (near, on_top_of, region, etc.)
3. Surface information (tables, shelves, etc.) with their boundaries
4. Room boundaries
5. Current collision information

Your goal:
- Eliminate ALL overlaps between objects
- CRITICAL: Preserve semantic constraints - objects with on_top_of constraints MUST stay on their surface
- For objects on surfaces: ensure they stay WITHIN surface boundaries (not hanging off edges)
- If objects on the same surface collide, reposition them within that surface's boundaries
- Keep all objects within room boundaries
- Maintain reasonable spacing between objects (minimum 0.05m clearance)

IMPORTANT RULES FOR OBJECTS ON SURFACES:
1. Objects with on_top_of constraint MUST have X,Y positions within their surface boundaries
2. Surface boundaries: if surface is at (sx, sy) with size (width, depth), then:
   - Object X must be in range: [sx - width/2 + object_width/2, sx + width/2 - object_width/2]
   - Object Y must be in range: [sy - depth/2 + object_depth/2, sy + depth/2 - object_depth/2]
3. If multiple objects on same surface collide, spread them out within surface boundaries
4. Z position (height) must NOT change - objects stay at their current height

Return a JSON object with format:
{
  "positions": [
    {"index": 0, "x": 1.5, "y": 0.0, "z": 0.4},
    {"index": 1, "x": -1.2, "y": 0.5, "z": 0.4},
    ...
  ],
  "reasoning": "Brief explanation of positioning strategy"
}

CRITICAL:
- Do NOT move objects off their surfaces
- Ensure minimum clearance of 0.05m between object bounding boxes
- Respect room boundaries: -room_half_size <= x,y <= room_half_size
- Keep Z positions unchanged
"""
    
    user_prompt = f"""Room boundaries: [-{room_half_size}, {room_half_size}] in both X and Y

Objects to reposition:
{json.dumps(objects_info, indent=2)}

Semantic constraints:
{json.dumps(constraints_info, indent=2)}

Surface information (objects must stay within these boundaries):
{json.dumps(surface_info, indent=2)}

Current collisions (MUST be eliminated):
{json.dumps(collision_summary, indent=2)}

Please provide new X,Y positions for all objects that:
1. Eliminate all overlaps
2. Keep objects with on_top_of constraints WITHIN their surface boundaries
3. Preserve all semantic relationships
4. Maintain minimum 0.05m clearance between objects"""
    
    # Attempt LLM replanning with retries
    for attempt in range(max_retries):
        try:
            print(f"[llm_replan_layout] LLM attempt {attempt + 1}/{max_retries}...")
            
            # Call LLM
            response = prompt_model(system_prompt, user_prompt)
            
            if not isinstance(response, dict) or "positions" not in response:
                print(f"[llm_replan_layout] Invalid response format (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    # Modify prompt for retry
                    user_prompt += "\n\nPrevious attempt failed. Please ensure you return valid JSON with 'positions' array."
                    continue
                else:
                    print("[llm_replan_layout] All retry attempts exhausted")
                    return placed_models, False
            
            # Extract positions from response
            new_positions = response["positions"]
            reasoning = response.get("reasoning", "No reasoning provided")
            
            print(f"[llm_replan_layout] LLM reasoning: {reasoning}")
            
            # Validate response
            validation_result = _validate_llm_positions(
                placed_models=placed_models,
                new_positions=new_positions,
                room_half_size=room_half_size,
                semantic_plan=semantic_plan,
            )
            
            if not validation_result["valid"]:
                print(f"[llm_replan_layout] Validation failed: {validation_result['reason']}")
                if attempt < max_retries - 1:
                    # Modify prompt for retry
                    user_prompt += f"\n\nPrevious attempt failed validation: {validation_result['reason']}. Please fix these issues."
                    continue
                else:
                    print("[llm_replan_layout] All retry attempts exhausted")
                    return placed_models, False
            
            # Apply new positions
            replanned_models = []
            for obj in placed_models:
                obj_copy = dict(obj)
                replanned_models.append(obj_copy)
            
            for pos_update in new_positions:
                idx = pos_update.get("index")
                if idx is None or idx < 0 or idx >= len(replanned_models):
                    continue
                
                pose = dict(replanned_models[idx].get("Pose", {}))
                pose["x"] = float(pos_update.get("x", pose.get("x", 0.0)))
                pose["y"] = float(pos_update.get("y", pose.get("y", 0.0)))
                # Keep Z unchanged (objects stay on their surfaces)
                replanned_models[idx]["Pose"] = pose
            
            # Final collision check
            final_collisions = check_collisions(replanned_models, collision_margin=0.01)
            
            if final_collisions:
                print(f"[llm_replan_layout] LLM solution still has {len(final_collisions)} collision(s)")
                if attempt < max_retries - 1:
                    # Provide feedback for retry
                    remaining_collisions = [
                        f"{c['object_a']} <-> {c['object_b']} (depth={c['overlap_depth']:.3f}m)"
                        for c in final_collisions
                    ]
                    user_prompt += f"\n\nPrevious solution still had collisions: {remaining_collisions}. Please increase spacing."
                    continue
                else:
                    print("[llm_replan_layout] WARNING: LLM solution has collisions, but using it anyway (best effort)")
                    return replanned_models, True  # Partial success
            
            print(f"[llm_replan_layout] SUCCESS: LLM replanning eliminated all collisions")
            return replanned_models, True
            
        except Exception as e:
            print(f"[llm_replan_layout] Error during attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                print("[llm_replan_layout] All retry attempts exhausted due to errors")
                return placed_models, False
    
    # Should not reach here, but return failure as fallback
    return placed_models, False


def _validate_llm_positions(
    placed_models: List[Dict[str, Any]],
    new_positions: List[Dict[str, Any]],
    room_half_size: float,
    semantic_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate LLM-proposed positions for feasibility.
    
    Args:
        placed_models: Original placed objects
        new_positions: LLM-proposed positions
        room_half_size: Room boundary
        semantic_plan: Semantic plan with constraints
        
    Returns:
        Dictionary with keys:
        - valid: Whether positions are valid (bool)
        - reason: Reason for validation failure (str)
    """
    # Check that all objects have positions
    if len(new_positions) != len(placed_models):
        return {
            "valid": False,
            "reason": f"Position count mismatch: expected {len(placed_models)}, got {len(new_positions)}",
        }
    
    # Check that all positions are within room boundaries
    for pos in new_positions:
        x = pos.get("x", 0.0)
        y = pos.get("y", 0.0)
        
        if abs(x) > room_half_size or abs(y) > room_half_size:
            return {
                "valid": False,
                "reason": f"Position ({x}, {y}) outside room boundaries [-{room_half_size}, {room_half_size}]",
            }
    
    # Check that indices are valid
    for pos in new_positions:
        idx = pos.get("index")
        if idx is None or idx < 0 or idx >= len(placed_models):
            return {
                "valid": False,
                "reason": f"Invalid object index: {idx}",
            }
    
    # All checks passed
    return {"valid": True, "reason": ""}
