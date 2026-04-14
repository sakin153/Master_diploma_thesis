from typing import Any, Dict, List, Sequence


def solve_wall_placements(
    *,
    placed_models: Sequence[Dict[str, Any]],
    semantic_plan: Dict[str, Any],
    room_half_size: float = 5.0,
) -> List[Dict[str, Any]]:
    plan_rows_by_model = {}
    objects = semantic_plan.get("objects", []) if isinstance(semantic_plan, dict) else []
    if isinstance(objects, list):
        for o in objects:
            if isinstance(o, dict) and o.get("Model"):
                key = str(o["Model"])
                plan_rows_by_model.setdefault(key, []).append(o.get("constraints") or [])

    out: List[Dict[str, Any]] = []
    rr_idx: Dict[str, int] = {}
    for m in placed_models:
        item = dict(m)
        name = str(item.get("Model") or item.get("name") or "")
        rows = plan_rows_by_model.get(name, [])
        if rows:
            i = rr_idx.get(name, 0) % len(rows)
            constraints = rows[i]
            rr_idx[name] = rr_idx.get(name, 0) + 1
        else:
            constraints = []
        wants_wall = any(str(c.get("type", "")).lower() == "edge" for c in constraints)

        # Heuristic: potential wall-mounted items.
        lname = name.lower()
        likely_wall_item = any(k in lname for k in ["lamp", "picture", "tv", "frame", "shelf"])

        if wants_wall or likely_wall_item:
            pose = dict(item.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.5})
            pose["x"] = room_half_size - 0.25
            yv = float(pose.get("y", 0.0))
            pose["y"] = max(-room_half_size + 0.25, min(room_half_size - 0.25, yv))
            # Lift wall items from floor if needed.
            pose["z"] = max(float(pose.get("z", 0.5)), 1.1)
            item["Pose"] = pose
        out.append(item)

    return out
