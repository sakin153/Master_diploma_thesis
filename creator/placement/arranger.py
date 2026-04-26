"""Arrangement patterns for groups of identical objects.

Best practices from:
- SceneSmith: specialized tools per arrangement type (grid, pile, stack)
- ImperativeScene: procedural layout with symmetry/tiling

Handles:
- Grid layout (classrooms, warehouses)
- Row layout (chairs along a table, theater rows)
- Cluster layout (chairs around a table)
- Pair layout (desk + chair as a unit)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from creator.placement.geometry import (
    OBB,
    Vec2,
    model_to_obb,
    obb_inside_polygon,
    obb_inside_room,
    obb_overlap,
)


# ---------------------------------------------------------------------------
# Arrangement types
# ---------------------------------------------------------------------------

@dataclass
class ArrangementGroup:
    """A group of same-type objects to be arranged together."""
    model_name: str
    models: List[Dict[str, Any]]
    arrangement: str = "auto"   # auto | grid | row | cluster | beam
    paired_with: str = ""       # name of companion object (e.g. desk→chair)


def detect_arrangement_groups(
    models: Sequence[Dict[str, Any]],
    semantic_plan: Dict[str, Any],
    *,
    grid_threshold: int = 4,
) -> List[ArrangementGroup]:
    """Detect which objects should use grid/row vs beam search.

    Objects with N >= grid_threshold identical instances → grid arrangement.
    Objects with "near" constraint to a grid object → paired arrangement.
    """
    # Count instances per model type
    counts: Dict[str, int] = {}
    for m in models:
        name = str(m.get("Model") or m.get("name") or "")
        counts[name] = counts.get(name, 0) + 1

    # Build near-pairs from semantic plan
    near_pairs: Dict[str, str] = {}  # follower → anchor
    def _role(name: str) -> str:
        lname = name.lower()
        if any(k in lname for k in ["desk", "table", "bench"]):
            return "desk"
        if any(k in lname for k in ["chair", "stool", "seat"]):
            return "chair"
        return "other"

    objects = semantic_plan.get("objects", []) if isinstance(semantic_plan, dict) else []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("Model") or "")
        for c in (obj.get("constraints") or []):
            if not isinstance(c, dict):
                continue
            if str(c.get("type", "")).lower() == "near":
                target = str(c.get("target", ""))
                if target and counts.get(target, 0) >= grid_threshold:
                    src_role = _role(name)
                    tgt_role = _role(target)

                    # Classroom specialization:
                    # if desks are constrained near chairs, treat chairs as
                    # followers of desks so we can place desk+chair pairs.
                    if (
                        src_role == "desk"
                        and tgt_role == "chair"
                        and counts.get(name, 0) >= grid_threshold
                    ):
                        near_pairs[target] = name
                    else:
                        near_pairs[name] = target

    # Extra classroom guard: if both desk-like and chair-like groups are large,
    # enforce chair→desk pairing even if the LLM constraints are noisy.
    desk_candidates = [n for n, c in counts.items() if c >= grid_threshold and _role(n) == "desk"]
    chair_candidates = [n for n, c in counts.items() if c >= grid_threshold and _role(n) == "chair"]
    if desk_candidates and chair_candidates:
        anchor = max(desk_candidates, key=lambda n: counts.get(n, 0))
        follower = max(chair_candidates, key=lambda n: counts.get(n, 0))
        near_pairs[follower] = anchor

    # Group models
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for m in models:
        name = str(m.get("Model") or m.get("name") or "")
        grouped.setdefault(name, []).append(m)

    groups: List[ArrangementGroup] = []
    handled = set()

    for name, mlist in grouped.items():
        if name in handled:
            continue
        n = len(mlist)
        # Pair has priority over standalone grid so follower groups are
        # consumed by anchor+follower classroom arrangement.
        if name in near_pairs and counts.get(near_pairs[name], 0) >= grid_threshold:
            groups.append(ArrangementGroup(
                model_name=name,
                models=mlist,
                arrangement="pair",
                paired_with=near_pairs[name],
            ))
            handled.add(name)
        elif n >= grid_threshold:
            groups.append(ArrangementGroup(
                model_name=name,
                models=mlist,
                arrangement="grid",
                paired_with="",
            ))
            handled.add(name)
        else:
            groups.append(ArrangementGroup(
                model_name=name,
                models=mlist,
                arrangement="beam",
                paired_with="",
            ))
            handled.add(name)

    return groups


# ---------------------------------------------------------------------------
# Grid placement
# ---------------------------------------------------------------------------

@dataclass
class GridParams:
    origin: Vec2          # bottom-left corner of grid
    cols: int
    rows: int
    spacing_x: float      # center-to-center X
    spacing_y: float      # center-to-center Y
    yaw_deg: float = 0.0
    grid_yaw_deg: float = 0.0  # rotation of entire grid


def compute_grid_params(
    n_objects: int,
    obj_size: Sequence[float],  # [sx, sy, sz]
    room_half_size: float,
    *,
    gap_x: float = 0.3,
    gap_y: float = 0.5,
    margin: float = 0.5,
    prefer_landscape: bool = True,
) -> GridParams:
    """Compute grid layout parameters for N objects of given size.

    Chooses cols/rows to make the grid roughly square within the room.
    Uses gap_x/gap_y as spacing between objects.
    """
    # size[0]=mesh X=scene width, size[2]=mesh Z=scene depth (after 90° X rotation)
    sx = max(0.1, float(obj_size[0]))
    sy = max(0.1, float(obj_size[2]) if len(obj_size) > 2 else float(obj_size[1]))

    spacing_x = sx + gap_x
    spacing_y = sy + gap_y

    usable = (room_half_size - margin) * 2.0
    max_cols = max(1, int(usable / spacing_x))
    max_rows = max(1, int(usable / spacing_y))

    # Try to make grid as square as possible
    best_cols, best_rows = 1, n_objects
    best_ratio = float("inf")
    for c in range(1, min(max_cols, n_objects) + 1):
        r = math.ceil(n_objects / c)
        if r > max_rows:
            continue
        ratio = abs(math.log(max(1, (c * spacing_x) / max(0.01, r * spacing_y))))
        if ratio < best_ratio:
            best_ratio = ratio
            best_cols, best_rows = c, r

    # Center the grid in the room
    total_w = best_cols * spacing_x - gap_x
    total_h = best_rows * spacing_y - gap_y
    origin_x = -total_w / 2.0
    origin_y = -total_h / 2.0

    return GridParams(
        origin=Vec2(origin_x, origin_y),
        cols=best_cols,
        rows=best_rows,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        yaw_deg=0.0,
    )


def apply_grid_arrangement(
    models: List[Dict[str, Any]],
    *,
    room_half_size: float = 5.0,
    room_polygon: Optional[List] = None,
    gap_x: float = 0.3,
    gap_y: float = 0.5,
    margin: float = 0.5,
    yaw_deg: float = 0.0,
) -> List[Dict[str, Any]]:
    """Place models in a grid pattern. Returns updated models with Pose set.

    Inspired by SceneSmith's arrangement tools and classroom layouts.
    """
    if not models:
        return models

    size = models[0].get("size", [0.5, 0.5, 1.0])
    # size[1] = mesh Y = scene Z (height) after euler="90 0 yaw" rotation
    sz = max(0.01, float(size[1])) / 2.0
    params = compute_grid_params(
        len(models), size, room_half_size,
        gap_x=gap_x, gap_y=gap_y, margin=margin,
    )

    result = []
    for idx, m in enumerate(models):
        col = idx % params.cols
        row = idx // params.cols
        cx = params.origin.x + col * params.spacing_x + float(size[0]) / 2.0
        # size[2] = scene Y depth (size[1] is height, not floor depth)
        cy = params.origin.y + row * params.spacing_y + (float(size[2]) if len(size) > 2 else float(size[1])) / 2.0

        item = dict(m)
        item["Pose"] = {"x": cx, "y": cy, "z": sz}
        item["yaw_deg"] = yaw_deg
        result.append(item)

    return result


def apply_paired_grid_arrangement(
    anchor_models: List[Dict[str, Any]],
    follower_models: List[Dict[str, Any]],
    *,
    room_half_size: float = 5.0,
    gap_x: float = 0.3,
    gap_y: float = 0.5,
    pair_offset_y: float = 0.0,  # follower offset relative to anchor
    anchor_yaw_deg: float = 0.0,
    follower_yaw_deg: float = 180.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Place anchors in a grid, followers directly behind each anchor.

    Classic classroom layout: desks in grid, chairs behind each desk.
    """
    if not anchor_models:
        return anchor_models, follower_models

    anchor_size = anchor_models[0].get("size", [0.6, 0.5, 0.75])
    follower_size = follower_models[0].get("size", [0.45, 0.45, 0.9]) if follower_models else [0.45, 0.45, 0.9]

    # Floor footprint: size[0]=width, size[2]=depth (mesh Z = scene Y after rotation)
    a_depth = float(anchor_size[2]) if len(anchor_size) > 2 else float(anchor_size[1])
    f_depth = float(follower_size[2]) if len(follower_size) > 2 else float(follower_size[1])
    pair_depth = a_depth + f_depth + 0.2
    pair_spacing_x = max(float(anchor_size[0]), float(follower_size[0])) + gap_x
    pair_spacing_y = pair_depth + gap_y

    n_pairs = max(len(anchor_models), len(follower_models))
    usable = (room_half_size - 0.5) * 2.0
    max_cols = max(1, int(usable / pair_spacing_x))
    best_cols = min(max_cols, math.ceil(math.sqrt(n_pairs)))
    best_rows = math.ceil(n_pairs / best_cols)

    total_w = best_cols * pair_spacing_x - gap_x
    total_h = best_rows * pair_spacing_y - gap_y
    origin_x = -total_w / 2.0
    origin_y = -total_h / 2.0

    updated_anchors = []
    updated_followers = []

    for idx in range(n_pairs):
        col = idx % best_cols
        row = idx // best_cols

        a_d = float(anchor_size[2]) if len(anchor_size) > 2 else float(anchor_size[0])
        f_d = float(follower_size[2]) if len(follower_size) > 2 else float(follower_size[0])
        ax = origin_x + col * pair_spacing_x + float(anchor_size[0]) / 2.0
        ay = origin_y + row * pair_spacing_y + a_d / 2.0
        az = float(anchor_size[1]) / 2.0  # size[1]=height

        fy = ay + a_d / 2.0 + 0.1 + f_d / 2.0
        fz = float(follower_size[1]) / 2.0  # size[1]=height

        if idx < len(anchor_models):
            a = dict(anchor_models[idx])
            a["Pose"] = {"x": ax, "y": ay, "z": az}
            a["yaw_deg"] = anchor_yaw_deg
            updated_anchors.append(a)

        if idx < len(follower_models):
            f = dict(follower_models[idx])
            f["Pose"] = {"x": ax, "y": fy + pair_offset_y, "z": fz}
            f["yaw_deg"] = follower_yaw_deg
            updated_followers.append(f)

    return updated_anchors, updated_followers


# ---------------------------------------------------------------------------
# Row arrangement (theater, banquet)
# ---------------------------------------------------------------------------

def apply_row_arrangement(
    models: List[Dict[str, Any]],
    *,
    room_half_size: float = 5.0,
    gap: float = 0.15,
    row_direction: str = "x",  # "x" or "y"
    center_y: float = 0.0,
    yaw_deg: float = 0.0,
) -> List[Dict[str, Any]]:
    """Place models in a single row."""
    if not models:
        return models

    size = models[0].get("size", [0.5, 0.5, 1.0])
    # Floor-plane: row goes along X or Z (mesh-Z = scene-Y), not size[1]
    dim = float(size[0]) if row_direction == "x" else float(size[2])
    sz = float(size[1]) / 2.0  # height = mesh Y
    spacing = dim + gap
    total = len(models) * spacing - gap
    start = -total / 2.0 + dim / 2.0

    result = []
    for idx, m in enumerate(models):
        offset = start + idx * spacing
        item = dict(m)
        if row_direction == "x":
            item["Pose"] = {"x": offset, "y": center_y, "z": sz}
        else:
            item["Pose"] = {"x": center_y, "y": offset, "z": sz}
        item["yaw_deg"] = yaw_deg
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# Integration: apply arrangements before beam search
# ---------------------------------------------------------------------------

def apply_arrangements(
    full_placed_models: List[Dict[str, Any]],
    semantic_plan: Dict[str, Any],
    *,
    room_half_size: float = 5.0,
    grid_threshold: int = 4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply grid/row arrangements and return:
       - arranged: models with Pose already set (skip beam search)
       - remaining: models that still need beam search placement

    This is called before floor_solver's beam search.
    """
    room = semantic_plan.get("room", {}) if isinstance(semantic_plan, dict) else {}
    room_polygon = room.get("polygon") if isinstance(room, dict) else None

    groups = detect_arrangement_groups(
        full_placed_models, semantic_plan, grid_threshold=grid_threshold,
    )

    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for m in full_placed_models:
        name = str(m.get("Model") or m.get("name") or "")
        by_name.setdefault(name, []).append(m)

    arranged: List[Dict[str, Any]] = []
    arranged_names: set = set()
    pair_anchors_done: set = set()

    # First pass: handle grid groups and their pairs
    for group in groups:
        if group.arrangement != "grid":
            continue
        if group.model_name in arranged_names:
            continue

        anchor_models = by_name.get(group.model_name, [])

        # Check if there's a paired follower
        follower_name = _find_follower(group.model_name, groups)
        follower_models = by_name.get(follower_name, []) if follower_name else []

        if follower_models and len(follower_models) >= grid_threshold // 2:
            # Paired grid (desk+chair)
            anchor_yaw = _default_yaw(group.model_name)
            follower_name_l = follower_name.lower()
            if any(k in follower_name_l for k in ["chair", "stool", "seat"]):
                # Keep same yaw as desk anchor to avoid visually flipped rows
                # when model forward-axis conventions differ across assets.
                follower_yaw = anchor_yaw
            else:
                follower_yaw = (anchor_yaw + 180.0) % 360.0

            updated_a, updated_f = apply_paired_grid_arrangement(
                anchor_models, follower_models,
                room_half_size=room_half_size,
                gap_x=0.45,
                gap_y=0.75,
                anchor_yaw_deg=anchor_yaw,
                follower_yaw_deg=follower_yaw,
            )
            arranged.extend(updated_a)
            arranged.extend(updated_f)
            arranged_names.add(group.model_name)
            arranged_names.add(follower_name)
        else:
            # Solo grid
            yaw = _default_yaw(group.model_name)
            updated = apply_grid_arrangement(
                anchor_models,
                room_half_size=room_half_size,
                yaw_deg=yaw,
            )
            arranged.extend(updated)
            arranged_names.add(group.model_name)

    # Second pass: handle pair groups whose anchor was not gridded
    for group in groups:
        if group.arrangement != "pair" or group.model_name in arranged_names:
            continue
        # Will be handled by beam search with near-target sampling
        pass

    # Remaining: beam search
    remaining = [
        m for m in full_placed_models
        if str(m.get("Model") or m.get("name") or "") not in arranged_names
    ]

    return arranged, remaining


def _find_follower(anchor_name: str, groups: List[ArrangementGroup]) -> str:
    """Find a group whose paired_with points to anchor_name."""
    for g in groups:
        if g.paired_with == anchor_name:
            return g.model_name
    return ""


def _default_yaw(name: str) -> float:
    """Default facing direction for common object types."""
    lname = name.lower()
    if any(k in lname for k in ["desk", "table", "bench"]):
        return 0.0
    if any(k in lname for k in ["chair", "stool", "seat"]):
        return 0.0
    return 0.0
