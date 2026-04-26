"""Small object placement on support surfaces.

Improvements inspired by SceneSmith:
- Support surface validation with area checking
- Multiple spawn positions on receptacle surface (not just center)
- Collision checking between small objects on the same surface
- Stacking support for explicit "on" constraints
"""

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


_CONTAINER_HINTS = (
    "box",
    "crate",
    "container",
    "drawer",
    "basket",
    "bin",
    "ящик",
)


def _volume(size: Sequence[float]) -> float:
    return float(size[0]) * float(size[1]) * float(size[2])


def _get_size(model: Dict[str, Any]) -> List[float]:
    size = model.get("size")
    if size is None or len(size) < 3:
        return [0.1, 0.1, 0.1]
    return [max(0.01, float(size[0])), max(0.01, float(size[1])), max(0.01, float(size[2]))]


def _footprint_area(size: Sequence[float]) -> float:
    # size[0]=width, size[2]=depth (scene XY footprint); size[1]=height
    return float(size[0]) * (float(size[2]) if len(size) > 2 else float(size[0]))


def _instance_constraints(
    placed_models: Sequence[Dict[str, Any]],
    semantic_plan: Dict[str, Any],
) -> List[List[Dict[str, Any]]]:
    objects = semantic_plan.get("objects", []) if isinstance(semantic_plan, dict) else []
    by_name: Dict[str, List[List[Dict[str, Any]]]] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("Model") or obj.get("name") or "")
        if not name:
            continue
        constraints = obj.get("constraints") if isinstance(obj.get("constraints"), list) else []
        by_name.setdefault(name, []).append(
            [c for c in constraints if isinstance(c, dict)]
        )

    rr_idx: Dict[str, int] = {}
    rows: List[List[Dict[str, Any]]] = []
    for model in placed_models:
        name = str(model.get("Model") or model.get("name") or "")
        variants = by_name.get(name, [])
        if not variants:
            rows.append([])
            continue
        i = rr_idx.get(name, 0) % len(variants)
        rr_idx[name] = rr_idx.get(name, 0) + 1
        rows.append(variants[i])
    return rows


def _find_target_by_name(
    *,
    item_index: int,
    target_name: str,
    placed_models: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    item_pose = placed_models[item_index].get("Pose") or {"x": 0.0, "y": 0.0}
    ix = float(item_pose.get("x", 0.0))
    iy = float(item_pose.get("y", 0.0))

    best = None
    best_d = float("inf")
    for idx, candidate in enumerate(placed_models):
        if idx == item_index:
            continue
        if str(candidate.get("Model") or candidate.get("name") or "") != target_name:
            continue
        cp = candidate.get("Pose") or {"x": 0.0, "y": 0.0}
        cx = float(cp.get("x", 0.0))
        cy = float(cp.get("y", 0.0))
        d = (ix - cx) ** 2 + (iy - cy) ** 2
        if d < best_d:
            best_d = d
            best = candidate
    return best


def _spawn_positions_on_surface(
    receptacle: Dict[str, Any],
    item_size: Sequence[float],
    n_samples: int = 8,
    rng: Optional[random.Random] = None,
) -> List[Tuple[float, float]]:
    """Generate candidate XY positions on the top surface of a receptacle.

    Inspired by SceneSmith's support surface extraction — we sample within
    the footprint of the receptacle, inset by the item's half-size.
    """
    if rng is None:
        rng = random.Random(42)

    rp = receptacle.get("Pose") or {"x": 0.0, "y": 0.0}
    rx = float(rp.get("x", 0.0))
    ry = float(rp.get("y", 0.0))
    rsize = _get_size(receptacle)

    # Usable surface = receptacle footprint minus item half-extents
    # After euler="90 0 yaw": size[0]=scene X (width), size[2]=scene Y (depth)
    margin_x = float(item_size[0]) / 2.0 + 0.02
    margin_y = (float(item_size[2]) / 2.0 + 0.02) if len(item_size) > 2 else (float(item_size[0]) / 2.0 + 0.02)
    usable_hx = max(0.0, rsize[0] / 2.0 - margin_x)
    usable_hy = max(0.0, (rsize[2] / 2.0 if len(rsize) > 2 else rsize[0] / 2.0) - margin_y)

    positions = [(rx, ry)]  # center always included

    # Systematic grid candidates — use (grid_steps+1)² including boundary points
    # so the search always reaches the extreme edges of the usable surface.
    grid_steps = max(3, int(n_samples ** 0.5))
    if usable_hx > 0.01 or usable_hy > 0.01:
        for gi in range(grid_steps + 1):
            for gj in range(grid_steps + 1):
                t = gi / grid_steps       # 0..1 inclusive (boundary reached)
                s = gj / grid_steps
                ox = usable_hx * (2 * t - 1) if usable_hx > 0.01 else 0.0
                oy = usable_hy * (2 * s - 1) if usable_hy > 0.01 else 0.0
                positions.append((rx + ox, ry + oy))

    # Random candidates for diversity (fills gaps between grid points)
    remaining = max(16, n_samples - len(positions))
    for _ in range(remaining):
        ox = rng.uniform(-usable_hx, usable_hx) if usable_hx > 0.01 else 0.0
        oy = rng.uniform(-usable_hy, usable_hy) if usable_hy > 0.01 else 0.0
        positions.append((rx + ox, ry + oy))

    return positions


def _xy_overlap(
    x1: float, y1: float, s1: Sequence[float],
    x2: float, y2: float, s2: Sequence[float],
    margin: float = 0.02,
) -> bool:
    """Check if two items overlap in XY (floor plane).

    After euler="90 0 yaw": size[0]=scene X, size[2]=scene Y (depth), size[1]=height.
    """
    hx1 = float(s1[0]) / 2.0
    hy1 = float(s1[2]) / 2.0 if len(s1) > 2 else float(s1[0]) / 2.0
    hx2 = float(s2[0]) / 2.0
    hy2 = float(s2[2]) / 2.0 if len(s2) > 2 else float(s2[0]) / 2.0
    sep_x = (x1 + hx1 + margin < x2 - hx2) or (x2 + hx2 + margin < x1 - hx1)
    sep_y = (y1 + hy1 + margin < y2 - hy2) or (y2 + hy2 + margin < y1 - hy1)
    return not (sep_x or sep_y)


_ON_CONSTRAINT_TYPES = frozenset({
    "on", "on_top_of", "on-top-of", "on top of", "on_top",
})


def _extract_on_target(constraints: Sequence[Dict[str, Any]]) -> str:
    for c in constraints:
        if not isinstance(c, dict):
            continue
        ctype = str(c.get("type", "")).lower()
        if ctype in _ON_CONSTRAINT_TYPES:
            target = str(c.get("target", "")).strip()
            if target:
                return target
    return ""


def _stacking_depth(
    name: str,
    on_target: str,
    name_to_target: Dict[str, str],
    seen: Optional[set] = None,
) -> int:
    """Recursive depth: floor=0, on-floor-object=1, on-on-floor-object=2, ..."""
    if not on_target or on_target == name:
        return 0
    seen = seen or set()
    if on_target in seen:
        return 0  # cycle guard
    seen.add(on_target)
    parent_target = name_to_target.get(on_target, "")
    return 1 + _stacking_depth(on_target, parent_target, name_to_target, seen)


def solve_small_object_placements(
    *,
    placed_models: Sequence[Dict[str, Any]],
    semantic_plan: Optional[Dict[str, Any]] = None,
    small_threshold_volume: float = 0.06,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Place objects on support surfaces with collision avoidance.

    Two kinds of objects are handled here:
      - **Explicit-on**: any object (regardless of size) whose plan contains
        `on_top_of(target)`. Crates on a table are large but stack-bound.
      - **Implicit-small**: objects with volume < `small_threshold_volume`
        (books, cups, apples) that have no explicit support — auto-attach
        to the nearest semantically appropriate receptacle.

    Processing order is topological by stacking depth so the underlying
    receptacle is positioned (e.g. crate moved onto table) BEFORE the items
    that rest on it (e.g. apple in crate). This is required for correct Z.

    For multiple identical receptacle instances of the same model name (e.g.
    4 crates), items with the same `on_top_of(target_name)` constraint are
    distributed round-robin across instances so apples spread evenly across
    crates instead of piling onto the closest one.
    """
    rng = random.Random(seed)
    out = [dict(m) for m in placed_models]
    constraint_rows = _instance_constraints(out, semantic_plan or {"objects": []})

    # Build per-model-name target lookup for stacking-depth computation.
    name_to_target: Dict[str, str] = {}
    for i, m in enumerate(out):
        nm = str(m.get("Model") or m.get("name") or "")
        if not nm:
            continue
        cs = constraint_rows[i] if i < len(constraint_rows) else []
        tgt = _extract_on_target(cs)
        if tgt and nm not in name_to_target:
            name_to_target[nm] = tgt

    # Eligibility & ordering: every item with explicit on-target is eligible
    # regardless of size; small items without explicit on-target also eligible.
    eligible: List[Tuple[int, int, str]] = []  # (depth, original_index, on_target)
    for i, m in enumerate(out):
        nm = str(m.get("Model") or m.get("name") or "")
        size = _get_size(m)
        cs = constraint_rows[i] if i < len(constraint_rows) else []
        on_target = _extract_on_target(cs)
        if on_target:
            depth = _stacking_depth(nm, on_target, name_to_target)
            eligible.append((depth, i, on_target))
        elif _volume(size) < small_threshold_volume:
            eligible.append((1, i, ""))

    # Process from shallowest to deepest stacking — crates first, apples after.
    # Stable secondary key keeps original order within the same depth.
    eligible.sort(key=lambda t: (t[0], t[1]))

    receptacles = [
        m for m in out if _volume(_get_size(m)) >= small_threshold_volume
    ]
    surface_items: Dict[int, List[Tuple[float, float, List[float]]]] = {}

    # Round-robin cursor per (target_name) — distribute multiple items across
    # multiple identical-name receptacles evenly.
    rr_cursor: Dict[str, int] = {}

    for _depth, i, on_target in eligible:
        m = out[i]
        size = _get_size(m)

        if on_target:
            # Find ALL receptacle instances matching the target name.
            # Distribute round-robin so 5 apples × 4 crates → 1-2 per crate.
            candidates = [
                idx for idx, c in enumerate(out)
                if idx != i
                and str(c.get("Model") or c.get("name") or "") == on_target
            ]
            if not candidates:
                # Try fuzzy fallback by name token
                continue
            cursor = rr_cursor.get(on_target, 0)
            chosen_idx = candidates[cursor % len(candidates)]
            rr_cursor[on_target] = cursor + 1
            receptacle = out[chosen_idx]
            _place_on_receptacle(out, i, receptacle, size, rng, surface_items)
            continue

        # Implicit small object: pick best support semantically + by distance.
        receptacle = _find_best_receptacle(m, receptacles)
        if receptacle is None:
            continue
        _place_on_receptacle(out, i, receptacle, size, rng, surface_items)

    return out


def _place_on_receptacle(
    out: List[Dict[str, Any]],
    item_idx: int,
    receptacle: Dict[str, Any],
    item_size: List[float],
    rng: random.Random,
    surface_items: Dict[int, List[Tuple[float, float, List[float]]]],
    room_half_size: float = 5.0,
) -> None:
    """Place an item on a receptacle surface, avoiding collisions with others."""
    rp = receptacle.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.0}
    rsize = _get_size(receptacle)

    z_base = float(rp.get("z", 0.0))
    # size[1] = mesh Y = scene Z (height) after euler="90 0 yaw" rotation
    z_receptacle = float(rsize[1]) / 2.0
    z_item = float(item_size[1]) / 2.0
    if _is_container_like(receptacle):
        # Spawn above opening so physics can settle item into the container.
        drop_start = min(0.12, max(0.05, z_receptacle * 0.5))
        target_z = z_base + z_receptacle + z_item + drop_start
    else:
        target_z = z_base + z_receptacle + z_item + 0.01

    # Find receptacle id for tracking
    recep_id = id(receptacle)
    existing = surface_items.get(recep_id, [])

    # Try candidate positions (grid + random for full surface coverage)
    candidates = _spawn_positions_on_surface(receptacle, item_size, n_samples=64, rng=rng)
    best_pos = candidates[0]  # default: center
    best_score = -float("inf")

    hx_item = float(item_size[0]) / 2.0
    # size[2] = scene Y depth (not size[1]=height) for XY clamping
    hy_item = float(item_size[2]) / 2.0 if len(item_size) > 2 else float(item_size[0]) / 2.0

    rx_center = float(rp.get("x", 0.0))
    ry_center = float(rp.get("y", 0.0))
    has_existing = bool(existing)

    for cx, cy in candidates:
        # Clamp to room bounds
        cx = max(-room_half_size + hx_item + 0.02, min(room_half_size - hx_item - 0.02, cx))
        cy = max(-room_half_size + hy_item + 0.02, min(room_half_size - hy_item - 0.02, cy))

        # Check collision with other items on this surface
        collision = False
        for ex, ey, esize in existing:
            if _xy_overlap(cx, cy, item_size, ex, ey, esize, margin=0.03):
                collision = True
                break

        if collision:
            continue

        if has_existing:
            # Score = maximize minimum distance to any existing item on this surface.
            # This spreads items evenly across the surface.
            min_d = min(
                math.sqrt((cx - ex) ** 2 + (cy - ey) ** 2)
                for ex, ey, _ in existing
            )
            score = min_d
        else:
            # No existing items: prefer center to anchor the first object.
            dist_to_center = math.sqrt((cx - rx_center) ** 2 + (cy - ry_center) ** 2)
            score = -dist_to_center

        if score > best_score:
            best_score = score
            best_pos = (cx, cy)

    pose = dict(out[item_idx].get("Pose") or {})
    pose["x"] = best_pos[0]
    pose["y"] = best_pos[1]
    pose["z"] = target_z
    out[item_idx]["Pose"] = pose

    # Track placement
    surface_items.setdefault(recep_id, []).append(
        (best_pos[0], best_pos[1], item_size)
    )


_SUPPORT_SURFACE_HINTS = (
    "table", "desk", "shelf", "bookshelf", "bookcase",
    "counter", "countertop", "nightstand", "dresser", "sideboard",
    "cabinet", "credenza", "buffet", "bar", "stand",
    "tray", "pedestal", "podium", "platform",
)
_NON_SUPPORT_HINTS = (
    "chair", "armchair", "stool", "sofa", "couch", "loveseat", "settee",
    "ottoman", "bed", "mattress", "lamp", "fan", "tv", "monitor",
    "screen", "fridge", "refrigerator", "wardrobe", "closet",
)


def _surface_score(receptacle: Dict[str, Any]) -> float:
    """Semantic appropriateness of putting a small object on this receptacle.
    Range roughly [-1, +1]; higher is better."""
    parts: List[str] = []
    name = str(receptacle.get("Model") or receptacle.get("name") or "")
    if name:
        parts.append(name)
    for key in ("categories", "tags"):
        raw = receptacle.get(key)
        if isinstance(raw, list):
            parts.extend(str(x) for x in raw if x)
    haystack = " ".join(parts).lower()

    score = 0.0
    if any(token in haystack for token in _SUPPORT_SURFACE_HINTS):
        score += 1.0
    if any(token in haystack for token in _NON_SUPPORT_HINTS):
        score -= 0.7
    if any(token in haystack for token in _CONTAINER_HINTS):
        # Containers explicitly want stuff inside them.
        score += 0.6
    return score


def _find_best_receptacle(
    item: Dict[str, Any],
    receptacles: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Pick the most appropriate support surface.

    Combines distance (closer is better) with semantic appropriateness:
    a table 2 m away beats a chair 0.5 m away.
    """
    if not receptacles:
        return None
    ip = item.get("Pose") or {"x": 0.0, "y": 0.0}
    ix, iy = float(ip.get("x", 0.0)), float(ip.get("y", 0.0))

    best = None
    best_score = float("-inf")
    for r in receptacles:
        rp = r.get("Pose") or {"x": 0.0, "y": 0.0}
        rx, ry = float(rp.get("x", 0.0)), float(rp.get("y", 0.0))
        d = math.sqrt((ix - rx) ** 2 + (iy - ry) ** 2)
        # Distance term decays with distance: ~1.0 at 0, ~0 at 4 m.
        dist_term = max(0.0, 1.0 - d / 4.0)
        sem_term = _surface_score(r)
        # Hard veto: never put items on clearly non-support objects unless
        # nothing else exists.
        score = sem_term * 1.5 + dist_term * 0.6
        if score > best_score:
            best_score = score
            best = r
    return best


def _is_container_like(item: Dict[str, Any]) -> bool:
    parts: List[str] = []
    name = str(item.get("Model") or item.get("name") or "")
    if name:
        parts.append(name)

    for key in ("categories", "tags"):
        raw = item.get(key)
        if isinstance(raw, list):
            parts.extend(str(x) for x in raw if x)

    haystack = " ".join(parts).lower()
    return any(token in haystack for token in _CONTAINER_HINTS)
