"""Place small or explicitly-stacked objects onto support surfaces.

Two kinds of items are processed here:
  - Explicit-on: any object with `on_top_of(target)` in its constraints,
    regardless of its size (e.g. a crate on a table).
  - Implicit-small: items whose volume is below `small_threshold_volume`
    and which lack any other anchor — these auto-attach to the nearest
    semantically appropriate receptacle.

For each receptacle we lay out every assigned item as a single grid that
fits inside the receptacle footprint, with cell pitch chosen from the
largest item in the assignment. Items of the same model name stay
contiguous in the grid so visual clusters match semantics ("apples on
table" → all apples cluster together).
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from creator.placement.targets import matching_indices, parse_instance_target


_CONTAINER_HINTS = (
    "box", "crate", "container", "drawer", "basket", "bin", "ящик",
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
_ON_CONSTRAINT_TYPES = frozenset({
    "on", "on_top_of", "on-top-of", "on top of", "on_top",
})


# ---------------------------------------------------------------------------
# Geometry helpers — scene XYZ after euler="90 0 yaw" rotation:
#   size[0] = scene-X (width), size[1] = scene-Z (height), size[2] = scene-Y (depth)
# ---------------------------------------------------------------------------

def _get_size(model: Dict[str, Any]) -> Tuple[float, float, float]:
    raw = model.get("size")
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return 0.1, 0.1, 0.1
    return (
        max(0.01, float(raw[0])),
        max(0.01, float(raw[1])),
        max(0.01, float(raw[2])),
    )


def _footprint_half(model: Dict[str, Any]) -> Tuple[float, float]:
    sx, _h, sy = _get_size(model)
    yaw = abs(float(model.get("yaw_deg", 0.0)) % 180.0)
    # At 90° rotation, X and Y footprint swap
    if 45.0 < yaw < 135.0:
        sx, sy = sy, sx
    return sx / 2.0, sy / 2.0


def _half_height(model: Dict[str, Any]) -> float:
    _sx, sh, _sy = _get_size(model)
    return sh / 2.0


def _volume(model: Dict[str, Any]) -> float:
    sx, sh, sy = _get_size(model)
    return sx * sh * sy


def _pose_xy(model: Dict[str, Any]) -> Tuple[float, float]:
    p = model.get("Pose") or {}
    return float(p.get("x", 0.0)), float(p.get("y", 0.0))


def _pose_z(model: Dict[str, Any]) -> float:
    p = model.get("Pose") or {}
    return float(p.get("z", 0.0))


# ---------------------------------------------------------------------------
# Semantic helpers
# ---------------------------------------------------------------------------

def _haystack(item: Dict[str, Any]) -> str:
    parts: List[str] = []
    name = str(item.get("Model") or item.get("name") or "")
    if name:
        parts.append(name)
    for key in ("categories", "tags"):
        raw = item.get(key)
        if isinstance(raw, list):
            parts.extend(str(x) for x in raw if x)
    return " ".join(parts).lower()


def _is_container_like(item: Dict[str, Any]) -> bool:
    return any(t in _haystack(item) for t in _CONTAINER_HINTS)


def _surface_score(receptacle: Dict[str, Any]) -> float:
    haystack = _haystack(receptacle)
    score = 0.0
    if any(t in haystack for t in _SUPPORT_SURFACE_HINTS):
        score += 1.0
    if any(t in haystack for t in _NON_SUPPORT_HINTS):
        score -= 0.7
    if any(t in haystack for t in _CONTAINER_HINTS):
        score += 0.6
    return score


# ---------------------------------------------------------------------------
# Per-instance constraints from semantic plan
# ---------------------------------------------------------------------------

def _instance_constraints(
    placed_models: Sequence[Dict[str, Any]],
    semantic_plan: Dict[str, Any],
) -> List[List[Dict[str, Any]]]:
    """Map each placed model to its per-instance constraint list.

    Plan objects with the same name are matched round-robin to placed
    instances — handles duplicates like four crates that each declared
    `on_top_of(table)` separately in the plan.
    """
    objects = (
        semantic_plan.get("objects", []) if isinstance(semantic_plan, dict) else []
    )
    by_name: Dict[str, List[List[Dict[str, Any]]]] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("Model") or obj.get("name") or "")
        if not name:
            continue
        cs = obj.get("constraints") if isinstance(obj.get("constraints"), list) else []
        by_name.setdefault(name, []).append([c for c in cs if isinstance(c, dict)])

    rr: Dict[str, int] = {}
    rows: List[List[Dict[str, Any]]] = []
    for m in placed_models:
        name = str(m.get("Model") or m.get("name") or "")
        variants = by_name.get(name, [])
        if not variants:
            rows.append([])
            continue
        i = rr.get(name, 0) % len(variants)
        rr[name] = rr.get(name, 0) + 1
        rows.append(variants[i])
    return rows


def _extract_on_target(constraints: Sequence[Dict[str, Any]]) -> str:
    for c in constraints:
        if not isinstance(c, dict):
            continue
        if str(c.get("type", "")).lower() in _ON_CONSTRAINT_TYPES:
            target = str(c.get("target", "")).strip()
            if target:
                return target
    return ""


def _has_center_intent(constraints: Sequence[Dict[str, Any]]) -> bool:
    for c in constraints:
        if not isinstance(c, dict):
            continue
        ctype = str(c.get("type", "")).lower()
        if ctype == "center_aligned":
            return True
        if ctype == "region" and str(c.get("value", "")).lower().strip() in (
            "middle", "center", "centre"
        ):
            return True
    return False


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
    parent = name_to_target.get(on_target, "")
    return 1 + _stacking_depth(on_target, parent, name_to_target, seen)


# ---------------------------------------------------------------------------
# Receptacle picking for items without an explicit on-target
# ---------------------------------------------------------------------------

def _find_best_receptacle(
    item: Dict[str, Any],
    receptacles: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Closest semantically-appropriate surface; semantic score dominates
    so a 2 m table beats a 0.5 m chair."""
    if not receptacles:
        return None
    ix, iy = _pose_xy(item)
    best, best_score = None, float("-inf")
    for r in receptacles:
        rx, ry = _pose_xy(r)
        d = math.hypot(ix - rx, iy - ry)
        score = _surface_score(r) * 1.5 + max(0.0, 1.0 - d / 4.0) * 0.6
        if score > best_score:
            best_score, best = score, r
    return best


# ---------------------------------------------------------------------------
# Pose application
# ---------------------------------------------------------------------------

def _apply_pose(
    out: List[Dict[str, Any]],
    item_idx: int,
    receptacle: Dict[str, Any],
    world_x: float,
    world_y: float,
    *,
    z_extra: float = 0.0,
    drop_offset: float = 0.0,
) -> None:
    """Set the item's pose so it rests on top of the receptacle at (x, y).

    Z = receptacle top + item half-height + small clearance. `drop_offset`
    raises the item further (used for containers so physics drops it in);
    `z_extra` is for breaking ties when items share a slot.
    """
    item = out[item_idx]
    rx, ry = _pose_xy(receptacle)
    rz = _pose_z(receptacle)
    r_hh = _half_height(receptacle)
    i_hh = _half_height(item)
    target_z = rz + r_hh + i_hh + 0.01 + z_extra + drop_offset

    pose = dict(item.get("Pose") or {})
    pose["x"] = world_x
    pose["y"] = world_y
    pose["z"] = target_z
    item["Pose"] = pose

    # Anchor metadata so later passes (gradient resolve, validation repair)
    # can re-attach the item to its receptacle if it drifts in XY.
    receptacle_name = str(receptacle.get("Model") or receptacle.get("name") or "")
    item["_surface_offset"] = {
        "x": world_x - rx,
        "y": world_y - ry,
        "receptacle": receptacle_name,
    }


# ---------------------------------------------------------------------------
# Grid layout
# ---------------------------------------------------------------------------

def _pick_grid_shape(n: int, hx: float, hy: float) -> Tuple[int, int]:
    """Pick (cols, rows) so cols*rows >= n and the grid aspect matches the
    surface aspect (hx along X, hy along Y)."""
    if n <= 1:
        return max(1, n), 1
    aspect = max(0.1, (hx + 0.05) / (hy + 0.05))
    cols = max(1, min(n, int(round(math.sqrt(n * aspect)))))
    rows = math.ceil(n / cols)
    while cols * rows < n:
        cols += 1
        rows = math.ceil(n / cols)
    return cols, rows


def _layout_on_surface(
    *,
    out: List[Dict[str, Any]],
    item_indices: Sequence[int],
    receptacle_idx: int,
    constraint_rows: Sequence[Sequence[Dict[str, Any]]],
) -> None:
    """Lay out all items destined for one open surface as a single grid.

    Centre-intent items (region:middle / center_aligned) snap to the centre
    with a tiny vertical stagger. The rest are clustered by model name and
    arranged in a row-major grid sized to fit on the receptacle.
    """
    receptacle = out[receptacle_idx]
    rx, ry = _pose_xy(receptacle)
    r_hx, r_hy = _footprint_half(receptacle)

    centre_items: List[int] = []
    regular_items: List[int] = []
    for i in item_indices:
        cs = constraint_rows[i] if i < len(constraint_rows) else []
        (centre_items if _has_center_intent(cs) else regular_items).append(i)

    for k, i in enumerate(centre_items):
        _apply_pose(out, i, receptacle, rx, ry, z_extra=k * 0.001)

    if not regular_items:
        return

    # Cluster same-name items: preserve first-appearance order of names.
    by_name: Dict[str, List[int]] = {}
    seen_names: List[str] = []
    for i in regular_items:
        name = str(out[i].get("Model") or out[i].get("name") or "").lower()
        if name not in by_name:
            seen_names.append(name)
            by_name[name] = []
        by_name[name].append(i)
    flat = [i for name in seen_names for i in by_name[name]]
    n = len(flat)

    # Cell pitch is sized for the biggest item in the assignment so even
    # heterogeneous mixes (apples + a mug) avoid mutual overlap.
    max_hx = max(_footprint_half(out[i])[0] for i in flat)
    max_hy = max(_footprint_half(out[i])[1] for i in flat)

    # Inset from the OBB edge: the receptacle's mesh top is often smaller
    # than its axis-aligned bounding box (a table's flared legs widen the
    # OBB but the tabletop itself is narrower). Inset by ~8% of the
    # receptacle half-extent (capped) so items don't appear past the
    # visible surface even when the mesh and OBB diverge.
    safety_x = max(0.02, min(0.08, 0.08 * r_hx))
    safety_y = max(0.02, min(0.08, 0.08 * r_hy))

    usable_hx = r_hx - max_hx - safety_x
    usable_hy = r_hy - max_hy - safety_y

    # Receptacle genuinely too small for these items: stack at centre.
    if usable_hx < 0.0 or usable_hy < 0.0:
        for k, i in enumerate(flat):
            _apply_pose(out, i, receptacle, rx, ry, z_extra=k * 0.001)
        return

    cols, rows = _pick_grid_shape(n, max(usable_hx, 1e-3), max(usable_hy, 1e-3))

    # Compact pitch = item-pitch + a small visible gap. If the surface is
    # large, items cluster instead of being scattered to its corners.
    desired_pitch_x = 2 * max_hx + 0.10
    desired_pitch_y = 2 * max_hy + 0.10
    max_pitch_x = (2 * usable_hx) / max(1, cols - 1) if cols > 1 else 0.0
    max_pitch_y = (2 * usable_hy) / max(1, rows - 1) if rows > 1 else 0.0

    pitch_x = min(desired_pitch_x, max_pitch_x) if cols > 1 else 0.0
    pitch_y = min(desired_pitch_y, max_pitch_y) if rows > 1 else 0.0

    # Centre the grid on the receptacle.
    origin_x = -(cols - 1) * pitch_x / 2.0
    origin_y = -(rows - 1) * pitch_y / 2.0

    for k, i in enumerate(flat):
        col = k % cols
        row = k // cols
        ox = origin_x + col * pitch_x
        oy = origin_y + row * pitch_y
        # Hard clamp to usable region so floating-point drift can't push an
        # item past the rim.
        ox = max(-usable_hx, min(usable_hx, ox))
        oy = max(-usable_hy, min(usable_hy, oy))
        _apply_pose(out, i, receptacle, rx + ox, ry + oy)


def _layout_in_container(
    *,
    out: List[Dict[str, Any]],
    item_indices: Sequence[int],
    receptacle_idx: int,
) -> None:
    """Spawn items above the container so physics settles them down.

    The proxy world treats the container as a solid box, so items end up
    resting on its top in the proxy. The full MuJoCo world uses the actual
    mesh, where they fall into the cavity if one exists. Either way, a
    small ring pattern keeps items from spawning co-axial.
    """
    receptacle = out[receptacle_idx]
    rx, ry = _pose_xy(receptacle)
    r_hx, r_hy = _footprint_half(receptacle)
    r_hh = _half_height(receptacle)

    n = len(item_indices)
    if n == 0:
        return

    drop_start = min(0.12, max(0.05, r_hh * 0.5))
    inset_x = max(0.0, r_hx * 0.4)
    inset_y = max(0.0, r_hy * 0.4)

    for k, i in enumerate(item_indices):
        if n == 1:
            ox, oy = 0.0, 0.0
        else:
            ang = 2.0 * math.pi * k / n
            ox = inset_x * math.cos(ang)
            oy = inset_y * math.sin(ang)
        # Stagger drop heights so items cascade in rather than stacking
        # exactly through each other at the same z.
        z_stagger = k * (2 * _half_height(out[i]) + 0.005)
        _apply_pose(
            out, i, receptacle, rx + ox, ry + oy,
            drop_offset=drop_start + z_stagger,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def solve_small_object_placements(
    *,
    placed_models: Sequence[Dict[str, Any]],
    semantic_plan: Optional[Dict[str, Any]] = None,
    small_threshold_volume: float = 0.06,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Position support-eligible items on their receptacles.

    Eligibility:
      - any item whose plan has `on_top_of(target)` (any size)
      - any item with volume < `small_threshold_volume` and no other anchor
        — auto-attaches to the nearest semantically appropriate surface

    Processing order is shallowest-stacking-first so a crate is positioned
    on the table BEFORE the apple destined for the crate, giving the apple
    the correct receptacle z to rest on.

    For multiple identical-name receptacles (e.g. four crates), items
    sharing the same `on_top_of(target_name)` are distributed round-robin
    across instances so they spread evenly rather than piling onto one.
    """
    print(f"[small_objects] Starting with {len(placed_models)} models...")
    out = [dict(m) for m in placed_models]
    constraint_rows = _instance_constraints(out, semantic_plan or {"objects": []})

    # Per-name → bare-target map (strip any `_N` instance suffix) so the
    # recursive depth lookup matches names rather than instances.
    name_to_target: Dict[str, str] = {}
    for i, m in enumerate(out):
        nm = str(m.get("Model") or m.get("name") or "")
        if not nm:
            continue
        cs = constraint_rows[i] if i < len(constraint_rows) else []
        tgt = _extract_on_target(cs)
        if tgt and nm not in name_to_target:
            base, _ = parse_instance_target(tgt)
            name_to_target[nm] = base

    eligible: List[Tuple[int, int, str]] = []  # (depth, idx, on_target)
    for i, m in enumerate(out):
        if m.get("is_robot", False):
            continue
        nm = str(m.get("Model") or m.get("name") or "")
        cs = constraint_rows[i] if i < len(constraint_rows) else []
        on_target = _extract_on_target(cs)
        if on_target:
            base, _ = parse_instance_target(on_target)
            eligible.append((_stacking_depth(nm, base, name_to_target), i, on_target))
        elif _volume(m) < small_threshold_volume:
            # Skip objects already handled by floor solver (beside/near/face_to)
            floor_types = {"beside", "near", "face_to", "region", "center_aligned",
                           "left_of", "right_of", "in_front_of", "behind"}
            if any(str(c.get("type", "")).lower() in floor_types for c in cs if isinstance(c, dict)):
                continue
            eligible.append((1, i, ""))

    eligible.sort(key=lambda t: (t[0], t[1]))

    receptacle_pool = [
        m for m in out
        if _volume(m) >= small_threshold_volume and not m.get("is_robot", False)
    ]

    # Phase 1: assign each eligible item to a concrete receptacle index.
    rr_cursor: Dict[str, int] = {}
    assignments: List[Tuple[int, int, int]] = []  # (depth, item, recept)
    for depth, i, on_target in eligible:
        if on_target:
            cands = matching_indices(on_target, out, exclude_idx=i)
            if not cands:
                continue
            cur = rr_cursor.get(on_target, 0)
            ridx = cands[cur % len(cands)]
            rr_cursor[on_target] = cur + 1
            print(f"[small_obj] {out[i].get('Model')} → {out[ridx].get('Model')}[{ridx}] (rr={cur})")
        else:
            recept = _find_best_receptacle(out[i], receptacle_pool)
            if recept is None:
                continue
            ridx = next((j for j, mm in enumerate(out) if mm is recept), -1)
            if ridx < 0:
                continue
        assignments.append((depth, i, ridx))

    # Phase 2: process receptacles shallow-first so support objects (crates
    # on table) are placed before their dependants (apples in crates).
    by_recept: Dict[int, List[Tuple[int, int]]] = {}
    for depth, i, ridx in assignments:
        by_recept.setdefault(ridx, []).append((depth, i))

    recept_order = sorted(
        by_recept.keys(),
        key=lambda r: min(d for d, _ in by_recept[r]),
    )

    for ridx in recept_order:
        items_here = [i for _, i in sorted(by_recept[ridx], key=lambda t: t[0])]
        if _is_container_like(out[ridx]):
            _layout_in_container(
                out=out, item_indices=items_here, receptacle_idx=ridx,
            )
        else:
            _layout_on_surface(
                out=out,
                item_indices=items_here,
                receptacle_idx=ridx,
                constraint_rows=constraint_rows,
            )

    print(f"[small_objects] Done, placed {len(out)} models")
    return out
