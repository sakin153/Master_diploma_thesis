"""Resolve planner target names to placed-model indices.

The semantic plan from Stage 4 references targets with a 1-based instance
suffix, e.g. `"table_1"`, `"box_2"`. The placed-model dicts carry only the
bare `Model` name (e.g. `"box"`), so a naive `model["Model"] == target`
comparison silently fails and downstream solvers/validators skip the
constraint. These helpers normalize the comparison.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


_INSTANCE_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_(?P<idx>\d+)$")


def parse_instance_target(target: str) -> Tuple[str, Optional[int]]:
    """Split `"table_1"` into `("table", 0)`. Returns `(name, None)` when no
    suffix is present. The index is 0-based for direct list indexing."""
    raw = str(target or "").strip()
    m = _INSTANCE_SUFFIX_RE.match(raw)
    if not m:
        return raw, None
    try:
        idx = int(m.group("idx"))
    except ValueError:
        return raw, None
    return m.group("base"), max(0, idx - 1)


def matching_indices(
    target: str,
    placed_models: Sequence[Dict[str, Any]],
    *,
    exclude_idx: Optional[int] = None,
) -> List[int]:
    """Return indices of `placed_models` matching the planner target.

    Honors the `_N` instance suffix when present; otherwise returns every
    instance whose `Model` (or `name`) equals the bare target.
    """
    base, instance_idx = parse_instance_target(target)
    same_name: List[int] = []
    for idx, c in enumerate(placed_models):
        if idx == exclude_idx:
            continue
        nm = str(c.get("Model") or c.get("name") or "")
        if nm == base:
            same_name.append(idx)
    if instance_idx is None or not same_name:
        return same_name
    if instance_idx < len(same_name):
        return [same_name[instance_idx]]
    # Suffix overshoots the available instances — fall back to all matches
    # so round-robin distribution still works rather than dropping the item.
    return same_name


def matching_models(
    target: str,
    placed_models: Sequence[Dict[str, Any]],
    *,
    exclude_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    return [placed_models[i] for i in matching_indices(target, placed_models, exclude_idx=exclude_idx)]
