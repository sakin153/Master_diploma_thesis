from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List, Sequence

import logging

from mujoco_scene_editor.inventory.objverse import ObjaverseInventory
from mujoco_scene_editor.inventory.objverse import ObjaverseItem
from mujoco_scene_editor.inventory.objverse import lookup_default_scale
from mujoco_scene_editor.utils.mesh_conversion import convert_to_mujoco_mesh

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedObjaverseMesh:
    name: str
    uid: str
    rel_file: str  # relative path from the mjcf file directory
    scale: float


def _norm_label(label: str) -> str:
    return label.replace("_", " ").strip().lower()


def _label_match_score(label_norm: str, keyword: str) -> int:
    if not label_norm or not keyword:
        return 0
    if label_norm == keyword:
        return 200 + len(keyword)

    # Prefer whole-word matches over generic substring matches.
    if re.search(rf"(^|\s){re.escape(keyword)}($|\s)", label_norm):
        return 150 + len(keyword)

    if keyword in label_norm:
        return 100 + len(keyword)
    if label_norm in keyword:
        return 70 + len(label_norm)
    return 0


def match_lvis_labels(labels: Sequence[str], keywords: Iterable[str], max_labels: int = 3) -> List[str]:
    """Best-effort mapping of extracted keywords to LVIS labels."""

    keywords_lc = [k.strip().lower() for k in keywords if k and k.strip()]
    if not keywords_lc:
        return []

    # Preserve order but deduplicate keywords.
    dedup_keywords: List[str] = []
    for kw in keywords_lc:
        if kw not in dedup_keywords:
            dedup_keywords.append(kw)

    norm_labels: List[tuple[str, str]] = []
    for lbl in labels:
        nl = _norm_label(lbl)
        if nl:
            norm_labels.append((lbl, nl))

    out: List[str] = []

    # First pass: ensure broad keyword coverage (at most one top label per keyword).
    for kw in dedup_keywords:
        best_label: str | None = None
        best_score = 0
        for lbl, nl in norm_labels:
            score = _label_match_score(nl, kw)
            if score <= 0:
                continue
            if score > best_score or (score == best_score and best_label and lbl < best_label):
                best_label = lbl
                best_score = score

        if best_label and best_label not in out:
            out.append(best_label)
            if len(out) >= max_labels:
                return out

    scored: List[tuple[int, str]] = []
    for lbl, nl in norm_labels:
        if lbl in out:
            continue
        score = 0
        for kw in dedup_keywords:
            score = max(score, _label_match_score(nl, kw))
        if score > 0:
            scored.append((score, lbl))

    scored.sort(key=lambda x: (-x[0], x[1]))
    out: List[str] = []
    for _score, lbl in scored:
        if lbl not in out:
            out.append(lbl)
        if len(out) >= max_labels:
            break
    return out


def _safe_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    s = "".join(keep).strip("_")
    return s or "objaverse"


def prepare_objaverse_meshes_for_mjcf(
    *,
    keywords: Sequence[str],
    mjcf_path: Path,
    max_labels: int = 3,
    limit_per_label: int = 2,
    max_total: int = 6,
    max_source_mb: int = 80,
) -> List[PreparedObjaverseMesh]:
    """Download/convert Objaverse meshes likely relevant to a prompt.

    Writes OBJ meshes under: <mjcf_dir>/assets/objaverse/
    Returns relative file paths to be used in MJCF.
    """

    mjcf_path = Path(mjcf_path)
    mjcf_dir = mjcf_path.expanduser().resolve().parent
    asset_dir = mjcf_dir / "assets" / "objaverse"
    asset_dir.mkdir(parents=True, exist_ok=True)

    inv = ObjaverseInventory()
    all_labels = inv.list_labels()
    labels = match_lvis_labels(all_labels, keywords, max_labels=max_labels)
    if not labels:
        return []

    candidates: List[ObjaverseItem] = inv.list_by_labels(
        labels=labels, limit_per_label=limit_per_label
    )
    if not candidates:
        return []

    selected: List[ObjaverseItem] = []
    for it in candidates:
        selected.append(it)
        if len(selected) >= max_total:
            break

    need_download = [it.uid for it in selected if it.path is None]
    downloaded: dict[str, Path] = {}
    if need_download:
        try:
            downloaded = inv.download(need_download)
        except Exception as e:
            logger.warning("Objaverse download failed: %s", e)

    prepared: List[PreparedObjaverseMesh] = []
    for it in selected:
        src = it.path or downloaded.get(it.uid)
        if src is None:
            continue

        src = Path(src)
        try:
            if src.exists() and src.is_file():
                size_mb = src.stat().st_size / (1024 * 1024)
                if size_mb > max_source_mb:
                    continue
        except OSError:
            pass

        safe = _safe_name(it.name)
        dest_obj = asset_dir / f"{safe}_{it.uid[:6]}.obj"

        try:
            converted = convert_to_mujoco_mesh(src, out_ext=".obj", out_path=dest_obj)
        except Exception as e:
            logger.warning("Failed to convert objaverse mesh %s: %s", it.uid, e)
            continue

        rel = converted.resolve().relative_to(mjcf_dir)
        scale = lookup_default_scale(it)
        prepared.append(
            PreparedObjaverseMesh(
                name=safe,
                uid=it.uid,
                rel_file=rel.as_posix(),
                scale=scale,
            )
        )

    return prepared
