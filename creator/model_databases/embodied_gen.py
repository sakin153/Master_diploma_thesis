"""Loader for the HorizonRobotics EmbodiedGenData dataset.

Reads `dataset_index.csv` (uuid, primary_category, secondary_category,
category, description, …, asset_dir, urdf_path, video_path) and resolves
each row to the canonical GLB mesh under `{asset_dir}/mesh/*.glb`.

GLB is preferred because it bundles geometry + textures + PBR materials
in a single file and is loaded directly by `obj2mjcf`-free MuJoCo paths.
Collision/gaussian variants (`*_collision.obj`, `*_gs.ply`) are ignored.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from creator.model_databases.base import BaseLoader


_PRIMARY_GLB_VARIANT_SUFFIXES = ("_collision", "_col", "_phys", "_low", "_lod0", "_lod1", "_proxy")


class EmbodiedGenLoader(BaseLoader):
    """Load model metadata from the EmbodiedGen dataset directory."""

    DEFAULT_DATASET_DIR = Path(
        "/home/sakin153/.cache/huggingface/hub/datasets--HorizonRobotics--EmbodiedGenData/"
        "snapshots/0da911ca5b1c26c64533d8579b2fe9d3f042acb4/dataset"
    )

    def __init__(self, dataset_dir: Optional[str] = None) -> None:
        root = Path(dataset_dir).expanduser() if dataset_dir else self.DEFAULT_DATASET_DIR
        self.dataset_dir = root.resolve()

        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"EmbodiedGen dataset dir not found: {self.dataset_dir}")

        index_path = self.dataset_dir / "dataset_index.csv"
        if not index_path.is_file():
            raise FileNotFoundError(f"dataset_index.csv missing in {self.dataset_dir}")

        self._models = self._read_index(index_path)
        if not self._models:
            raise RuntimeError(f"No usable models loaded from {index_path}")

    def get_models_full(self) -> List[Dict[str, Any]]:
        return [dict(m) for m in self._models]

    def get_models(self) -> Tuple[List[Dict[str, Any]], None]:
        return self.get_models_full(), None

    def _read_index(self, index_path: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with index_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = self._row_to_entry(row)
                if entry is not None:
                    out.append(entry)
        out.sort(key=lambda m: (str(m["categories"][0]).lower(), str(m["name"]).lower()))
        return out

    def _row_to_entry(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        uid = (row.get("uuid") or "").strip()
        if not uid:
            return None

        asset_dir_rel = (row.get("asset_dir") or "").strip()
        if not asset_dir_rel:
            return None

        asset_dir = (self.dataset_dir / asset_dir_rel).resolve()
        mesh_path = self._pick_glb(asset_dir / "mesh")
        if mesh_path is None:
            return None

        primary = self._normalize(row.get("primary_category"))
        secondary = self._normalize(row.get("secondary_category"))
        category = self._normalize(row.get("category"))
        description = (row.get("description") or "").strip()

        # The "name" used by ranking/LLM picks the most-specific label.
        # E.g. "lounge_chair" wins over "chair" / "basic furniture".
        name = category or secondary or primary or mesh_path.stem
        name = self._humanize(name)

        categories = [c for c in (category, secondary, primary) if c]

        tags = self._build_tags(name=name, categories=categories, description=description)

        return {
            "uuid": uid,
            "name": name,
            "categories": categories,
            "tags": tags,
            "description": description,
            "model_loc": str(mesh_path),
            # EmbodiedGen GLBs follow the glTF default (Y-up). Tag every
            # entry so the assembly stage applies euler="90 yaw 0" instead
            # of running the geometric heuristic, which mis-identifies
            # chairs whose Y/Z spans both fall inside the target-height
            # tolerance window.
            "_up_axis": "y",
        }

    @staticmethod
    def _pick_glb(mesh_dir: Path) -> Optional[Path]:
        if not mesh_dir.is_dir():
            return None
        candidates = [p for p in mesh_dir.glob("*.glb") if p.is_file()]
        if not candidates:
            return None

        def is_variant(p: Path) -> bool:
            s = p.stem.lower()
            return any(s.endswith(suf) for suf in _PRIMARY_GLB_VARIANT_SUFFIXES)

        non_variant = [p for p in candidates if not is_variant(p)]
        pool = non_variant or candidates
        # Prefer shortest stem (typically the canonical mesh).
        pool.sort(key=lambda p: (len(p.stem), p.name))
        return pool[0]

    @staticmethod
    def _normalize(value: Optional[str]) -> str:
        raw = (value or "").strip().lower()
        raw = raw.replace("-", " ")
        raw = re.sub(r"\s+", " ", raw)
        return raw

    @staticmethod
    def _humanize(value: str) -> str:
        # "lounge_chair" -> "lounge chair"; keep lowercase for matching.
        return value.replace("_", " ").strip()

    @staticmethod
    def _build_tags(*, name: str, categories: Sequence[str], description: str) -> List[str]:
        tags: List[str] = []
        for c in categories:
            if c and c not in tags:
                tags.append(c)

        for token in re.findall(r"[a-z0-9]+", name.lower()):
            if len(token) > 1 and token not in tags:
                tags.append(token)

        if description:
            for token in re.findall(r"[a-z0-9]+", description.lower()):
                if len(token) > 2 and token not in tags:
                    tags.append(token)

        if "glb" not in tags:
            tags.append("glb")
        return tags
