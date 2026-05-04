# Загрузчик каталога 3D-моделей из датасета EmbodiedGen.

import csv
import re
from pathlib import Path

_EMBODIED_GEN_DIR = Path(
    "/home/sakin153/.cache/huggingface/hub/"
    "datasets--HorizonRobotics--EmbodiedGenData/"
    "snapshots/0da911ca5b1c26c64533d8579b2fe9d3f042acb4/dataset"
)
_GLB_VARIANT_SUFFIXES = ("_collision", "_col", "_phys", "_low", "_lod0", "_lod1", "_proxy")


def load_catalog(dataset_dir=None):
    """Загружает каталог из датасета EmbodiedGen.

    Возвращает список моделей:
        [{"uuid": "...", "name": "...", "categories": [...], "tags": [...], "model_loc": "..."}, ...]
    """
    root = Path(dataset_dir).resolve() if dataset_dir else _EMBODIED_GEN_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"Датасет EmbodiedGen не найден: {root}")

    index_path = root / "dataset_index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"dataset_index.csv не найден в {root}")

    models = []
    with index_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entry = _parse_row(row, root)
            if entry:
                models.append(entry)

    models.sort(key=lambda m: (str((m.get("categories") or [""])[0]).lower(), str(m.get("name", "")).lower()))
    print(f"[catalog] Загружено {len(models)} моделей из EmbodiedGen")
    return models


def _parse_row(row, root):
    uid = (row.get("uuid") or "").strip()
    asset_dir_rel = (row.get("asset_dir") or "").strip()
    if not uid or not asset_dir_rel:
        return None

    mesh_path = _pick_glb((root / asset_dir_rel / "mesh").resolve())
    if mesh_path is None:
        return None

    def norm(val):
        return re.sub(r"\s+", " ", (val or "").strip().lower().replace("-", " "))

    primary = norm(row.get("primary_category"))
    secondary = norm(row.get("secondary_category"))
    category = norm(row.get("category"))
    description = (row.get("description") or "").strip()
    name = (category or secondary or primary or mesh_path.stem).replace("_", " ").strip()

    categories = [c for c in (category, secondary, primary) if c]
    tags = list(dict.fromkeys(
        categories
        + [t for t in re.findall(r"[a-z0-9]+", name) if len(t) > 1]
        + [t for t in re.findall(r"[a-z0-9]+", description.lower()) if len(t) > 2]
        + ["glb"]
    ))

    return {"uuid": uid, "name": name, "categories": categories, "tags": tags,
            "description": description, "model_loc": str(mesh_path), "_up_axis": "y"}


def _pick_glb(mesh_dir):
    if not mesh_dir.is_dir():
        return None
    candidates = [p for p in mesh_dir.glob("*.glb") if p.is_file()]
    if not candidates:
        return None
    non_variant = [p for p in candidates if not any(p.stem.lower().endswith(s) for s in _GLB_VARIANT_SUFFIXES)]
    pool = non_variant or candidates
    pool.sort(key=lambda p: (len(p.stem), p.name))
    return pool[0]
