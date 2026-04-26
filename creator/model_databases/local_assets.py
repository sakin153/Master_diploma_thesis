import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple  # noqa: F401

from creator.model_databases.base import BaseLoader


_MESH_EXTENSIONS = {".obj", ".stl", ".glb", ".gltf", ".ply", ".dae"}
_CATEGORY_TAG_ALIASES = {
    "apple": ["fruit", "fruits"],
    "crate": ["box", "container"],
    "table": ["desk"],
}

# Format priority for the same logical asset: prefer GLB (single-file PBR,
# Y-up enforced) over OBJ over others. Lower index = higher preference.
_FORMAT_PRIORITY = {".glb": 0, ".gltf": 1, ".obj": 2, ".dae": 3, ".stl": 4, ".ply": 5}

# Suffixes that indicate this file is a *variant* of a primary asset
# (e.g. `wooden_patterned_table_collision.obj` is the collision-only version
# of `wooden_patterned_table.glb`). When the primary file exists, we drop
# the variant from the catalog so the LLM doesn't get duplicate entries.
_VARIANT_SUFFIXES = (
    "_collision", "_col", "_phys",
    "_gs", "_gaussian", "_splat",
    "_low", "_lowpoly", "_lod0", "_lod1", "_lod2",
    "_proxy", "_simple", "_decim", "_decimated",
)


class LocalAssetsLoader(BaseLoader):
    """Load model metadata from a local assets directory.

    Priority order:
    1. Use manifest.json entries when available.
    2. Add any remaining mesh files found by recursive scan.
    """

    def __init__(self, assets_dir: Optional[str] = None) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        default_assets = self.project_root / "assets"
        raw_assets_dir = Path(assets_dir).expanduser() if assets_dir else default_assets
        self.assets_dir = raw_assets_dir.resolve()

        if not self.assets_dir.exists():
            raise FileNotFoundError(f"Assets directory does not exist: {self.assets_dir}")
        if not self.assets_dir.is_dir():
            raise NotADirectoryError(f"Assets path is not a directory: {self.assets_dir}")

        self._models = self._discover_models()
        if not self._models:
            raise RuntimeError(
                f"No mesh files found under assets directory: {self.assets_dir}"
            )

    def get_models_full(self) -> List[Dict[str, Any]]:
        return [dict(m) for m in self._models]

    def get_models(self) -> Tuple[List[Dict[str, Any]], None]:
        return self.get_models_full(), None

    def _discover_models(self) -> List[Dict[str, Any]]:
        seen_paths: Set[Path] = set()
        models: List[Dict[str, Any]] = []

        models.extend(self._models_from_manifest(seen_paths))
        models.extend(self._models_from_filesystem(seen_paths))

        models.sort(
            key=lambda m: (
                str((m.get("categories") or [""])[0]).lower(),
                str(m.get("name") or "").lower(),
            )
        )
        return models

    def _models_from_manifest(self, seen_paths: Set[Path]) -> List[Dict[str, Any]]:
        manifest_path = self.assets_dir / "manifest.json"
        if not manifest_path.exists():
            return []

        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        if not isinstance(data, dict):
            return []

        out: List[Dict[str, Any]] = []
        for group_name, rows in data.items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue

                mesh_path = self._resolve_manifest_mesh_path(row.get("obj_file"))
                use_manifest_identity = mesh_path is not None
                if mesh_path is None:
                    mesh_path = self._guess_mesh_path_from_row(group_name, row)
                if mesh_path is None:
                    continue

                mesh_path = mesh_path.resolve()
                if mesh_path in seen_paths:
                    continue
                seen_paths.add(mesh_path)

                uid = str(row.get("uid") or "").strip() if use_manifest_identity else ""
                name = str(row.get("name") or "").strip() if use_manifest_identity else ""
                category = str(row.get("category") or group_name or "").strip()
                out.append(
                    self._build_model_entry(
                        mesh_path,
                        preferred_uid=uid,
                        preferred_name=name,
                        category=category,
                    )
                )

        return out

    def _models_from_filesystem(self, seen_paths: Set[Path]) -> List[Dict[str, Any]]:
        # Group all mesh files in the assets tree by their "logical asset"
        # key: (parent_dir, stem-with-variant-suffix-stripped). For each
        # group keep only the highest-priority file (GLB > OBJ > others)
        # and drop variant siblings (_collision, _gs, _low, etc.).
        groups: Dict[Tuple[str, str], List[Path]] = {}
        for path in self.assets_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _MESH_EXTENSIONS:
                continue
            key = (str(path.parent), self._logical_stem(path.stem))
            groups.setdefault(key, []).append(path)

        out: List[Dict[str, Any]] = []
        for (_parent, _logical), files in sorted(groups.items()):
            primary = self._pick_primary_mesh(files)
            resolved = primary.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            category = primary.parent.name
            out.append(self._build_model_entry(resolved, category=category))

        return out

    @staticmethod
    def _logical_stem(stem: str) -> str:
        """Strip variant suffixes so collision/lod/gs siblings group together."""
        s = stem.lower()
        # Repeatedly strip the longest matching suffix (handle e.g. "_lod0_collision")
        changed = True
        while changed:
            changed = False
            for suffix in _VARIANT_SUFFIXES:
                if s.endswith(suffix):
                    s = s[: -len(suffix)]
                    changed = True
                    break
        return s

    @staticmethod
    def _pick_primary_mesh(files: List[Path]) -> Path:
        """Pick the best representative file: lowest format-priority index,
        and within that, prefer the one whose stem has NO variant suffix."""
        def has_variant_suffix(p: Path) -> bool:
            s = p.stem.lower()
            return any(s.endswith(v) for v in _VARIANT_SUFFIXES)

        def sort_key(p: Path) -> Tuple[int, int, str]:
            ext_pri = _FORMAT_PRIORITY.get(p.suffix.lower(), 99)
            variant_pri = 1 if has_variant_suffix(p) else 0
            return (ext_pri, variant_pri, str(p))

        return sorted(files, key=sort_key)[0]

    def _resolve_manifest_mesh_path(self, obj_file: Any) -> Optional[Path]:
        if not isinstance(obj_file, str) or not obj_file.strip():
            return None

        raw = obj_file.strip()
        candidate = Path(raw)
        probes = []
        if candidate.is_absolute():
            probes.append(candidate)
        else:
            probes.append((self.project_root / candidate).resolve())
            probes.append((self.assets_dir / candidate).resolve())
            probes.append((self.assets_dir / candidate.name).resolve())

        for probe in probes:
            if probe.exists() and probe.is_file() and probe.suffix.lower() in _MESH_EXTENSIONS:
                return probe

        return None

    def _guess_mesh_path_from_row(
        self,
        group_name: str,
        row: Dict[str, Any],
    ) -> Optional[Path]:
        group_dir = self.assets_dir / str(group_name)
        search_dir = (
            group_dir if group_dir.exists() and group_dir.is_dir() else self.assets_dir
        )
        mesh_files = self._mesh_files_in_dir(search_dir)
        if not mesh_files:
            return None

        uid = str(row.get("uid") or "").strip().lower()
        if uid:
            uid_short = uid[:6]
            for fpath in mesh_files:
                stem = fpath.stem.lower()
                if uid in stem or uid_short in stem:
                    return fpath

        name = str(row.get("name") or "").strip().lower()
        if name:
            name_tokens = [t for t in re.findall(r"[a-z0-9]+", name) if len(t) > 2]
            for fpath in mesh_files:
                stem_tokens = set(re.findall(r"[a-z0-9]+", fpath.stem.lower()))
                if name_tokens and any(token in stem_tokens for token in name_tokens):
                    return fpath

        return mesh_files[0]

    def _mesh_files_in_dir(self, directory: Path) -> List[Path]:
        files: List[Path] = []
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in _MESH_EXTENSIONS:
                files.append(path)
        return files

    def _build_model_entry(
        self,
        mesh_path: Path,
        *,
        preferred_uid: str = "",
        preferred_name: str = "",
        category: str = "",
    ) -> Dict[str, Any]:
        uid = preferred_uid
        if not uid:
            uid = self._extract_uid(mesh_path.stem)
        if not uid:
            uid = self._hash_relative(mesh_path)
        name = preferred_name or self._pretty_name(mesh_path.stem)
        category_norm = self._normalize_label(category)
        if not category_norm:
            category_norm = self._normalize_label(mesh_path.parent.name)
        tags = self._build_tags(name=name, category=category_norm, mesh_path=mesh_path)

        return {
            "uuid": uid,
            "name": name,
            "categories": [category_norm] if category_norm else [],
            "tags": tags,
            "model_loc": str(mesh_path),
        }

    def _extract_uid(self, stem: str) -> str:
        match = re.search(r"([0-9a-f]{6,32})$", stem.lower())
        return match.group(1) if match else ""

    def _hash_relative(self, mesh_path: Path) -> str:
        rel = str(mesh_path.resolve().relative_to(self.assets_dir.resolve())).encode("utf-8")
        return hashlib.md5(rel).hexdigest()[:16]

    def _pretty_name(self, stem: str) -> str:
        # Strip common hex suffixes, e.g. Some_Model_ab12cd
        clean = re.sub(r"[_-][0-9a-f]{6,32}$", "", stem, flags=re.IGNORECASE)
        clean = clean.replace("_", " ").replace("-", " ")
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean.title() if clean else "Asset Model"

    def _normalize_label(self, value: str) -> str:
        raw = (value or "").strip().lower()
        raw = raw.replace("_", " ").replace("-", " ")
        raw = re.sub(r"\s+", " ", raw)
        return raw

    def _build_tags(self, *, name: str, category: str, mesh_path: Path) -> List[str]:
        tags: List[str] = []
        if category:
            tags.append(category)
            tags.extend(_CATEGORY_TAG_ALIASES.get(category, []))

        for token in re.findall(r"[a-z0-9]+", name.lower()):
            if len(token) <= 2:
                continue
            if token not in tags:
                tags.append(token)

        ext_tag = mesh_path.suffix.lower().lstrip(".")
        if ext_tag and ext_tag not in tags:
            tags.append(ext_tag)

        return tags
