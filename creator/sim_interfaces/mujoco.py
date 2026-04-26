import copy
import json
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import trimesh
from obj2mjcf.cli import Args, CoacdArgs, process_obj

from creator.contexts_prompts.constraints import fmt_constraints_plan_tmpl
from creator.placement import (
    build_semantic_plan,
    evaluate_constraint_violations,
    repair_layout_by_constraints,
    solve_floor_placements,
    solve_small_object_placements,
    solve_wall_placements,
    validate_and_repair_layout,
)
from creator.sim_interfaces.base import BaseSimInterface
from creator.utils.cache import Cache


class MujocoSimInterface(BaseSimInterface):
    # Target HEIGHT (Z in scene, after 90° X rotation) per category in metres.
    # Key = tuple of keywords. Uses the first match (most-specific first).
    # Max horizontal extent (max of width, depth) for flat/compact objects.
    # After height-based scaling, if the horizontal span exceeds this, we clamp it.
    # Key = tuple of keywords, value = max allowed horizontal extent in metres.
    _MAX_HORIZ_M: Tuple[Tuple[Tuple[str, ...], float], ...] = (
        (("crate", "container", "box"), 0.45),  # tabletop crate footprint clamp
        (("plate", "dish"), 0.32),      # dinner plate ≤ 32 cm
        (("bowl",), 0.25),              # mixing bowl ≤ 25 cm
        (("cup", "mug"), 0.12),         # cup width ≤ 12 cm
        (("laptop",), 0.40),            # laptop footprint ≤ 40 cm
        (("keyboard",), 0.45),
        (("remote",), 0.22),
        (("phone", "smartphone"), 0.08),
        (("apple", "fruit"), 0.10),
        (("candle",), 0.10),
        (("book",), 0.35),
        (("tray",), 0.50),
    )

    _HEIGHT_TARGETS_M: Tuple[Tuple[Tuple[str, ...], float], ...] = (
        # Small objects
        (("cup", "mug"), 0.11),
        (("plate", "dish"), 0.04),
        (("bowl",), 0.09),
        (("bottle", "water bottle"), 0.26),
        (("phone", "smartphone"), 0.15),
        (("laptop",), 0.02),   # closed lid height
        (("book",), 0.24),
        (("keyboard",), 0.04),
        (("remote",), 0.03),
        (("candle",), 0.12),
        (("apple", "fruit"), 0.08),
        (("vase",), 0.30),
        (("pillow", "cushion"), 0.15),
        (("crate", "container", "box"), 0.35),
        # Furniture
        (("stool",), 0.45),
        (("coffee table", "cocktail table"), 0.45),
        (("side table", "end table", "accent table"), 0.55),
        (("nightstand", "bedside"), 0.60),
        (("ottoman",), 0.45),
        (("chair", "armchair", "seat"), 0.85),
        (("bar stool",), 0.75),
        (("sofa", "couch", "loveseat", "settee"), 0.85),
        (("table", "dining table"), 0.75),
        (("desk", "writing desk", "work table"), 0.75),
        (("counter", "kitchen counter"), 0.90),
        (("bed", "mattress"), 0.55),
        (("bench",), 0.50),
        (("bookcase", "bookshelf"), 1.80),
        (("shelf", "shelving", "rack"), 1.60),
        (("cabinet", "cupboard"), 1.80),
        (("wardrobe", "closet", "armoire"), 2.00),
        (("dresser", "chest", "sideboard", "buffet"), 1.00),
        # Electronics / appliances
        (("tv", "television", "flat screen"), 0.60),
        (("monitor", "screen", "display"), 0.40),
        (("refrigerator", "fridge"), 1.70),
        (("microwave",), 0.35),
        (("oven", "stove", "range"), 0.90),
        (("washing machine", "washer", "dryer"), 0.90),
        # Lighting
        (("floor lamp", "standing lamp", "torchiere"), 1.60),
        (("table lamp", "desk lamp", "bedside lamp"), 0.45),
        (("chandelier",), 0.60),
        (("lamp",), 0.50),
        # Sanitaryware
        (("toilet",), 0.75),
        (("sink", "wash basin"), 0.90),
        (("bathtub",), 0.55),
        # Structural
        (("door",), 2.10),
        (("window",), 1.20),
        (("stairs", "staircase"), 2.40),
        # Decor
        (("plant", "potted plant", "tree"), 1.00),
        (("painting", "picture frame", "wall art"), 0.60),
        (("mirror",), 1.20),
        # Vehicles
        (("car", "vehicle", "truck"), 1.50),
    )

    _CONCAVE_COLLISION_HINTS: Tuple[str, ...] = (
        "crate",
        "container",
        "box",
        "basket",
        "bin",
        "drawer",
        "ящик",
    )

    def __init__(self, chosen_model: str, cache_dir: Optional[str] = None) -> None:
        super().__init__(chosen_model)
        self.cache = Cache(cache=cache_dir)

    def check_world(self, world: Dict[str, Union[str, int, float]]) -> None:
        pass

    def generate_world(self) -> str:
        template_world_path = os.path.join(self.cache.worlds_path, "empty.sdf")

        return template_world_path

    def find_entries_by_name(
        self, name: str, full_list: List[Dict]
    ) -> Tuple[List[int], List[Dict]]:
        locs = []
        models = []
        for i, entry in enumerate(full_list):
            if entry["name"] == name:
                locs.append(i)
                models.append(entry)
        return locs, models

    def add_models(
        self,
        chosen_models: List[Dict],
        models: List[Dict],
        query: str,
        path_to_save: str,
        world_path: Optional[str] = None,
        room_half_size: float = 5.0,
        pre_placed_models: Optional[List[Dict]] = None,
        semantic_plan: Optional[Dict] = None,
    ) -> List[Dict]:
        """Assemble MuJoCo scene XML from chosen models and placement data.

        When `pre_placed_models` is provided (from runner.py stages 2-4), the
        loading / sizing / placement pipeline is skipped and XML assembly uses
        the pre-computed positions directly.

        When called standalone (no pre_placed_models), the full pipeline runs.
        """
        if pre_placed_models is not None:
            # Fast path: positions already computed by the runner pipeline.
            # We only need the uuid→mesh-path dict for XML assembly.
            full_placed_models = pre_placed_models
            objects = {
                m["uuid"]: m["model_loc"]
                for m in full_placed_models
                if m.get("uuid") and m.get("model_loc")
            }
        else:
            # Legacy / standalone path: run the full internal pipeline.
            full_placed_models = self.get_full_placed_models(chosen_models, models)
            objects = self.load_objects(full_placed_models)
            for i, _ in enumerate(full_placed_models):
                full_placed_models[i]["model_loc"] = objects[full_placed_models[i]["uuid"]]
                full_placed_models[i]["save_fn"] = full_placed_models[i]["uuid"] + f"_{i}"
            full_placed_models = self.update_model_sizes(full_placed_models)
            full_placed_models = self.normalize_models_to_realistic_scale(
                full_placed_models,
                query=query,
            )

            if semantic_plan is None:
                semantic_plan = build_semantic_plan(
                    prompt_model=self.prompt_model_for_constraints,
                    prompt_template=fmt_constraints_plan_tmpl,
                    query=query,
                    chosen_model=self.chosen_model,
                    chosen_models=chosen_models,
                    context_models=models,
                )
                self.save_constraint_graph(
                    semantic_plan=semantic_plan,
                    query=query,
                    output_filename="scene_graph_latest.json",
                )

            # Deterministic geometry: floor -> wall -> surface objects
            full_placed_models = solve_floor_placements(
                full_placed_models=full_placed_models,
                semantic_plan=semantic_plan,
                room_half_size=room_half_size,
                grid_step=0.8,
                yaw_candidates_deg=(0.0, 90.0, 180.0, 270.0),
                beam_width=12,
            )
            full_placed_models = solve_wall_placements(
                placed_models=full_placed_models,
                semantic_plan=semantic_plan,
                room_half_size=room_half_size,
            )
            full_placed_models = solve_small_object_placements(
                placed_models=full_placed_models,
                semantic_plan=semantic_plan,
                small_threshold_volume=0.06,
            )

            full_placed_models = validate_and_repair_layout(full_placed_models)

            violations_before = evaluate_constraint_violations(
                full_placed_models,
                semantic_plan,
                room_half_size=room_half_size,
            )
            violations_after = list(violations_before)
            accepted_repair = False
            if violations_before:
                repaired_candidate = repair_layout_by_constraints(
                    full_placed_models,
                    semantic_plan,
                    room_half_size=room_half_size,
                    iterations=2,
                )
                candidate_after = evaluate_constraint_violations(
                    repaired_candidate,
                    semantic_plan,
                    room_half_size=room_half_size,
                )
                if len(candidate_after) <= len(violations_before):
                    full_placed_models = repaired_candidate
                    violations_after = candidate_after
                    accepted_repair = True

            print(
                "Constraint validation:"
                f" before={len(violations_before)} after={len(violations_after)}"
            )
            report_path = Path(__file__).resolve().parents[2] / "scene_constraint_report_latest.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "query": query,
                        "violations_before": violations_before,
                        "violations_after": violations_after,
                        "accepted_repair": accepted_repair,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        main_root = self.create_main_root(full_placed_models, objects)
        tree = self.create_tree(main_root, room_half_size=room_half_size)
        self.write_tree_to_file(tree, path_to_save)
        self.try_compile_in_mujoco(path_to_save)

        # Post-assembly validation
        from creator.scene.validator import validate_placements, validate_mjcf, validate_scales
        validate_scales(full_placed_models, verbose=True)
        validate_placements(full_placed_models, verbose=True)
        validate_mjcf(path_to_save, verbose=True)

        return full_placed_models

    def get_world_extension(self) -> str:
        return ".xml"

    def get_full_placed_models(
        self, placed_models: List[Dict], models: List[Dict]
    ) -> List[Dict]:
        full_placed_models = []
        per_name_cursor: Dict[str, int] = {}
        for model in placed_models:
            locs, model_entries = self.find_entries_by_name(model["Model"], models)
            if not model_entries:
                continue

            name = str(model.get("Model", ""))
            idx = per_name_cursor.get(name, 0) % len(model_entries)
            per_name_cursor[name] = per_name_cursor.get(name, 0) + 1

            selected_entry = copy.deepcopy(model_entries[idx])
            selected_entry.update(model)
            full_placed_models.append(selected_entry)
        return full_placed_models

    def load_objects(self, full_placed_models: List[Dict]) -> Dict[str, Optional[str]]:
        """Resolve mesh paths for models.

        The generation pipeline now uses a local assets catalog. Each model row
        should already include `model_loc` (absolute path) or a relative path.
        Missing paths are marked as None so XML assembly can substitute a
        fallback cube and continue.
        """
        result: Dict[str, Optional[str]] = {}
        project_root = Path(__file__).resolve().parents[2]

        for i, entry in enumerate(full_placed_models):
            uid = str(entry.get("uuid") or entry.get("name") or f"model_{i}")
            raw_path = entry.get("model_loc")
            if not raw_path:
                raw_path = entry.get("local_path")
            if not raw_path:
                raw_path = entry.get("obj_file")
            if not raw_path:
                raw_path = entry.get("mesh_path")

            resolved_path: Optional[str] = None
            if isinstance(raw_path, str) and raw_path.strip():
                candidate = Path(raw_path.strip())
                if not candidate.is_absolute():
                    candidate = (project_root / candidate).resolve()
                if candidate.exists() and candidate.is_file():
                    resolved_path = str(candidate)

            if resolved_path is None:
                model_name = entry.get("Model") or entry.get("name") or uid
                print(f"Warning: mesh path not found for '{model_name}'.")
            result[uid] = resolved_path

        return result

    def _detect_up_axis(
        self,
        model_name: str,
        mesh: "trimesh.Trimesh",
    ) -> str:
        """Pick the mesh axis ('y' or 'z') that should map to scene-Z (vertical).

        Euler conventions used downstream:
          'y' → euler="90 0 yaw" (rotate +90° around X: mesh-Y → scene-Z).
          'z' → euler="0 0 yaw"  (already upright).

        Strategy, in order of reliability:
          1. Category-aware match: pick axis whose unit-scaled extent best
             matches `target_height_m(model_name)` on a log scale. This is
             the strongest signal — it directly resolves "table on edge"
             cases where the geometric heuristics tie.
          2. Floor-rest test: if exactly one of Y/Z has min ≈ 0 with the
             other axis centered, that axis is up.
          3. Centroid asymmetry: axis with strongest |centroid - midpoint|/span
             (mass distribution gives away the up direction).
          4. Footprint/height ratio: prefer axis where the perpendicular
             footprint dominates (most furniture is wider than tall).
          5. Fallback to 'y' (GLB/Objaverse convention).
        """
        bmin, bmax = mesh.bounds
        bmin = [float(v) for v in bmin]
        bmax = [float(v) for v in bmax]
        spans = [max(1e-9, bmax[i] - bmin[i]) for i in range(3)]
        max_dim = max(spans)
        unit_scale = self.infer_unit_scale(max_dim)
        try:
            centroid = mesh.centroid
            cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])
        except Exception:
            cx = 0.5 * (bmin[0] + bmax[0])
            cy = 0.5 * (bmin[1] + bmax[1])
            cz = 0.5 * (bmin[2] + bmax[2])
        centers = [cx, cy, cz]

        # Stage 1: category-aware. Use target height to disambiguate.
        # Only meaningful if the category is in our table (avoid LLM here
        # to keep `update_model_sizes` cheap and deterministic).
        target_h = self._lookup_target_height(model_name)
        if target_h is not None and target_h > 0:
            log_t = math.log(target_h)
            best_axis: Optional[str] = None
            best_dist = float("inf")
            for axis_idx, axis_name in ((1, "y"), (2, "z")):
                h_m = spans[axis_idx] * unit_scale
                if h_m <= 1e-6:
                    continue
                d = abs(math.log(h_m) - log_t)
                if d < best_dist:
                    best_dist = d
                    best_axis = axis_name
            # Accept only if the best candidate is within ~2.5x of target
            # (log(2.5) ≈ 0.92). Otherwise the category target is unreliable
            # for this asset and we fall through to geometric heuristics.
            if best_axis is not None and best_dist < math.log(2.5):
                return best_axis

        # Stage 2: floor-rest test
        floor_thresh = 0.10
        y_on_floor = abs(bmin[1]) <= floor_thresh * spans[1]
        z_on_floor = abs(bmin[2]) <= floor_thresh * spans[2]
        x_centered = abs(cx - 0.5 * (bmin[0] + bmax[0])) <= 0.10 * spans[0]
        y_centered = abs(cy - 0.5 * (bmin[1] + bmax[1])) <= 0.10 * spans[1]
        z_centered = abs(cz - 0.5 * (bmin[2] + bmax[2])) <= 0.10 * spans[2]
        if y_on_floor and not z_on_floor and x_centered and z_centered:
            return "y"
        if z_on_floor and not y_on_floor and x_centered and y_centered:
            return "z"

        # Stage 3: centroid asymmetry — axis with mass concentrated off-center
        # is the natural up axis (table top / chair seat / etc.)
        asymm = []
        for axis_idx in range(3):
            mid = 0.5 * (bmin[axis_idx] + bmax[axis_idx])
            asymm.append(abs(centers[axis_idx] - mid) / spans[axis_idx])
        # Restrict to Y/Z (X-up not supported by assembly euler).
        if asymm[1] > 0.10 and asymm[1] >= asymm[2]:
            return "y"
        if asymm[2] > 0.10 and asymm[2] > asymm[1]:
            return "z"

        # Stage 4: footprint/height ratio
        ratio_y = (spans[0] * spans[2]) / max(1e-6, spans[1])
        ratio_z = (spans[0] * spans[1]) / max(1e-6, spans[2])
        if ratio_y > ratio_z * 1.15:
            return "y"
        if ratio_z > ratio_y * 1.15:
            return "z"

        # Stage 5: fallback
        return "y"

    def _lookup_target_height(self, model_name: str) -> Optional[float]:
        """Like `target_height_m` but returns None for unknown categories
        (no LLM call). Used by up-axis detection to avoid recursion / cost."""
        name = str(model_name or "").lower()
        for keywords, height in self._HEIGHT_TARGETS_M:
            if any(kw in name for kw in keywords):
                return float(height)
        return None

    def update_model_sizes(self, models: List[Dict]) -> List[Dict]:
        updated_models = models
        for i, model in enumerate(models):
            if not model.get("model_loc") or not os.path.exists(str(model.get("model_loc"))):
                # Skip missing models - will use fallback cube
                continue
            try:
                mesh = trimesh.load(model["model_loc"], force="mesh")
                updated_models[i]["size"] = mesh.extents
                model_name = str(model.get("Model") or model.get("name") or "")
                updated_models[i]["_up_axis"] = self._detect_up_axis(
                    model_name, mesh,
                )
            except Exception as e:
                print(f"Warning: Failed to load mesh for {model.get('Model')}: {e}")
                # Keep original size, will use fallback
        return updated_models

    def try_compile_in_mujoco(self, xml_path: str) -> None:
        try:
            import mujoco  # type: ignore

            mujoco.MjModel.from_xml_path(xml_path)
            print("MuJoCo compile validation: OK")
        except Exception as exc:
            print(f"MuJoCo compile validation: FAILED ({exc})")

    def infer_unit_scale(self, max_dim: float) -> float:
        if max_dim <= 0.0:
            return 1.0
        if max_dim > 100.0:
            return 0.001  # millimeters -> meters
        if max_dim > 10.0:
            return 0.01  # centimeters -> meters
        if max_dim > 3.5:
            return 0.1  # decimeters -> meters for overly large indoor assets
        if max_dim < 0.01:
            return 100.0
        if max_dim < 0.05:
            return 10.0
        return 1.0
    def target_height_m(self, model_name: str) -> float:
        """Return expected real-world height (Z after rotation) in metres.

        Uses the hardcoded table first; falls back to LLM inference for
        categories not covered by the table.
        """
        name = str(model_name or "").lower()
        for keywords, height in self._HEIGHT_TARGETS_M:
            if any(kw in name for kw in keywords):
                return height

        # Unknown category: ask LLM (result is cached in SQLite)
        from creator.llm.model import ask_object_height_m
        llm_h = ask_object_height_m(model_name)
        if llm_h is not None:
            return llm_h

        return 1.0  # last-resort generic fallback

    def target_height_bounds_m(self, model_name: str) -> Tuple[float, float]:
        """Acceptable height range: [65%, 160%] of target."""
        target = self.target_height_m(model_name)
        return max(0.03, target * 0.65), max(0.06, target * 1.60)

    def parse_scale_hints(self, raw_scale_output: Any) -> Dict[str, float]:
        rows: List[Dict[str, Any]] = []
        if isinstance(raw_scale_output, list):
            rows = [r for r in raw_scale_output if isinstance(r, dict)]
        elif isinstance(raw_scale_output, dict):
            if "Model" in raw_scale_output and "Scale" in raw_scale_output:
                rows = [raw_scale_output]
            else:
                nested = raw_scale_output.get("answer") or raw_scale_output.get("models")
                if isinstance(nested, list):
                    rows = [r for r in nested if isinstance(r, dict)]

        out: Dict[str, float] = {}
        for row in rows:
            model_name = row.get("Model") or row.get("model")
            raw_scale = row.get("Scale") or row.get("scale")
            if not isinstance(model_name, str):
                continue
            try:
                scale = float(raw_scale)
            except (TypeError, ValueError):
                continue
            if scale <= 0.0:
                continue
            out[model_name.strip().lower()] = max(1e-4, min(500.0, scale))
        return out

    def llm_scale_hints(self, models: List[Dict], query: str) -> Dict[str, float]:
        if not models:
            return {}

        use_llm = os.getenv("CIARE_LLM_SCALE_HINTS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not use_llm:
            return {}

        models_for_scale: List[Dict[str, Union[str, List[float]]]] = []
        for m in models:
            raw_size = m.get("size")
            if raw_size is None or len(raw_size) < 3:
                continue
            try:
                sx = float(raw_size[0])
                sy = float(raw_size[1])
                sz = float(raw_size[2])
            except (TypeError, ValueError):
                continue

            name = str(m.get("Model") or m.get("name") or "").strip()
            if not name:
                continue
            models_for_scale.append({"Model": name, "Size": [sx, sy, sz]})

        if not models_for_scale:
            return {}

        try:
            raw = self.prompt_model_for_scale(models_for_scale, query)
        except BaseException:
            return {}
        return self.parse_scale_hints(raw)

    def normalize_models_to_realistic_scale(
        self,
        models: List[Dict],
        query: str = "",
    ) -> List[Dict]:
        """Scale each model so its HEIGHT matches real-world category height.

        Strategy (height-based, more accurate than max-dim):
        1. Detect unit (mm/cm/m) from raw mesh extents.
        2. The mesh HEIGHT = mesh Y extent (GLB/Objaverse use Y-up convention).
           After MuJoCo's euler="90 0 yaw" rotation, Y_mesh → Z_scene = height.
        3. Scale uniformly so Y_mesh * unit_scale * scale = target_height.
        4. Clamp to [65%, 160%] of target to avoid extremes.
        5. Optional LLM hint can refine further (geometric mean with deterministic).
        """
        updated_models = models
        llm_hints = self.llm_scale_hints(updated_models, query)

        for model in updated_models:
            raw_size = model.get("size")
            if raw_size is None or len(raw_size) < 3:
                model["scale"] = float(model.get("scale", 1.0))
                continue

            sx = max(1e-6, float(raw_size[0]))
            sy = max(1e-6, float(raw_size[1]))
            sz = max(1e-6, float(raw_size[2]))
            max_dim = max(sx, sy, sz)
            model_name = str(model.get("Model") or model.get("name") or "")

            up_axis = str(model.get("_up_axis", "y"))

            # Step 1: unit detection (same as before — based on raw max extent)
            unit_scale = self.infer_unit_scale(max_dim)

            # Step 2: height in metres after unit normalisation.
            # For Y-up models (standard GLB): mesh Y → scene Z after euler="90 0 yaw".
            # For Z-up models (some Objaverse assets): mesh Z → scene Z, no rotation.
            if up_axis == "z":
                height_raw = sz if sz >= 0.05 * max_dim else max_dim
            elif sy >= 0.05 * max_dim:
                height_raw = sy
            else:
                height_raw = max_dim  # atypical orientation; use max

            height_m = height_raw * unit_scale

            # Step 3: deterministic scale based on target height
            target_h = self.target_height_m(model_name)
            deterministic_scale = unit_scale * (target_h / max(1e-6, height_m))

            # Step 4: optional LLM refinement
            llm_scale = llm_hints.get(model_name.lower())
            if llm_scale is not None:
                lo = deterministic_scale / 8.0
                hi = deterministic_scale * 8.0
                llm_scale = max(lo, min(hi, llm_scale))
                final_scale = (deterministic_scale * llm_scale) ** 0.5
            else:
                final_scale = deterministic_scale

            # Step 5: clamp resulting height to acceptable bounds
            min_h, max_h = self.target_height_bounds_m(model_name)
            actual_h = height_raw * final_scale
            if actual_h < min_h and min_h > 0:
                final_scale *= min_h / max(1e-9, actual_h)
            elif actual_h > max_h and max_h > 0:
                final_scale *= max_h / max(1e-9, actual_h)

            final_scale = max(1e-4, min(1000.0, final_scale))

            # Step 6: clamp horizontal extent for flat/compact objects.
            lname_lower = model_name.lower()
            for h_keywords, max_horiz in self._MAX_HORIZ_M:
                if any(kw in lname_lower for kw in h_keywords):
                    actual_horiz = max(sx, sz) * final_scale
                    if actual_horiz > max_horiz:
                        final_scale *= max_horiz / actual_horiz
                    break

            final_scale = max(1e-4, min(1000.0, final_scale))
            model["scale"] = final_scale

            # Normalise size so that size[1] is ALWAYS the scene height.
            # Y-up (euler="90 0 yaw"): mesh-Y → scene-Z, so size[1]=mesh_Y=height ✓
            # Z-up (euler="0 0 yaw"):  mesh-Z → scene-Z, so swap Y↔Z in stored size.
            if up_axis == "z":
                # [mesh_X, mesh_Z, mesh_Y] → [scene_width, scene_height, scene_depth]
                model["size"] = [sx * final_scale, sz * final_scale, sy * final_scale]
                model["_height_m"] = sz * final_scale
            else:
                model["size"] = [sx * final_scale, sy * final_scale, sz * final_scale]
                model["_height_m"] = sy * final_scale

        return updated_models

    def format_models_for_scale_prompt(self, models: List[Dict]) -> List[Dict]:
        formatted_models = []
        for model in models:
            formatted_models.append({"Model": model["name"], "Size": model["size"]})
        return formatted_models

    def scale_models(self, models: List[Dict], scaled_models: List[Dict]) -> List[Dict]:
        # Create a dictionary to store unique scales by 'Model'
        unique_scale = {model["Model"]: model for model in scaled_models}
        updated_models = models

        for model in updated_models:
            model["size"] = model["size"] * unique_scale[model["Model"]]["Scale"]
            model["scale"] = unique_scale[model["Model"]]["Scale"]

        return updated_models

    def place_models(self, models: List[Dict], placed_models: List[Dict]) -> List[Dict]:
        updated_models = models
        for i, model in enumerate(placed_models):
            updated_models[i].update(model)
        return updated_models

    def format_models_for_place_prompt(
        self, full_placed_models: List[Dict]
    ) -> List[Dict]:
        models_for_placement = []
        for model in full_placed_models:
            models_for_placement.append({"Model": model["name"], "Size": model["size"]})
        return models_for_placement

    def create_main_root(
        self, full_placed_models: List[Dict], objects: Dict[str, str]
    ) -> ET.Element:
        main_root = ET.Element("mujoco", model="test")
        visual_count = 0
        collision_count = 0
        material_count = 0
        for i, _ in enumerate(full_placed_models):
            material_map = {}
            model = full_placed_models[i]

            path = self.create_model_path(model)
            model_loc = model.get("model_loc")

            # Use fallback cube if model is missing (model_loc is None or doesn't exist)
            if model_loc is None or not os.path.exists(str(model_loc)):
                model_name = model.get('Model', model.get('name', 'unknown'))
                print(f"Model '{model_name}' not found in local assets, using fallback cube.")
                fallback_xml = self.create_fallback_cube_xml(path, model)
                self.insert_include_tags(main_root, fallback_xml)
                continue

            mesh = self.load_and_scale_mesh(model)
            obj, data = self.export_mesh(mesh)

            obj_path = self.write_obj_file(obj, path, model)
            self.save_material_and_images(data, path)
            args = self.create_args(path, model)
            printed_output = self.process_obj_file(obj_path, args)
            saved_mjc_path = self.get_saved_mjc_path(path, model)
            self.copy_referenced_textures_to_model_dir(path, saved_mjc_path)
            if "Error compiling model" in printed_output:
                if not self.model_xml_compiles(saved_mjc_path):
                    print(
                        f"Error compiling model {model['Model']},"
                        " it will be replaced with fallback cube."
                    )
                    fallback_xml = self.create_fallback_cube_xml(path, model)
                    self.insert_include_tags(main_root, fallback_xml)
                    continue
            tree, root, included_tree, included_root = self.parse_xml(saved_mjc_path)

            self.modify_default_class_attributes(
                included_root, material_map, visual_count, collision_count
            )
            visual_count += 1
            collision_count += 1
            self.modify_body_tag(included_root, model)
            material_count = self.rewrite_material_name_and_references(
                included_root, material_map, material_count
            )
            material_count += 1
            self.write_modified_xml(included_tree, saved_mjc_path)
            self.insert_include_tags(main_root, saved_mjc_path)
        return main_root


    def create_room_bodies(
        self,
        main_worldbody: ET.Element,
        room_half_size: float = 5.0,
        wall_height: float = 3.0,
    ) -> Dict[str, ET.Element]:
        """Create wall bodies as named XML elements.

        Wall-mounted objects can be attached as children of these bodies
        (no joint = rigidly welded to the wall).

        Returns a dict: wall_name → body element.
        """
        rhs = room_half_size
        wall_t = 0.05
        wh2 = wall_height / 2.0

        walls = [
            # (name,  pos_x, pos_y, pos_z,  size_x, size_y, size_z)
            ("wall_north", 0.0,  rhs,  wh2,    rhs, wall_t, wh2),
            ("wall_south", 0.0, -rhs,  wh2,    rhs, wall_t, wh2),
            ("wall_east",  rhs,  0.0,  wh2, wall_t,    rhs, wh2),
            ("wall_west", -rhs,  0.0,  wh2, wall_t,    rhs, wh2),
        ]
        wall_bodies: Dict[str, ET.Element] = {}
        for name, px, py, pz, sx, sy, sz in walls:
            body = ET.SubElement(main_worldbody, "body",
                name=name,
                pos=f"{px} {py} {pz}",
            )
            # No joint → welded to worldbody (wall never moves)
            ET.SubElement(body, "geom",
                type="box",
                size=f"{sx} {sy} {sz}",
                rgba="0.85 0.82 0.78 1.0",
                condim="1",
                friction="0.7 0.005 0.0001",
            )
            wall_bodies[name] = body
        return wall_bodies

    def attach_to_wall(
        self,
        wall_body: ET.Element,
        model: Dict[str, Any],
        wall_side: str,
        room_half_size: float = 5.0,
    ) -> None:
        """Attach a wall-mounted model XML as a child of the wall body.

        The child body has NO joint → rigidly attached to the wall.
        Position is relative to the wall body's coordinate frame.
        """
        from creator.scene.physics_profile import get_physics_profile

        model_name = str(model.get("Model") or model.get("name") or "")
        profile = get_physics_profile(model_name)

        pose = model.get("Pose") or {}
        px = float(pose.get("x", 0.0))
        py = float(pose.get("y", 0.0))
        pz = float(pose.get("z", 1.4))
        size = model.get("size") or [0.3, 0.3, 0.3]
        half_width = float(size[0]) / 2.0
        half_height = float(size[1]) / 2.0 if len(size) > 1 else half_width
        half_depth = float(size[2]) / 2.0 if len(size) > 2 else half_width

        wall_t = 0.05
        wall_pos = (wall_body.get("pos") or "0 0 0").split()
        wall_center_z = float(wall_pos[2]) if len(wall_pos) == 3 else 0.0
        rel_z = pz - wall_center_z

        # Position of mounted object relative to wall body centre
        # For north/south wall: object sticks out in -Y direction (into room)
        # For east/west wall: object sticks out in -X direction
        if wall_side in ("north",):
            rel_x, rel_y = px, -wall_t - half_depth
        elif wall_side in ("south",):
            rel_x, rel_y = px, wall_t + half_depth
        elif wall_side in ("east",):
            rel_x, rel_y = -wall_t - half_depth, py
        else:  # west
            rel_x, rel_y = wall_t + half_depth, py

        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", model_name)
        save_fn = re.sub(r"[^a-zA-Z0-9_]", "_", str(model.get("save_fn") or "0"))
        body_name = f"wallobj_{safe_name}_{save_fn}"

        child_body = ET.SubElement(wall_body, "body",
            name=body_name,
            pos=f"{rel_x:.4f} {rel_y:.4f} {rel_z:.4f}",
        )
        # No joint: welded to wall

        friction_str = " ".join(str(v) for v in profile.friction)
        ET.SubElement(child_body, "geom",
            type="box",
            size=f"{half_width} {half_depth} {half_height}",
            rgba="0.7 0.6 0.5 1",
            friction=friction_str,
            density=str(profile.density),
            condim=str(profile.condim),
        )

    def build_constraint_graph(
        self,
        *,
        semantic_plan: Dict[str, Union[str, int, float, list, dict]],
        query: str,
    ) -> Dict[str, Union[str, List[Dict[str, Union[str, int, float, bool]]]]]:
        objects = semantic_plan.get("objects", []) if isinstance(semantic_plan, dict) else []
        nodes: List[Dict[str, Union[str, int]]] = []
        edges: List[Dict[str, Union[str, float, bool, list]]] = []

        name_to_ids: Dict[str, List[str]] = {}
        object_rows: List[Tuple[str, str, List[Dict[str, Union[str, int, float, bool, list]]]]] = []
        for idx, obj in enumerate(objects):
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("Model") or obj.get("name") or "").strip()
            if not name:
                continue
            node_id = f"obj_{idx}"
            name_to_ids.setdefault(name, []).append(node_id)
            constraints = (
                obj.get("constraints")
                if isinstance(obj.get("constraints"), list)
                else []
            )
            norm_constraints = [c for c in constraints if isinstance(c, dict)]
            object_rows.append((node_id, name, norm_constraints))
            nodes.append(
                {
                    "id": node_id,
                    "name": name,
                    "constraints_count": len(norm_constraints),
                }
            )

        rr_target_idx: Dict[str, int] = {}
        edge_signatures = set()
        for src_id, src_name, constraints in object_rows:
            for c in constraints:
                ctype = str(c.get("type", "")).strip()
                target_name = str(c.get("target", "")).strip()
                target_ids = name_to_ids.get(target_name, []) if target_name else []
                target_id = ""
                if target_ids:
                    start = rr_target_idx.get(target_name, 0)
                    idx_choice = start % len(target_ids)
                    if len(target_ids) > 1 and target_ids[idx_choice] == src_id:
                        idx_choice = (idx_choice + 1) % len(target_ids)
                    target_id = target_ids[idx_choice]
                    rr_target_idx[target_name] = start + 1

                edge: Dict[str, Union[str, float, bool, list]] = {
                    "from": src_id,
                    "from_name": src_name,
                    "type": ctype,
                    "hard": bool(c.get("hard", False)),
                    "weight": float(c.get("weight", 1.0)),
                }
                if "value" in c:
                    edge["value"] = str(c.get("value"))
                if "distance" in c:
                    edge["distance"] = c.get("distance")
                if target_id:
                    edge["to"] = target_id
                    edge["to_name"] = target_name
                else:
                    edge["to"] = ""
                    edge["to_name"] = target_name

                sig = (
                    edge["from"],
                    edge.get("to", ""),
                    edge.get("type", ""),
                    edge.get("weight", 1.0),
                    edge.get("hard", False),
                    edge.get("value", ""),
                    tuple(edge.get("distance", [])) if isinstance(edge.get("distance"), list) else (),
                    edge.get("to_name", ""),
                )
                if sig in edge_signatures:
                    continue
                edge_signatures.add(sig)
                edges.append(edge)

        return {"query": query, "nodes": nodes, "edges": edges, "raw_plan": semantic_plan}

    def save_constraint_graph(

        self,
        *,
        semantic_plan: Dict[str, Union[str, int, float, list, dict]],
        query: str,
        output_filename: str,
    ) -> str:
        graph = self.build_constraint_graph(semantic_plan=semantic_plan, query=query)
        project_root = Path(__file__).resolve().parents[2]
        out_path = project_root / output_filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        return str(out_path)

    def create_fallback_cube_xml(
        self,
        path: Path,
        model: Dict[str, Union[str, int, float]],
    ) -> Path:
        size = model.get("size")
        if size is None:
            width, depth, height = 0.5, 0.5, 0.5
        else:
            width = max(0.1, float(size[0]))
            height = max(0.1, float(size[1])) if len(size) > 1 else width
            depth = max(0.1, float(size[2])) if len(size) > 2 else width

        gx, gy, gz = width / 2.0, depth / 2.0, height / 2.0
        pose = model.get("Pose") or {"x": 0.0, "y": 0.0, "z": gz}
        px = float(pose.get("x", 0.0))
        py = float(pose.get("y", 0.0))
        pz = float(pose.get("z", gz))
        yaw_deg = float(model.get("yaw_deg", 0.0))
        model_name = str(model.get("Model") or model.get("name") or "fallback")
        safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", model_name)
        save_fn = re.sub(r"[^a-zA-Z0-9_]+", "_", str(model.get("save_fn") or "0"))
        unique_name = f"fallback_{safe_name}_{save_fn}"

        from creator.scene.physics_profile import get_physics_profile
        fallback_profile = get_physics_profile(str(model.get("Model") or model.get("name") or ""))
        friction_str = " ".join(str(v) for v in fallback_profile.friction)
        solref_str = " ".join(str(v) for v in fallback_profile.solref)
        solimp_str = " ".join(str(v) for v in fallback_profile.solimp)

        root = ET.Element("mujoco", model=unique_name)
        worldbody = ET.SubElement(root, "worldbody")
        body = ET.SubElement(
            worldbody,
            "body",
            name=unique_name,
            pos=f"{px} {py} {pz}",
            euler=f"0 0 {yaw_deg}",
        )
        if not fallback_profile.is_static:
            ET.SubElement(body, "joint", type="free", damping="0.01", stiffness="0")
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=f"{gx} {gy} {gz}",
            rgba="0.85 0.2 0.2 1",
            friction=friction_str,
            condim=str(fallback_profile.condim),
            density=str(fallback_profile.density),
            solref=solref_str,
            solimp=solimp_str,
        )

        fallback_xml_path = Path(os.path.abspath(str(path) + f"/{model['save_fn']}_fallback.xml"))
        ET.ElementTree(root).write(fallback_xml_path, encoding="utf-8", xml_declaration=True)
        return fallback_xml_path

    def load_and_scale_mesh(
        self, model: Dict[str, Union[str, int, float]]
    ) -> trimesh.Trimesh:
        mesh = trimesh.load(model["model_loc"], force="mesh")
        mesh.apply_scale(float(model.get("scale", 1.0)))

        # Normalize mesh origin to bbox center so body pose z=height/2 really
        # means floor contact and not floating/intersecting due mesh offsets.
        bmin, bmax = mesh.bounds
        cx = 0.5 * (float(bmin[0]) + float(bmax[0]))
        cy = 0.5 * (float(bmin[1]) + float(bmax[1]))
        cz = 0.5 * (float(bmin[2]) + float(bmax[2]))
        if all(math.isfinite(v) for v in (cx, cy, cz)):
            mesh.apply_translation([-cx, -cy, -cz])

        return mesh

    def export_mesh(self, mesh: trimesh.Trimesh) -> Tuple[str, Dict[str, bytes]]:
        return trimesh.exchange.export.export_obj(
            mesh, include_texture=True, return_texture=True
        )

    def create_model_path(self, model: Dict[str, Union[str, int, float]]) -> Path:
        path = os.path.join(self.cache.cache_path, f"./converted/{model['save_fn']}")
        path = os.path.abspath(path)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_obj_file(
        self, obj: str, path: Path, model: Dict[str, Union[str, int, float]]
    ) -> str:
        obj_path = f"{path}/{model['save_fn']}.obj"
        with open(obj_path, "w") as f:
            f.write(obj)
        return obj_path

    def save_material_and_images(self, data: Dict[str, bytes], path: Path) -> None:
        for k, v in data.items():
            target_path = path / k
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "wb") as f:
                f.write(v)

    def copy_referenced_textures_to_model_dir(
        self,
        source_dir: Path,
        saved_mjc_path: Path,
    ) -> None:
        if not saved_mjc_path.exists():
            return

        try:
            xml_root = ET.parse(saved_mjc_path).getroot()
        except ET.ParseError:
            return

        target_dir = saved_mjc_path.parent
        seen_files: Set[str] = set()

        for texture in xml_root.findall(".//texture"):
            raw_file = (texture.get("file") or "").strip()
            if not raw_file:
                continue

            texture_rel_path = raw_file.replace("\\", "/")
            if texture_rel_path in seen_files:
                continue
            seen_files.add(texture_rel_path)

            source_path = source_dir / texture_rel_path
            if not source_path.exists():
                # Some exporters flatten texture file names in the source folder.
                source_path = source_dir / Path(texture_rel_path).name
            if not source_path.exists() or not source_path.is_file():
                continue

            destination_path = target_dir / texture_rel_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.resolve() == destination_path.resolve():
                continue
            shutil.copy2(source_path, destination_path)

    def _needs_convex_decomposition(self, model: Optional[Dict[str, Any]]) -> bool:
        if not model:
            return False
        parts: List[str] = []
        name = str(model.get("Model") or model.get("name") or "")
        if name:
            parts.append(name)
        for key in ("categories", "tags"):
            raw = model.get(key)
            if isinstance(raw, list):
                parts.extend(str(x) for x in raw if x)
        haystack = " ".join(parts).lower()
        return any(token in haystack for token in self._CONCAVE_COLLISION_HINTS)

    def create_args(
        self,
        path: Path,
        model: Optional[Dict[str, Any]] = None,
    ) -> Args:
        use_decompose = self._needs_convex_decomposition(model)
        kwargs: Dict[str, Any] = {
            "obj_dir": path,
            "verbose": True,
            "save_mjcf": True,
            "compile_model": True,
            "overwrite": True,
            "decompose": use_decompose,
        }
        if use_decompose:
            kwargs["coacd_args"] = CoacdArgs(
                preprocess_resolution=30,
                threshold=0.08,
                max_convex_hull=24,
                mcts_iterations=60,
                mcts_max_depth=3,
                mcts_nodes=16,
                resolution=800,
                pca=False,
                seed=0,
            )
        return Args(**kwargs)

    def process_obj_file(self, obj_path: str, args: Args) -> str:
        sys.stdout = StringIO()
        process_obj(Path(obj_path), args)
        printed_output = sys.stdout.getvalue()
        sys.stdout = sys.__stdout__
        return printed_output

    def model_xml_compiles(self, xml_path: Path) -> bool:
        if not xml_path.exists():
            return False
        try:
            import mujoco  # type: ignore

            mujoco.MjModel.from_xml_path(str(xml_path))
            return True
        except ImportError:
            # Keep non-mujoco environments usable; final scene compile check is separate.
            return True
        except Exception:
            return False

    def get_saved_mjc_path(
        self, path: Path, model: Dict[str, Union[str, int, float]]
    ) -> Path:
        return Path(
            os.path.abspath(
                str(path) + f"/{model['save_fn']}" + f"/{model['save_fn']}.xml"
            )
        )

    def parse_xml(
        self, saved_mjc_path: Path
    ) -> Tuple[ET.ElementTree, ET.Element, ET.ElementTree, ET.Element]:
        tree = ET.parse(saved_mjc_path)
        root = tree.getroot()
        included_tree = ET.parse(saved_mjc_path)
        included_root = included_tree.getroot()
        return tree, root, included_tree, included_root

    def modify_default_class_attributes(
        self,
        included_root: ET.Element,
        material_map: Dict[str, str],
        visual_count: int,
        collision_count: int,
    ) -> None:
        for default in included_root.findall(".//default"):
            class_attribute = default.get("class")
            if class_attribute == "visual":
                material_map[class_attribute] = f"visual{visual_count}"
                default.set("class", material_map[class_attribute])

            elif class_attribute == "collision":
                material_map[class_attribute] = f"collision{visual_count}"
                default.set("class", material_map[class_attribute])

    def modify_body_tag(
        self, included_root: ET.Element, model: Dict[str, Union[str, int, float]]
    ) -> None:
        from creator.scene.physics_profile import get_physics_profile

        yaw_deg = float(model.get("yaw_deg", 0.0))
        pose = model.get("Pose") or {"x": 0.0, "y": 0.0, "z": 0.0}
        px = float(pose.get("x", 0.0))
        py = float(pose.get("y", 0.0))
        pz = float(pose.get("z", 0.0))
        model_name = str(model.get("Model") or model.get("name") or "")
        profile = get_physics_profile(model_name)

        # MuJoCo uses XYZ-intrinsic Tait-Bryan euler. The third angle rotates
        # around the body's already-rotated Z axis, not the world Z.
        #
        # Y-up GLB: we want a final rotation = Rx(90°) · Rz(yaw_world). The
        # equivalent intrinsic euler is "90 yaw 0" — the second angle becomes
        # a rotation around the body's Y axis which, after the preceding
        # Rx(90°), points along world-Z (i.e. yaw around world vertical).
        # Using "90 0 yaw" instead would tip the table over because by then
        # the body-Z axis is horizontal.
        #
        # Z-up models stay upright; "0 0 yaw" rotates around world-Z directly.
        up_axis = str(model.get("_up_axis", "y"))
        euler_str = f"0 0 {yaw_deg}" if up_axis == "z" else f"90 {yaw_deg} 0"

        for body in included_root.findall(".//body"):
            body.set("pos", f"{px} {py} {pz}")
            body.set("euler", euler_str)

            if not profile.is_static:
                # Dynamic small objects: freejoint so they can fall/be picked up
                ET.SubElement(body, "joint", type="free",
                               damping="0.01", stiffness="0")
            # Static objects: NO joint → welded to worldbody, never flies off.

            # Apply physics params to collision geoms (skip visual-only geoms)
            friction_str = " ".join(str(v) for v in profile.friction)
            solref_str = " ".join(str(v) for v in profile.solref)
            solimp_str = " ".join(str(v) for v in profile.solimp)
            for geom in body.findall(".//geom"):
                g_class = geom.get("class", "")
                if "visual" in g_class:
                    continue  # visual-only geom, skip physics
                geom.set("friction", friction_str)
                geom.set("condim", str(profile.condim))
                geom.set("density", str(profile.density))
                geom.set("solref", solref_str)
                geom.set("solimp", solimp_str)

    def rewrite_material_name_and_references(
        self,
        included_root: ET.Element,
        material_map: Dict[str, str],
        material_count: int,
    ) -> int:
        materials = included_root.findall(".//material")
        for texture in included_root.findall(".//texture"):
            old_name = texture.get("name")
            if not old_name:
                continue
            material_map[old_name] = f"material_{material_count}"
            texture.set("name", material_map[old_name])
            material_count += 1

        for material in materials:
            old_name = material.get("name")
            if not old_name:
                continue
            if old_name not in material_map:
                material_map[old_name] = f"material_{material_count}"
                material_count += 1
            material.set("name", material_map[old_name])

            texture_name = material.get("texture")
            if texture_name and texture_name in material_map:
                material.set("texture", material_map[texture_name])

        for geom in included_root.findall(".//geom"):
            material = geom.get("material")
            if material and material in material_map:
                geom.set("material", material_map[material])

            class_ = geom.get("class")
            if class_ and class_ in material_map:
                geom.set("class", material_map[class_])
        return material_count

    def write_modified_xml(
        self, included_tree: ET.ElementTree, saved_mjc_path: Path
    ) -> None:
        included_tree.write(saved_mjc_path)

    def insert_include_tags(self, main_root: ET.Element, saved_mjc_path: Path) -> None:
        include = ET.SubElement(main_root, "include", file=str(saved_mjc_path))
        include.tail = "\n"

    def create_tree(self, main_root: ET.Element, room_half_size: float = 5.0) -> ET.ElementTree:
        tree = ET.ElementTree(main_root)

        # --- Physics solver options ---
        # timestep=0.002s is a good balance between stability and speed.
        # iterations=50 gives robust contact resolution.
        # integrator=implicitfast is recommended for scenes with many contacts.
        option = ET.SubElement(main_root, "option",
            timestep="0.002",
            gravity="0 0 -9.81",
            iterations="50",
            tolerance="1e-10",
            integrator="implicitfast",
        )
        option.tail = "\n"

        # --- Global defaults ---
        # These apply to all geoms/joints unless overridden per-object.
        default = ET.SubElement(main_root, "default")
        default.tail = "\n"
        default_geom = ET.SubElement(default, "geom",
            condim="3",
            friction="0.8 0.005 0.0001",
            solref="0.01 1",
            solimp="0.95 0.99 0.001 0.5 2",
            density="600",
        )
        default_geom.tail = "\n"

        env_xml_elements = [ET.Element("asset"), ET.Element("worldbody")]
        self.add_asset_elements(env_xml_elements)
        self.add_worldbody_elements(env_xml_elements, room_half_size=room_half_size)
        main_root.extend(env_xml_elements)
        return tree

    def add_asset_elements(
        self, asset_xml_elements: List[ET.Element]
    ) -> List[ET.Element]:
        asset_elements = [
            ET.SubElement(
                asset_xml_elements[0],
                "texture",
                type="skybox",
                builtin="gradient",
                rgb1=".3 .5 .7",
                rgb2="0 0 0",
                width="32",
                height="512",
            ),
            ET.SubElement(
                asset_xml_elements[0],
                "texture",
                name="body",
                type="cube",
                builtin="flat",
                mark="cross",
                width="128",
                height="128",
                rgb1="0.8 0.6 0.4",
                rgb2="0.8 0.6 0.4",
                markrgb="1 1 1",
                random="0.01",
            ),
            ET.SubElement(
                asset_xml_elements[0],
                "texture",
                name="grid",
                type="2d",
                builtin="checker",
                width="512",
                height="512",
                rgb1=".1 .2 .3",
                rgb2=".2 .3 .4",
            ),
            ET.SubElement(
                asset_xml_elements[0],
                "material",
                name="grid",
                texture="grid",
                texrepeat="1 1",
                texuniform="true",
                reflectance=".2",
            ),
        ]
        for elem in asset_elements:
            elem.tail = "\n"
        return asset_elements

    def add_worldbody_elements(
        self, asset_xml_elements: List[ET.Element], room_half_size: float = 5.0
    ) -> List[ET.Element]:
        worldbody = asset_xml_elements[1]
        rhs = room_half_size
        wall_h = 3.0   # room height in metres
        wall_t = 0.05  # wall half-thickness

        elements: List[ET.Element] = []

        # Floor — high friction, stiff contact
        floor = ET.SubElement(worldbody, "geom",
            name="floor",
            type="plane",
            size="0 0 0.05",
            material="grid",
            condim="3",
            friction="1.0 0.005 0.0001",
            solref="0.01 1",
            solimp="0.95 0.99 0.001 0.5 2",
        )
        floor.tail = "\n"
        elements.append(floor)

        # Room walls as bodies (allow wall-mounted objects to be children)
        wall_defs = [
            # (body_name, body_pos, geom_size, inward_facing_normal)
            ("wall_north", f"0 {rhs} {wall_h/2}",  f"{rhs} {wall_t} {wall_h/2}"),
            ("wall_south", f"0 {-rhs} {wall_h/2}", f"{rhs} {wall_t} {wall_h/2}"),
            ("wall_east",  f"{rhs} 0 {wall_h/2}",  f"{wall_t} {rhs} {wall_h/2}"),
            ("wall_west",  f"{-rhs} 0 {wall_h/2}", f"{wall_t} {rhs} {wall_h/2}"),
        ]
        for body_name, body_pos, geom_size in wall_defs:
            wall_body = ET.SubElement(worldbody, "body",
                name=body_name, pos=body_pos)
            wall_body.tail = "\n"
            # No joint on wall body → welded to worldbody
            wg = ET.SubElement(wall_body, "geom",
                type="box", size=geom_size,
                rgba="0.85 0.82 0.78 1.0",
                condim="1",
                contype="1", conaffinity="1",
                friction="0.7 0.005 0.0001",
            )
            wg.tail = "\n"
            elements.append(wall_body)

        # Ceiling (optional — blocks objects from flying upward)
        ceiling = ET.SubElement(worldbody, "geom",
            name="ceiling",
            type="plane",
            pos=f"0 0 {wall_h}",
            size="0 0 0.05",
            rgba="0.9 0.9 0.9 0.1",
            condim="1",
            contype="1", conaffinity="1",
        )
        ceiling.set("euler", "180 0 0")  # flip plane to face downward
        ceiling.tail = "\n"
        elements.append(ceiling)

        # Main ambient light
        light_main = ET.SubElement(worldbody, "light",
            name="light_main",
            pos="0 0 4",
            dir="0 0 -1",
            diffuse="0.8 0.8 0.8",
            specular="0.2 0.2 0.2",
            castshadow="true",
        )
        light_main.tail = "\n"
        elements.append(light_main)

        # Fill light (softer, no shadow)
        light_fill = ET.SubElement(worldbody, "light",
            name="light_fill",
            pos="0 0 6",
            diffuse="0.4 0.4 0.4",
            specular="0.0 0.0 0.0",
            castshadow="false",
        )
        light_fill.tail = "\n"
        elements.append(light_fill)

        return elements

    def render_preview(
        self,
        world_path: str,
        output_path: Optional[str] = None,
        *,
        width: int = 640,
        height: int = 480,
    ) -> Optional[str]:
        """Render the assembled MuJoCo scene and save a PNG preview.

        Returns the saved PNG path, or None if rendering failed.
        Saves next to the XML file as *_preview.png when output_path is None.
        """
        from creator.scene.vlm_validator import save_scene_preview
        return save_scene_preview(
            world_path,
            output_path,
            width=width,
            height=height,
        )

    def write_tree_to_file(self, tree: ET.ElementTree, path: str) -> None:
        tree.write(path, encoding="utf-8", xml_declaration=True)
