import copy
import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import objaverse
import trimesh
from obj2mjcf.cli import Args, process_obj

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
    _REALISTIC_MAX_DIM_HINTS: Tuple[Tuple[Tuple[str, ...], float], ...] = (
        (("chair", "stool", "armchair"), 1.0),
        (("table", "desk"), 1.6),
        (("sofa", "couch"), 2.4),
        (("bed", "mattress"), 2.1),
        (("cabinet", "wardrobe", "closet"), 2.2),
        (("shelf", "bookcase", "rack"), 2.0),
        (("tv", "monitor", "screen"), 1.4),
        (("lamp", "chandelier"), 1.8),
        (("plant", "vase"), 1.2),
        (("door",), 2.1),
        (("window",), 1.6),
        (("car", "vehicle"), 4.5),
        (("bottle",), 0.32),
        (("cup", "mug", "glass"), 0.14),
        (("plate", "dish", "bowl"), 0.3),
        (("phone",), 0.2),
        (("laptop", "notebook"), 0.4),
        (("book",), 0.28),
    )

    def __init__(self, chosen_model: str) -> None:
        super().__init__(chosen_model)
        self.cache = Cache()

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
    ) -> List[Dict]:
        # Actually methods are badly designed here. TODO is to refactor this.
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
        # 1) LLM semantic plan
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

        # 2) Deterministic geometry: floor -> wall -> small objects
        full_placed_models = solve_floor_placements(
            full_placed_models=full_placed_models,
            semantic_plan=semantic_plan,
            room_half_size=5.0,
            grid_step=0.8,
            yaw_candidates_deg=(0.0, 90.0, 180.0, 270.0),
            beam_width=12,
        )
        full_placed_models = solve_wall_placements(
            placed_models=full_placed_models,
            semantic_plan=semantic_plan,
            room_half_size=5.0,
        )
        full_placed_models = solve_small_object_placements(
            placed_models=full_placed_models,
            semantic_plan=semantic_plan,
            small_threshold_volume=0.06,
        )

        # 3) Physics-aware post-pass (repair invalid Z/XY placements)
        full_placed_models = validate_and_repair_layout(full_placed_models)

        # 3.5) Validate constraints against graph and run safe repair.
        violations_before = evaluate_constraint_violations(
            full_placed_models,
            semantic_plan,
            room_half_size=5.0,
        )
        violations_after = list(violations_before)
        accepted_repair = False
        if violations_before:
            repaired_candidate = repair_layout_by_constraints(
                full_placed_models,
                semantic_plan,
                room_half_size=5.0,
                iterations=2,
            )
            candidate_after = evaluate_constraint_violations(
                repaired_candidate,
                semantic_plan,
                room_half_size=5.0,
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
        tree = self.create_tree(main_root)
        self.write_tree_to_file(tree, path_to_save)
        self.try_compile_in_mujoco(path_to_save)

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

    def load_objects(self, full_placed_models: List[Dict]) -> Dict[str, str]:
        # TODO abstract dataset loader
        return objaverse.load_objects(
            uids=[entry["uuid"] for entry in full_placed_models]
        )

    def update_model_sizes(self, models: List[Dict]) -> List[Dict]:
        updated_models = models
        for i, model in enumerate(models):
            mesh = trimesh.load(model["model_loc"], force="mesh")
            updated_models[i]["size"] = mesh.extents
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
    def target_max_dimension_m(self, model_name: str) -> float:
        name = str(model_name or "").lower()
        for keywords, target in self._REALISTIC_MAX_DIM_HINTS:
            if any(keyword in name for keyword in keywords):
                return target
        return 1.0

    def target_dimension_bounds_m(self, model_name: str) -> Tuple[float, float]:
        target = self.target_max_dimension_m(model_name)
        min_dim = max(0.06, target * 0.35)
        max_dim = max(min_dim + 0.05, target * 1.5)
        return min_dim, max_dim

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

            unit_scale = self.infer_unit_scale(max_dim)
            normalized_max = max_dim * unit_scale
            model_name = str(model.get("Model") or model.get("name") or "")
            target_max = self.target_max_dimension_m(model_name)
            semantic_scale = target_max / max(1e-6, normalized_max)
            semantic_scale = max(0.25, min(4.0, semantic_scale))
            deterministic_scale = unit_scale * semantic_scale

            llm_scale = llm_hints.get(model_name.lower())
            if llm_scale is not None:
                lo = deterministic_scale / 8.0
                hi = deterministic_scale * 8.0
                llm_scale = max(lo, min(hi, llm_scale))
                final_scale = (deterministic_scale * llm_scale) ** 0.5
            else:
                final_scale = deterministic_scale

            min_dim_m, max_dim_m = self.target_dimension_bounds_m(model_name)
            scaled_max = max_dim * final_scale
            if scaled_max < min_dim_m:
                final_scale *= min_dim_m / max(1e-6, scaled_max)
            elif scaled_max > max_dim_m:
                final_scale *= max_dim_m / max(1e-6, scaled_max)

            final_scale = max(1e-4, min(500.0, final_scale))
            model["scale"] = final_scale
            model["size"] = [sx * final_scale, sy * final_scale, sz * final_scale]

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
            if not model.get("model_loc") or not os.path.exists(str(model.get("model_loc"))):
                print(
                    f"Model {model.get('Model', model.get('name', 'unknown'))} not found, "
                    "using fallback cube."
                )
                fallback_xml = self.create_fallback_cube_xml(path, model)
                self.insert_include_tags(main_root, fallback_xml)
                continue

            mesh = self.load_and_scale_mesh(model)
            obj, data = self.export_mesh(mesh)

            obj_path = self.write_obj_file(obj, path, model)
            self.save_material_and_images(data, path)
            args = self.create_args(path)
            printed_output = self.process_obj_file(obj_path, args)
            self.copy_images_to_nested_path(path, model)
            saved_mjc_path = self.get_saved_mjc_path(path, model)
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
            sx, sy, sz = 0.5, 0.5, 0.5
        else:
            sx = max(0.1, float(size[0]))
            sy = max(0.1, float(size[1]))
            sz = max(0.1, float(size[2]))

        gx, gy, gz = sx / 2.0, sy / 2.0, sz / 2.0
        pose = model.get("Pose") or {"x": 0.0, "y": 0.0, "z": gz}
        px = float(pose.get("x", 0.0))
        py = float(pose.get("y", 0.0))
        pz = float(pose.get("z", gz))
        yaw_deg = float(model.get("yaw_deg", 0.0))
        model_name = str(model.get("Model") or model.get("name") or "fallback")
        safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", model_name)
        save_fn = re.sub(r"[^a-zA-Z0-9_]+", "_", str(model.get("save_fn") or "0"))
        unique_name = f"fallback_{safe_name}_{save_fn}"

        root = ET.Element("mujoco", model=unique_name)
        worldbody = ET.SubElement(root, "worldbody")
        body = ET.SubElement(
            worldbody,
            "body",
            name=unique_name,
            pos=f"{px} {py} {pz}",
            euler=f"90 0 {yaw_deg}",
        )
        ET.SubElement(body, "joint", type="free")
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=f"{gx} {gy} {gz}",
            rgba="0.85 0.2 0.2 1",
        )

        fallback_xml_path = Path(os.path.abspath(str(path) + f"/{model['save_fn']}_fallback.xml"))
        ET.ElementTree(root).write(fallback_xml_path, encoding="utf-8", xml_declaration=True)
        return fallback_xml_path

    def load_and_scale_mesh(
        self, model: Dict[str, Union[str, int, float]]
    ) -> trimesh.Trimesh:
        mesh = trimesh.load(model["model_loc"], force="mesh")
        mesh.apply_scale(model["scale"])
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
            with open(os.path.join(path, k), "wb") as f:
                f.write(v)

    def copy_images_to_nested_path(
        self, path: Path, model: Dict[str, Union[str, int, float]]
    ) -> None:
        nested_path = Path(os.path.abspath(str(path) + f"/{model['save_fn']}"))
        nested_path.mkdir(parents=True, exist_ok=True)
        files = os.listdir(path)
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                source_path = os.path.join(path, file)
                destination_path = os.path.join(nested_path, file)
                shutil.copy(source_path, destination_path)

    def create_args(self, path: Path) -> Args:
        return Args(
            obj_dir=path,
            verbose=True,
            save_mjcf=True,
            compile_model=True,
            overwrite=True,
        )

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
        yaw_deg = float(model.get("yaw_deg", 0.0))
        for body in included_root.findall(".//body"):
            body.set("pos", " ".join(str(e) for e in model["Pose"].values()))
            body.set("euler", f"90 0 {yaw_deg}")
            ET.SubElement(body, "joint", type="free")

    def rewrite_material_name_and_references(
        self,
        included_root: ET.Element,
        material_map: Dict[str, str],
        material_count: int,
    ) -> int:
        materials = included_root.findall(".//material")
        for i, texture in enumerate(included_root.findall(".//texture")):
            old_name = texture.get("name")
            material_map[old_name] = f"material_{material_count}"
            texture.set("name", material_map[old_name])
            material_count += 1
        for i, material in enumerate(materials):
            old_name = material.get("name")
            if old_name not in material_map:
                material_map[old_name] = f"material_{material_count}"
                material_count += 1
            material.set("name", material_map[old_name])
            texture = material.get("texture")
            if texture:
                material.set("texture", material_map[old_name])
        for i, geom in enumerate(included_root.findall(".//geom")):
            material = geom.get("material")
            if material:
                geom.set("material", material_map[material])
            class_ = geom.get("class")
            if class_:
                geom.set("class", material_map[class_])
        return material_count

    def write_modified_xml(
        self, included_tree: ET.ElementTree, saved_mjc_path: Path
    ) -> None:
        included_tree.write(saved_mjc_path)

    def insert_include_tags(self, main_root: ET.Element, saved_mjc_path: Path) -> None:
        include = ET.SubElement(main_root, "include", file=str(saved_mjc_path))
        include.tail = "\n"

    def create_tree(self, main_root: ET.Element) -> ET.ElementTree:
        tree = ET.ElementTree(main_root)
        env_xml_elements = [ET.Element("asset"), ET.Element("worldbody")]
        self.add_asset_elements(env_xml_elements)
        self.add_worldbody_elements(env_xml_elements)
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
        self, asset_xml_elements: List[ET.Element]
    ) -> List[ET.Element]:
        worldbody_elements = [
            ET.SubElement(
                asset_xml_elements[1],
                "geom",
                name="floor",
                size="0 0 .05",
                type="plane",
                material="grid",
                condim="3",
            ),
            ET.SubElement(
                asset_xml_elements[1],
                "light",
                castshadow="false",
                pos="0 0 1000",
            ),
        ]
        for elem in worldbody_elements:
            elem.tail = "\n"
        return worldbody_elements

    def write_tree_to_file(self, tree: ET.ElementTree, path: str) -> None:
        tree.write(path, encoding="utf-8", xml_declaration=True)
