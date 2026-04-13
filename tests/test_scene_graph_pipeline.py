from __future__ import annotations

from mujoco_scene_editor.pipeline.graph_to_mjcf import scene_graph_to_mjcf
from mujoco_scene_editor.pipeline.scene_graph import (
    GlobalConstraints,
    RelationType,
    SceneGraph,
    SceneObject,
    SceneRelation,
)
from mujoco_scene_editor.pipeline.scene_graph_llm import parse_scene_graph_from_llm
from mujoco_scene_editor.utils.mjcf_verify import verify_mjcf_physics


def test_parse_scene_graph_from_llm_accepts_alias_class():
    txt = """
{
  "scene_type": "kitchen",
  "objects": [
    {"id": "table1", "class": "table", "movable": false, "affordances": []},
    {"id": "mug1", "class": "mug", "movable": true, "affordances": ["grasp"]}
  ],
  "relations": [
    {"type": "on", "subject": "mug1", "object": "table1", "distance_m": null}
  ],
  "constraints": {"keep_clear_radius_m": null, "workspace_height_m": 0.75},
  "llm_notes": null,
  "metadata": {}
}
""".strip()

    graph = parse_scene_graph_from_llm(txt)
    assert graph.scene_type == "kitchen"
    assert len(graph.objects) == 2
    assert graph.objects[0].class_name == "table"


def test_scene_graph_to_mjcf_verifies(tmp_path):
    graph = SceneGraph(
        scene_type="office",
        constraints=GlobalConstraints(workspace_height_m=0.75),
        objects=[
            SceneObject(id="desk1", class_name="desk", movable=False),
            SceneObject(id="laptop1", class_name="laptop", movable=True),
            SceneObject(id="mug1", class_name="mug", movable=True),
            SceneObject(id="bottle1", class_name="bottle", movable=True),
        ],
        relations=[
            SceneRelation(type=RelationType.on, subject="laptop1", object="desk1"),
            SceneRelation(type=RelationType.on, subject="mug1", object="desk1"),
            SceneRelation(type=RelationType.on, subject="bottle1", object="desk1"),
        ],
    )

    xml = scene_graph_to_mjcf(graph, seed=0)
    res = verify_mjcf_physics(xml, mjcf_dir=tmp_path, steps=120)
    assert res.ok, f"verification failed: {res.errors} / {res.warnings}"
