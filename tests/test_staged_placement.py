from __future__ import annotations

from mujoco_scene_editor.pipeline.scene_graph import (
    GlobalConstraints,
    RelationType,
    SceneGraph,
    SceneObject,
    SceneRelation,
)
from mujoco_scene_editor.pipeline.staged_placement import PlacementConfig, scene_graph_to_mjcf_staged
from mujoco_scene_editor.utils.mjcf_verify import verify_mjcf_physics


def test_scene_graph_to_mjcf_staged_verifies(tmp_path):
    graph = SceneGraph(
        scene_type="office",
        constraints=GlobalConstraints(workspace_height_m=0.75),
        objects=[
            SceneObject(id="desk1", class_name="desk", movable=False, size_m=[1.2, 0.8, 0.75]),
            SceneObject(id="mug1", class_name="mug", movable=True, size_m=[0.09, 0.09, 0.11]),
        ],
        relations=[
            SceneRelation(type=RelationType.on, subject="mug1", object="desk1"),
        ],
    )

    xml = scene_graph_to_mjcf_staged(
        graph,
        mjcf_dir=tmp_path,
        config=PlacementConfig(seed=0, attempts_per_object=80),
        replace_placeholders_with_hipoly=False,
    )

    res = verify_mjcf_physics(xml, mjcf_dir=tmp_path, steps=120)
    assert res.ok, f"verification failed: {res.errors} / {res.warnings}"


def test_scene_graph_to_mjcf_staged_adds_hipoly_visual(tmp_path):
    # Minimal valid OBJ mesh.
    # MuJoCo requires at least 4 vertices for mesh assets.
    (tmp_path / "mug.obj").write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "f 1 2 3",
                "f 1 2 4",
            ]
        ),
        encoding="utf-8",
    )

    graph = SceneGraph(
        scene_type="desk",
        objects=[
            SceneObject(id="desk1", class_name="desk", movable=False, size_m=[1.2, 0.8, 0.75]),
            SceneObject(
                id="mug1",
                class_name="mug",
                movable=True,
                size_m=[0.09, 0.09, 0.11],
                mesh_file="mug.obj",
                mesh_scale=0.05,
            ),
        ],
        relations=[
            SceneRelation(type=RelationType.on, subject="mug1", object="desk1"),
        ],
    )

    xml = scene_graph_to_mjcf_staged(
        graph,
        mjcf_dir=tmp_path,
        config=PlacementConfig(seed=0, attempts_per_object=80),
        replace_placeholders_with_hipoly=True,
    )

    assert "mesh_mug1" in xml
    assert "mug.obj" in xml
    assert "mug1_hipoly_vis" in xml
    assert "mug1_placeholder_col" in xml

    res = verify_mjcf_physics(xml, mjcf_dir=tmp_path, steps=50)
    assert res.ok, f"verification failed: {res.errors} / {res.warnings}"
