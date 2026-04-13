from mujoco_scene_editor.inventory.objaverse_prompt import match_lvis_labels


def test_match_lvis_labels_basic():
    labels = ["coffee_mug", "dining_table", "tennis_racket"]
    out = match_lvis_labels(labels, ["mug", "table"], max_labels=2)
    assert "coffee_mug" in out
    assert "dining_table" in out


def test_match_lvis_labels_empty_keywords():
    labels = ["coffee_mug"]
    assert match_lvis_labels(labels, [], max_labels=3) == []


def test_match_lvis_labels_prefers_keyword_coverage():
    labels = [
        "chair",
        "dining_chair",
        "coffee_table",
        "table_lamp",
        "laptop_computer",
        "mug",
    ]

    out = match_lvis_labels(
        labels,
        ["chair", "table", "laptop", "mug"],
        max_labels=4,
    )

    assert "chair" in out
    assert "coffee_table" in out
    assert "laptop_computer" in out
    assert "mug" in out
