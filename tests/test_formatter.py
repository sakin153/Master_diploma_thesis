"""Unit tests for ResultFormatter.

Tests the formatting and export functionality for placement results
to MuJoCo XML, USD, and glTF formats.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from creator.placement.formatter import (
    ResultFormatter,
    PlacedObject,
    PlacementResult,
)
from creator.placement.geometry import Vec2, OBB


class TestResultFormatter:
    """Test suite for ResultFormatter class."""

    @pytest.fixture
    def formatter(self):
        """Create a ResultFormatter instance."""
        return ResultFormatter()

    @pytest.fixture
    def simple_placement_result(self):
        """Create a simple placement result for testing."""
        objects = [
            PlacedObject(
                id="table_1",
                model_name="table",
                position=(0.0, 0.0, 0.5),
                orientation=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
                metadata={"type": "furniture"}
            ),
            PlacedObject(
                id="chair_1",
                model_name="chair",
                position=(1.0, 0.0, 0.4),
                orientation=(90.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
                metadata={"type": "furniture"}
            )
        ]

        room_geometry = {
            "half_size": 5.0,
            "height": 3.0
        }

        metadata = {
            "scene_name": "test_scene",
            "num_objects": 2
        }

        return PlacementResult(
            objects=objects,
            room_geometry=room_geometry,
            metadata=metadata
        )

    def test_formatter_initialization(self, formatter):
        """Test that formatter initializes correctly."""
        assert formatter is not None
        assert formatter._metadata_enabled is True

    def test_to_mujoco_xml_basic(self, formatter, simple_placement_result):
        """Test basic MuJoCo XML generation."""
        xml_str = formatter.to_mujoco_xml(simple_placement_result)

        # Parse XML to verify structure
        root = ET.fromstring(xml_str)

        # Check root element
        assert root.tag == "mujoco"
        assert root.get("model") == "generated_scene"

        # Check for compiler element
        compiler = root.find("compiler")
        assert compiler is not None
        assert compiler.get("angle") == "degree"

        # Check for worldbody
        worldbody = root.find("worldbody")
        assert worldbody is not None

        # Check for floor
        floor = worldbody.find(".//geom[@name='floor']")
        assert floor is not None
        assert floor.get("type") == "plane"

    def test_to_mujoco_xml_objects(self, formatter, simple_placement_result):
        """Test that objects are correctly added to MuJoCo XML."""
        xml_str = formatter.to_mujoco_xml(simple_placement_result)
        root = ET.fromstring(xml_str)

        worldbody = root.find("worldbody")

        # Check for table body
        table_body = worldbody.find(".//body[@name='table_1']")
        assert table_body is not None
        assert table_body.get("pos") == "0.0 0.0 0.5"
        assert table_body.get("euler") == "0.0 0.0 0.0"

        # Check for chair body
        chair_body = worldbody.find(".//body[@name='chair_1']")
        assert chair_body is not None
        assert chair_body.get("pos") == "1.0 0.0 0.4"
        assert chair_body.get("euler") == "90.0 0.0 0.0"

    def test_to_mujoco_xml_room_boundaries(self, formatter, simple_placement_result):
        """Test that room boundaries are added to MuJoCo XML."""
        xml_str = formatter.to_mujoco_xml(simple_placement_result)
        root = ET.fromstring(xml_str)

        worldbody = root.find("worldbody")

        # Check for walls
        north_wall = worldbody.find(".//geom[@name='north_wall']")
        assert north_wall is not None
        assert north_wall.get("type") == "box"

        south_wall = worldbody.find(".//geom[@name='south_wall']")
        assert south_wall is not None

        east_wall = worldbody.find(".//geom[@name='east_wall']")
        assert east_wall is not None

        west_wall = worldbody.find(".//geom[@name='west_wall']")
        assert west_wall is not None

    def test_to_mujoco_xml_metadata(self, formatter, simple_placement_result):
        """Test that metadata is included in MuJoCo XML."""
        xml_str = formatter.to_mujoco_xml(simple_placement_result, include_metadata=True)

        # Check for metadata comment
        assert "Placement Metadata" in xml_str
        assert "scene_name" in xml_str
        assert "num_objects" in xml_str

    def test_to_mujoco_xml_no_metadata(self, formatter, simple_placement_result):
        """Test MuJoCo XML generation without metadata."""
        xml_str = formatter.to_mujoco_xml(simple_placement_result, include_metadata=False)

        # Metadata comment should not be present
        assert "Placement Metadata" not in xml_str

    def test_to_mujoco_xml_file_output(self, formatter, simple_placement_result, tmp_path):
        """Test writing MuJoCo XML to file."""
        output_file = tmp_path / "test_scene.xml"

        xml_str = formatter.to_mujoco_xml(
            simple_placement_result,
            output_path=str(output_file)
        )

        # Check file was created
        assert output_file.exists()

        # Check file content matches returned string
        file_content = output_file.read_text()
        assert file_content == xml_str

    def test_to_usd_basic(self, formatter, simple_placement_result):
        """Test basic USD generation."""
        usd_str = formatter.to_usd(simple_placement_result)

        # Check USD header
        assert "#usda 1.0" in usd_str
        assert 'defaultPrim = "Scene"' in usd_str
        assert 'upAxis = "Z"' in usd_str

        # Check scene definition
        assert 'def Xform "Scene"' in usd_str

    def test_to_usd_objects(self, formatter, simple_placement_result):
        """Test that objects are correctly added to USD."""
        usd_str = formatter.to_usd(simple_placement_result)

        # Check for table
        assert 'def Xform "table_1"' in usd_str
        assert "double3 xformOp:translate = (0.0, 0.0, 0.5)" in usd_str

        # Check for chair
        assert 'def Xform "chair_1"' in usd_str
        assert "double3 xformOp:translate = (1.0, 0.0, 0.4)" in usd_str

    def test_to_usd_metadata(self, formatter, simple_placement_result):
        """Test that metadata is included in USD."""
        usd_str = formatter.to_usd(simple_placement_result, include_metadata=True)

        # Check for metadata comments
        assert "# Placement Metadata" in usd_str
        assert "# scene_name:" in usd_str
        assert "# num_objects:" in usd_str

    def test_to_usd_file_output(self, formatter, simple_placement_result, tmp_path):
        """Test writing USD to file."""
        output_file = tmp_path / "test_scene.usda"

        usd_str = formatter.to_usd(
            simple_placement_result,
            output_path=str(output_file)
        )

        # Check file was created
        assert output_file.exists()

        # Check file content matches returned string
        file_content = output_file.read_text()
        assert file_content == usd_str

    def test_to_gltf_basic(self, formatter, simple_placement_result):
        """Test basic glTF generation."""
        gltf_str = formatter.to_gltf(simple_placement_result)

        # Parse JSON
        gltf = json.loads(gltf_str)

        # Check asset info
        assert gltf["asset"]["version"] == "2.0"
        assert gltf["asset"]["generator"] == "UniversalPlacementSystem"

        # Check scene structure
        assert "scenes" in gltf
        assert len(gltf["scenes"]) == 1
        assert gltf["scenes"][0]["name"] == "Scene"

        # Check nodes
        assert "nodes" in gltf
        assert len(gltf["nodes"]) > 0

    def test_to_gltf_objects(self, formatter, simple_placement_result):
        """Test that objects are correctly added to glTF."""
        gltf_str = formatter.to_gltf(simple_placement_result)
        gltf = json.loads(gltf_str)

        # Find table node
        table_node = None
        chair_node = None
        for node in gltf["nodes"]:
            if node["name"] == "table_1":
                table_node = node
            elif node["name"] == "chair_1":
                chair_node = node

        # Check table
        assert table_node is not None
        assert table_node["translation"] == [0.0, 0.0, 0.5]
        assert table_node["scale"] == [1.0, 1.0, 1.0]

        # Check chair
        assert chair_node is not None
        assert chair_node["translation"] == [1.0, 0.0, 0.4]

    def test_to_gltf_metadata(self, formatter, simple_placement_result):
        """Test that metadata is included in glTF."""
        gltf_str = formatter.to_gltf(simple_placement_result, include_metadata=True)
        gltf = json.loads(gltf_str)

        # Check for metadata in asset extras
        assert "extras" in gltf["asset"]
        assert "placementMetadata" in gltf["asset"]["extras"]
        assert gltf["asset"]["extras"]["placementMetadata"]["scene_name"] == "test_scene"

    def test_to_gltf_file_output(self, formatter, simple_placement_result, tmp_path):
        """Test writing glTF to file."""
        output_file = tmp_path / "test_scene.gltf"

        gltf_str = formatter.to_gltf(
            simple_placement_result,
            output_path=str(output_file)
        )

        # Check file was created
        assert output_file.exists()

        # Check file content matches returned string
        file_content = output_file.read_text()
        assert file_content == gltf_str

    def test_serialize_json(self, formatter, simple_placement_result):
        """Test serialization to JSON format."""
        json_str = formatter.serialize(simple_placement_result, format="json")

        # Parse JSON
        data = json.loads(json_str)

        # Check structure
        assert "objects" in data
        assert len(data["objects"]) == 2
        assert "room_geometry" in data
        assert "metadata" in data

        # Check object data
        assert data["objects"][0]["id"] == "table_1"
        assert data["objects"][0]["model_name"] == "table"
        assert data["objects"][0]["position"] == [0.0, 0.0, 0.5]

    def test_serialize_dict(self, formatter, simple_placement_result):
        """Test serialization to dict format."""
        data = formatter.serialize(simple_placement_result, format="dict")

        # Check structure
        assert isinstance(data, dict)
        assert "objects" in data
        assert len(data["objects"]) == 2

    def test_euler_to_quaternion(self, formatter):
        """Test Euler to quaternion conversion."""
        # Test identity rotation (0, 0, 0)
        quat = formatter._euler_to_quaternion((0.0, 0.0, 0.0))
        assert len(quat) == 4
        # Identity quaternion is [0, 0, 0, 1]
        assert abs(quat[3] - 1.0) < 1e-6
        assert abs(quat[0]) < 1e-6
        assert abs(quat[1]) < 1e-6
        assert abs(quat[2]) < 1e-6

        # Test 90 degree yaw rotation
        quat = formatter._euler_to_quaternion((90.0, 0.0, 0.0))
        assert len(quat) == 4
        # Should have non-zero z component
        assert abs(quat[2]) > 0.5

    def test_empty_placement_result(self, formatter):
        """Test formatting with empty placement result."""
        empty_result = PlacementResult(objects=[])

        # Should not raise errors
        xml_str = formatter.to_mujoco_xml(empty_result)
        assert xml_str is not None

        usd_str = formatter.to_usd(empty_result)
        assert usd_str is not None

        gltf_str = formatter.to_gltf(empty_result)
        assert gltf_str is not None

    def test_object_with_custom_scale(self, formatter):
        """Test formatting object with non-default scale."""
        obj = PlacedObject(
            id="scaled_obj",
            model_name="box",
            position=(0.0, 0.0, 0.0),
            scale=(2.0, 1.5, 1.0)
        )

        result = PlacementResult(objects=[obj])

        # Test MuJoCo XML
        xml_str = formatter.to_mujoco_xml(result)
        root = ET.fromstring(xml_str)
        body = root.find(".//body[@name='scaled_obj']")
        geom = body.find("geom")
        assert geom.get("scale") == "2.0 1.5 1.0"

        # Test glTF
        gltf_str = formatter.to_gltf(result)
        gltf = json.loads(gltf_str)
        node = next(n for n in gltf["nodes"] if n["name"] == "scaled_obj")
        assert node["scale"] == [2.0, 1.5, 1.0]

    def test_object_with_rotation(self, formatter):
        """Test formatting object with rotation."""
        obj = PlacedObject(
            id="rotated_obj",
            model_name="box",
            position=(0.0, 0.0, 0.0),
            orientation=(45.0, 30.0, 15.0)
        )

        result = PlacementResult(objects=[obj])

        # Test MuJoCo XML
        xml_str = formatter.to_mujoco_xml(result)
        root = ET.fromstring(xml_str)
        body = root.find(".//body[@name='rotated_obj']")
        assert body.get("euler") == "45.0 30.0 15.0"

        # Test USD
        usd_str = formatter.to_usd(result)
        assert "float3 xformOp:rotateXYZ = (45.0, 30.0, 15.0)" in usd_str

    def test_multiple_objects_same_model(self, formatter):
        """Test formatting multiple objects with the same model."""
        objects = [
            PlacedObject(
                id="chair_1",
                model_name="chair",
                position=(1.0, 0.0, 0.4)
            ),
            PlacedObject(
                id="chair_2",
                model_name="chair",
                position=(-1.0, 0.0, 0.4)
            ),
            PlacedObject(
                id="chair_3",
                model_name="chair",
                position=(0.0, 1.0, 0.4)
            )
        ]

        result = PlacementResult(objects=objects)

        # Test MuJoCo XML - should only define mesh once
        xml_str = formatter.to_mujoco_xml(result)
        root = ET.fromstring(xml_str)
        asset = root.find("asset")
        chair_meshes = asset.findall(".//mesh[@name='mesh_chair']")
        assert len(chair_meshes) == 1  # Only one mesh definition

        # But should have three bodies
        worldbody = root.find("worldbody")
        chair_bodies = [
            worldbody.find(".//body[@name='chair_1']"),
            worldbody.find(".//body[@name='chair_2']"),
            worldbody.find(".//body[@name='chair_3']")
        ]
        assert all(body is not None for body in chair_bodies)

    def test_room_without_geometry(self, formatter):
        """Test formatting when room geometry is not provided."""
        obj = PlacedObject(
            id="obj_1",
            model_name="box",
            position=(0.0, 0.0, 0.0)
        )

        result = PlacementResult(objects=[obj], room_geometry=None)

        # Should not raise errors
        xml_str = formatter.to_mujoco_xml(result)
        assert xml_str is not None

        # Should not have wall elements
        root = ET.fromstring(xml_str)
        worldbody = root.find("worldbody")
        walls = worldbody.findall(".//geom[@name='north_wall']")
        assert len(walls) == 0

    def test_object_metadata_preservation(self, formatter):
        """Test that object metadata is preserved in exports."""
        obj = PlacedObject(
            id="obj_1",
            model_name="box",
            position=(0.0, 0.0, 0.0),
            metadata={
                "category": "furniture",
                "weight": 10.5,
                "material": "wood"
            }
        )

        result = PlacementResult(objects=[obj])

        # Test JSON serialization
        json_str = formatter.serialize(result, format="json")
        data = json.loads(json_str)
        assert data["objects"][0]["metadata"]["category"] == "furniture"
        assert data["objects"][0]["metadata"]["weight"] == 10.5

        # Test glTF extras
        gltf_str = formatter.to_gltf(result)
        gltf = json.loads(gltf_str)
        node = gltf["nodes"][0]
        assert "extras" in node
        assert node["extras"]["category"] == "furniture"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
