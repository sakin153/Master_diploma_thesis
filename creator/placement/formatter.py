"""Result Formatter for placement results.

This module provides formatting and export functionality for placement results
to various 3D formats including MuJoCo XML, USD, and glTF.
"""

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from creator.placement.scene_graph import SceneGraph, SceneNode
from creator.placement.geometry import Vec2, OBB


@dataclass
class PlacedObject:
    """Represents a placed object with position and orientation.

    Attributes:
        id: Unique identifier for the object
        model_name: Name of the 3D model
        position: (x, y, z) position in world coordinates
        orientation: (yaw, pitch, roll) in degrees
        scale: (sx, sy, sz) scale factors
        bounding_box: OBB for collision detection
        metadata: Additional object-specific data
    """
    id: str
    model_name: str
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    bounding_box: Optional[OBB] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlacementResult:
    """Complete placement result with all placed objects.

    Attributes:
        objects: List of placed objects
        scene_graph: Optional scene graph structure
        room_geometry: Room boundary and dimensions
        metadata: Additional scene metadata (timestamp, constraints, etc.)
    """
    objects: List[PlacedObject]
    scene_graph: Optional[SceneGraph] = None
    room_geometry: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResultFormatter:
    """Formatter for exporting placement results to various formats.

    This class provides methods to export placement results to:
    - MuJoCo XML format for physics simulation
    - USD (Universal Scene Description) for rendering and interchange
    - glTF format for web and real-time rendering
    """

    def __init__(self):
        """Initialize the formatter."""
        self._metadata_enabled = True

    def to_mujoco_xml(
        self,
        result: PlacementResult,
        output_path: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """Generate MuJoCo XML from placement result.

        Args:
            result: Placement result to export
            output_path: Optional path to write XML file
            include_metadata: Whether to include placement metadata as comments

        Returns:
            MuJoCo XML string

        Example:
            >>> formatter = ResultFormatter()
            >>> xml = formatter.to_mujoco_xml(placement_result)
            >>> # or save to file
            >>> formatter.to_mujoco_xml(placement_result, "scene.xml")
        """
        # Create root mujoco element
        mujoco = ET.Element("mujoco", model="generated_scene")

        # Add metadata as comment if enabled
        if include_metadata and self._metadata_enabled:
            metadata_comment = self._generate_metadata_comment(result.metadata)
            mujoco.append(ET.Comment(metadata_comment))

        # Add compiler settings
        compiler = ET.SubElement(mujoco, "compiler", angle="degree", coordinate="local")

        # Add option settings for physics
        option = ET.SubElement(mujoco, "option", timestep="0.002", gravity="0 0 -9.81")

        # Add visual settings
        visual = ET.SubElement(mujoco, "visual")
        ET.SubElement(visual, "headlight", ambient="0.5 0.5 0.5", diffuse="0.8 0.8 0.8")

        # Add asset section
        asset = ET.SubElement(mujoco, "asset")
        self._add_mujoco_assets(asset, result)

        # Add worldbody
        worldbody = ET.SubElement(mujoco, "worldbody")

        # Add ground plane
        ET.SubElement(
            worldbody,
            "geom",
            name="floor",
            type="plane",
            size="10 10 0.1",
            rgba="0.8 0.8 0.8 1"
        )

        # Add room boundaries if available
        if result.room_geometry:
            self._add_room_boundaries(worldbody, result.room_geometry)

        # Add placed objects
        for obj in result.objects:
            self._add_mujoco_body(worldbody, obj)

        # Convert to string with pretty formatting
        xml_str = self._prettify_xml(mujoco)

        # Write to file if path provided
        if output_path:
            Path(output_path).write_text(xml_str, encoding="utf-8")

        return xml_str

    def to_usd(
        self,
        result: PlacementResult,
        output_path: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """Export placement result to USD format.

        Args:
            result: Placement result to export
            output_path: Optional path to write USD file
            include_metadata: Whether to include placement metadata

        Returns:
            USD file content as string

        Note:
            This generates USDA (ASCII) format for readability.
            For production use, consider converting to USDC (binary).
        """
        lines = []

        # USD header
        lines.append("#usda 1.0")
        lines.append("(")
        lines.append('    defaultPrim = "Scene"')
        lines.append('    metersPerUnit = 1')
        lines.append('    upAxis = "Z"')
        lines.append(")")
        lines.append("")

        # Add metadata as custom attributes if enabled
        if include_metadata and self._metadata_enabled:
            lines.append("# Placement Metadata")
            for key, value in result.metadata.items():
                lines.append(f"# {key}: {value}")
            lines.append("")

        # Define scene root
        lines.append('def Xform "Scene" (')
        lines.append('    kind = "component"')
        lines.append(")")
        lines.append("{")

        # Add room if available
        if result.room_geometry:
            lines.extend(self._generate_usd_room(result.room_geometry, indent=1))

        # Add objects
        for obj in result.objects:
            lines.extend(self._generate_usd_object(obj, indent=1))

        lines.append("}")

        usd_content = "\n".join(lines)

        # Write to file if path provided
        if output_path:
            Path(output_path).write_text(usd_content, encoding="utf-8")

        return usd_content

    def to_gltf(
        self,
        result: PlacementResult,
        output_path: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """Export placement result to glTF format.

        Args:
            result: Placement result to export
            output_path: Optional path to write glTF file
            include_metadata: Whether to include placement metadata

        Returns:
            glTF JSON string

        Note:
            This generates glTF 2.0 JSON format.
            Mesh data should be provided separately or embedded as base64.
        """
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "UniversalPlacementSystem"
            },
            "scene": 0,
            "scenes": [
                {
                    "name": "Scene",
                    "nodes": []
                }
            ],
            "nodes": [],
            "meshes": []
        }

        # Add metadata as extras if enabled
        if include_metadata and self._metadata_enabled:
            gltf["asset"]["extras"] = {
                "placementMetadata": result.metadata
            }

        # Add room node if available
        if result.room_geometry:
            room_node_idx = len(gltf["nodes"])
            gltf["scenes"][0]["nodes"].append(room_node_idx)
            gltf["nodes"].append(self._generate_gltf_room_node(result.room_geometry))

        # Add object nodes
        for obj in result.objects:
            node_idx = len(gltf["nodes"])
            gltf["scenes"][0]["nodes"].append(node_idx)
            gltf["nodes"].append(self._generate_gltf_object_node(obj))

        gltf_json = json.dumps(gltf, indent=2)

        # Write to file if path provided
        if output_path:
            Path(output_path).write_text(gltf_json, encoding="utf-8")

        return gltf_json

    def serialize(
        self,
        result: PlacementResult,
        format: str = "json"
    ) -> str:
        """Serialize placement result to a generic format.

        Args:
            result: Placement result to serialize
            format: Output format ("json" or "dict")

        Returns:
            Serialized result as string or dict
        """
        data = {
            "objects": [
                {
                    "id": obj.id,
                    "model_name": obj.model_name,
                    "position": obj.position,
                    "orientation": obj.orientation,
                    "scale": obj.scale,
                    "metadata": obj.metadata
                }
                for obj in result.objects
            ],
            "room_geometry": result.room_geometry,
            "metadata": result.metadata
        }

        if format == "json":
            return json.dumps(data, indent=2)
        else:
            return data

    # Private helper methods

    def _generate_metadata_comment(self, metadata: Dict[str, Any]) -> str:
        """Generate metadata comment for XML."""
        lines = ["\nPlacement Metadata:"]
        lines.append(f"Generated: {datetime.now().isoformat()}")

        for key, value in metadata.items():
            lines.append(f"{key}: {value}")

        return "\n".join(lines) + "\n"

    def _add_mujoco_assets(self, asset_elem: ET.Element, result: PlacementResult) -> None:
        """Add asset definitions to MuJoCo XML."""
        # Add textures and materials
        ET.SubElement(
            asset_elem,
            "texture",
            name="grid",
            type="2d",
            builtin="checker",
            width="512",
            height="512",
            rgb1="0.8 0.8 0.8",
            rgb2="0.9 0.9 0.9"
        )

        ET.SubElement(
            asset_elem,
            "material",
            name="grid_material",
            texture="grid",
            texrepeat="10 10"
        )

        # Add meshes for each unique model
        seen_models = set()
        for obj in result.objects:
            if obj.model_name not in seen_models:
                # Reference mesh file (assumes meshes are in assets directory)
                mesh_file = f"assets/{obj.model_name}/{obj.model_name}.obj"
                ET.SubElement(
                    asset_elem,
                    "mesh",
                    name=f"mesh_{obj.model_name}",
                    file=mesh_file
                )
                seen_models.add(obj.model_name)

    def _add_room_boundaries(
        self,
        worldbody: ET.Element,
        room_geometry: Dict[str, Any]
    ) -> None:
        """Add room boundary walls to MuJoCo worldbody."""
        # Extract room dimensions
        half_size = room_geometry.get("half_size", 5.0)
        height = room_geometry.get("height", 3.0)

        # Add four walls
        walls = [
            ("north_wall", (0, half_size, height/2), (half_size, 0.05, height/2)),
            ("south_wall", (0, -half_size, height/2), (half_size, 0.05, height/2)),
            ("east_wall", (half_size, 0, height/2), (0.05, half_size, height/2)),
            ("west_wall", (-half_size, 0, height/2), (0.05, half_size, height/2))
        ]

        for name, pos, size in walls:
            ET.SubElement(
                worldbody,
                "geom",
                name=name,
                type="box",
                pos=f"{pos[0]} {pos[1]} {pos[2]}",
                size=f"{size[0]} {size[1]} {size[2]}",
                rgba="0.9 0.9 0.9 1"
            )

    def _add_mujoco_body(self, worldbody: ET.Element, obj: PlacedObject) -> None:
        """Add a body element for a placed object."""
        # Create body element
        body = ET.SubElement(worldbody, "body", name=obj.id)

        # Set position
        pos = f"{obj.position[0]} {obj.position[1]} {obj.position[2]}"
        body.set("pos", pos)

        # Set orientation (euler angles in degrees)
        euler = f"{obj.orientation[0]} {obj.orientation[1]} {obj.orientation[2]}"
        body.set("euler", euler)

        # Add geom with mesh
        geom = ET.SubElement(
            body,
            "geom",
            type="mesh",
            mesh=f"mesh_{obj.model_name}",
            rgba="0.7 0.7 0.7 1"
        )

        # Add scale if not default
        if obj.scale != (1.0, 1.0, 1.0):
            geom.set("scale", f"{obj.scale[0]} {obj.scale[1]} {obj.scale[2]}")

        # Add metadata as user data if available
        if obj.metadata:
            for key, value in obj.metadata.items():
                ET.SubElement(body, "user", name=key, value=str(value))

    def _prettify_xml(self, elem: ET.Element) -> str:
        """Return a pretty-printed XML string."""
        from xml.dom import minidom

        rough_string = ET.tostring(elem, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def _generate_usd_room(
        self,
        room_geometry: Dict[str, Any],
        indent: int = 0
    ) -> List[str]:
        """Generate USD code for room boundaries."""
        lines = []
        ind = "    " * indent

        half_size = room_geometry.get("half_size", 5.0)
        height = room_geometry.get("height", 3.0)

        lines.append(f'{ind}def Xform "Room"')
        lines.append(f"{ind}{{")

        # Floor
        lines.append(f'{ind}    def Mesh "Floor"')
        lines.append(f"{ind}    {{")
        lines.append(f'{ind}        float3[] extent = [({-half_size}, {-half_size}, 0), ({half_size}, {half_size}, 0)]')
        lines.append(f"{ind}    }}")

        # Walls (simplified representation)
        lines.append(f'{ind}    def Xform "Walls"')
        lines.append(f"{ind}    {{")
        lines.append(f'{ind}        # Wall geometry would be defined here')
        lines.append(f"{ind}    }}")

        lines.append(f"{ind}}}")

        return lines

    def _generate_usd_object(
        self,
        obj: PlacedObject,
        indent: int = 0
    ) -> List[str]:
        """Generate USD code for a placed object."""
        lines = []
        ind = "    " * indent

        # Sanitize object ID for USD
        usd_id = obj.id.replace("-", "_").replace(" ", "_")

        lines.append(f'{ind}def Xform "{usd_id}"')
        lines.append(f"{ind}{{")

        # Position
        lines.append(f'{ind}    double3 xformOp:translate = ({obj.position[0]}, {obj.position[1]}, {obj.position[2]})')

        # Rotation (convert to quaternion or euler as needed)
        lines.append(f'{ind}    float3 xformOp:rotateXYZ = ({obj.orientation[0]}, {obj.orientation[1]}, {obj.orientation[2]})')

        # Scale
        lines.append(f'{ind}    float3 xformOp:scale = ({obj.scale[0]}, {obj.scale[1]}, {obj.scale[2]})')

        lines.append(f'{ind}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]')

        # Reference to mesh (would be actual mesh path in production)
        lines.append(f'{ind}    # Mesh reference: {obj.model_name}')

        # Metadata as custom attributes
        if obj.metadata:
            lines.append(f'{ind}    # Metadata:')
            for key, value in obj.metadata.items():
                lines.append(f'{ind}    # {key}: {value}')

        lines.append(f"{ind}}}")

        return lines

    def _generate_gltf_room_node(self, room_geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Generate glTF node for room."""
        return {
            "name": "Room",
            "children": [],
            "extras": {
                "room_geometry": room_geometry
            }
        }

    def _generate_gltf_object_node(self, obj: PlacedObject) -> Dict[str, Any]:
        """Generate glTF node for a placed object."""
        node = {
            "name": obj.id,
            "translation": list(obj.position),
            "rotation": self._euler_to_quaternion(obj.orientation),
            "scale": list(obj.scale)
        }

        # Add metadata as extras
        if obj.metadata:
            node["extras"] = obj.metadata

        return node

    def _euler_to_quaternion(self, euler: Tuple[float, float, float]) -> List[float]:
        """Convert Euler angles (degrees) to quaternion [x, y, z, w].

        Args:
            euler: (yaw, pitch, roll) in degrees

        Returns:
            Quaternion as [x, y, z, w]
        """
        import math

        # Convert to radians
        yaw = math.radians(euler[0])
        pitch = math.radians(euler[1])
        roll = math.radians(euler[2])

        # Compute quaternion components
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return [x, y, z, w]
