"""Scene Assembly for Semantic Plan Enforcement.

This module implements the SceneAssembly class, which creates MuJoCo XML
from a PlacementSolution. It faithfully represents the placement solution
without modifications, ensuring all objects are included with exact positions
and orientations.

The SceneAssembly's role is to:
1. Take a PlacementSolution with all objects positioned
2. Create a valid MuJoCo XML scene file
3. Include ALL objects from the solution without omissions
4. Use exact positions and orientations from the solution
5. Verify model files exist before including them
6. Log errors for missing files but continue with other objects
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List
from xml.dom import minidom

from creator.placement.semantic_enforcement.models import PlacementSolution, PlacedObject

logger = logging.getLogger(__name__)


class SceneAssembly:
    """Assembles MuJoCo XML scene from placement solution.

    This class creates a complete MuJoCo XML file from a PlacementSolution,
    including floor, walls, and all placed objects with their exact positions
    and orientations.

    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8**
    """

    def __init__(self, workspace_root: str = "."):
        """Initialize SceneAssembly.

        Args:
            workspace_root: Root directory for resolving model_loc paths
        """
        self.workspace_root = Path(workspace_root)

    def assemble_scene(
        self,
        placement_solution: PlacementSolution,
        output_path: str = None,
    ) -> str:
        """Assemble MuJoCo XML from placement solution.

        Creates a complete MuJoCo XML scene with:
        - Floor plane
        - Room walls based on room_size
        - All objects from placement_solution with exact positions and orientations
        - Mesh assets in <asset> section
        - Body elements with geom references to meshes
        - Freejoint for dynamic objects

        Args:
            placement_solution: Placement solution with all objects positioned
            output_path: Optional path to write XML file

        Returns:
            MuJoCo XML string

        **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8**
        """
        logger.info(
            f"[SceneAssembly] Starting scene assembly for "
            f"{len(placement_solution.objects)} objects"
        )

        # Create root mujoco element
        mujoco = ET.Element("mujoco", model="semantic_scene")

        # Add compiler settings
        ET.SubElement(mujoco, "compiler", angle="degree", coordinate="local")

        # Add option settings for physics
        ET.SubElement(mujoco, "option", timestep="0.002", gravity="0 0 -9.81")

        # Add visual settings
        visual = ET.SubElement(mujoco, "visual")
        ET.SubElement(
            visual, "headlight", ambient="0.5 0.5 0.5", diffuse="0.8 0.8 0.8"
        )

        # Add asset section with mesh definitions
        asset = ET.SubElement(mujoco, "asset")
        self._add_mesh_assets(asset, placement_solution.objects)

        # Add worldbody
        worldbody = ET.SubElement(mujoco, "worldbody")

        # Add lighting
        ET.SubElement(worldbody, "light", pos="0 0 3", dir="0 0 -1")

        # Add floor
        self._add_floor(worldbody)

        # Add walls based on room_size - DISABLED (walls not needed in pipeline)
        # self._add_walls(worldbody, placement_solution.room_size)

        # Add all objects from placement solution
        objects_added = 0
        objects_skipped = 0

        for obj in placement_solution.objects:
            if self._add_object(worldbody, obj):
                objects_added += 1
            else:
                objects_skipped += 1

        logger.info(
            f"[SceneAssembly] Scene assembly completed. "
            f"Added {objects_added} objects, skipped {objects_skipped} objects."
        )

        # Convert to string with pretty formatting
        xml_str = self._prettify_xml(mujoco)

        # Write to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(xml_str, encoding="utf-8")
            logger.info(f"[SceneAssembly] Wrote XML to {output_path}")

        return xml_str

    def _add_mesh_assets(
        self, asset_elem: ET.Element, objects: List[PlacedObject]
    ) -> None:
        """Add mesh asset definitions for ALL objects.

        Each object gets its own mesh asset with unique name (mesh_{obj.id}),
        even if multiple objects share the same model file. This allows
        multiple instances of the same model in the scene.

        Args:
            asset_elem: MuJoCo asset element
            objects: List of placed objects
        """
        for obj in objects:
            # Verify model file exists
            model_path = self.workspace_root / obj.model_loc
            if not model_path.exists():
                logger.warning(
                    f"[SceneAssembly] Model file not found: {obj.model_loc}. "
                    f"Skipping mesh asset for {obj.id}"
                )
                continue

            # Add mesh asset with unique name based on object ID
            # Use absolute path to ensure MuJoCo can find the file
            mesh_file = str(model_path.resolve())
            ET.SubElement(
                asset_elem, "mesh", name=f"mesh_{obj.id}", file=mesh_file
            )

            logger.debug(f"[SceneAssembly] Added mesh asset: mesh_{obj.id} -> {mesh_file}")

    def _add_floor(self, worldbody: ET.Element) -> None:
        """Add floor plane to worldbody.

        Args:
            worldbody: MuJoCo worldbody element
        """
        ET.SubElement(
            worldbody,
            "geom",
            name="floor",
            type="plane",
            size="10 10 0.1",
            rgba="0.8 0.8 0.8 1",
        )

    def _add_walls(self, worldbody: ET.Element, room_size: Dict[str, float]) -> None:
        """Add room walls to worldbody based on room_size.

        Args:
            worldbody: MuJoCo worldbody element
            room_size: Room dimensions with keys 'width', 'length', 'height'
        """
        width = room_size.get("width", 10.0)
        length = room_size.get("length", 10.0)
        height = room_size.get("height", 3.0)

        half_width = width / 2
        half_length = length / 2
        half_height = height / 2

        # Wall thickness
        wall_thickness = 0.1

        # Add four walls
        walls = [
            (
                "north_wall",
                (0, half_length, half_height),
                (half_width, wall_thickness, half_height),
            ),
            (
                "south_wall",
                (0, -half_length, half_height),
                (half_width, wall_thickness, half_height),
            ),
            (
                "east_wall",
                (half_width, 0, half_height),
                (wall_thickness, half_length, half_height),
            ),
            (
                "west_wall",
                (-half_width, 0, half_height),
                (wall_thickness, half_length, half_height),
            ),
        ]

        for name, pos, size in walls:
            ET.SubElement(
                worldbody,
                "geom",
                name=name,
                type="box",
                pos=f"{pos[0]} {pos[1]} {pos[2]}",
                size=f"{size[0]} {size[1]} {size[2]}",
                rgba="0.9 0.9 0.9 1",
            )

    def _add_object(self, worldbody: ET.Element, obj: PlacedObject) -> bool:
        """Add a placed object to worldbody.

        Creates a body element with:
        - Position from obj.position (x, y, z)
        - Orientation from obj.orientation (yaw, pitch, roll as euler angles)
        - Geom element referencing mesh asset
        - Freejoint for dynamic objects (is_static=False)
        - Object ID as name attribute

        Args:
            worldbody: MuJoCo worldbody element
            obj: Placed object to add

        Returns:
            True if object was added successfully, False if skipped

        **Validates: Requirements 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8**
        """
        # Verify model_loc file exists
        model_path = self.workspace_root / obj.model_loc
        if not model_path.exists():
            logger.error(
                f"[SceneAssembly] Model file not found for object {obj.id}: "
                f"{obj.model_loc}. Skipping object."
            )
            return False

        # Create body element with object ID as name
        body = ET.SubElement(worldbody, "body", name=obj.id)

        # Set position (x, y, z)
        pos = f"{obj.position['x']} {obj.position['y']} {obj.position['z']}"
        body.set("pos", pos)

        # Set orientation (euler angles: yaw, pitch, roll in degrees)
        yaw = obj.orientation.get("yaw_deg", 0.0)
        pitch = obj.orientation.get("pitch_deg", 0.0)
        roll = obj.orientation.get("roll_deg", 0.0)
        euler = f"{yaw} {pitch} {roll}"
        body.set("euler", euler)

        # Add freejoint for dynamic objects
        if not obj.is_static:
            ET.SubElement(body, "freejoint")

        # Add geom element referencing the mesh asset
        ET.SubElement(
            body,
            "geom",
            type="mesh",
            mesh=f"mesh_{obj.id}",
            rgba="0.7 0.7 0.7 1",
        )

        logger.debug(
            f"[SceneAssembly] Added object {obj.id} at "
            f"position ({obj.position['x']:.3f}, {obj.position['y']:.3f}, "
            f"{obj.position['z']:.3f}), "
            f"orientation (yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°), "
            f"static={obj.is_static}"
        )

        return True

    def _prettify_xml(self, elem: ET.Element) -> str:
        """Return a pretty-printed XML string.

        Args:
            elem: XML element to prettify

        Returns:
            Pretty-printed XML string
        """
        rough_string = ET.tostring(elem, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
