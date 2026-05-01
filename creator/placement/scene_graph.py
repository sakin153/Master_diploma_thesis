"""Scene Graph for hierarchical object representation.

This module implements a hierarchical scene graph with parent-child relationships
for representing complex spatial arrangements of objects in 3D scenes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SceneNode:
    """Node in the scene graph representing a single object.

    Attributes:
        id: Unique identifier for this node
        object_type: Type/category of the object (e.g., "table", "chair")
        parent: Reference to parent node (None for root-level objects)
        children: List of child nodes (objects positioned relative to this one)
        constraints: Spatial constraints associated with this object
        anchor_points: Available anchor points for positioning other objects
        metadata: Additional object-specific data (size, model name, etc.)
    """

    id: str
    object_type: str
    parent: Optional['SceneNode'] = None
    children: List['SceneNode'] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    anchor_points: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: 'SceneNode') -> None:
        """Add a child node and establish bidirectional parent-child link.

        Args:
            child: The child node to add

        Raises:
            ValueError: If child already has a different parent
        """
        if child.parent is not None and child.parent != self:
            raise ValueError(
                f"Node {child.id} already has parent {child.parent.id}, "
                f"cannot add as child of {self.id}"
            )

        if child not in self.children:
            self.children.append(child)
        child.parent = self

    def remove_child(self, child: 'SceneNode') -> None:
        """Remove a child node and clear its parent reference.

        Args:
            child: The child node to remove
        """
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def get_descendants(self) -> List['SceneNode']:
        """Get all descendant nodes (children, grandchildren, etc.).

        Returns:
            List of all descendant nodes in depth-first order
        """
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants

    def find_anchor_candidates(
        self, anchor_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find available anchor points on this node and its ancestors.

        Args:
            anchor_type: Optional filter for specific anchor type

        Returns:
            List of anchor point dictionaries matching the criteria
        """
        candidates = []

        # Check this node's anchor points
        for anchor in self.anchor_points:
            if anchor.get("available", True):
                if anchor_type is None or anchor.get("anchor_type") == anchor_type:
                    candidates.append(anchor)

        # Check parent's anchor points (for hierarchical positioning)
        if self.parent is not None:
            parent_anchors = self.parent.find_anchor_candidates(anchor_type)
            candidates.extend(parent_anchors)

        return candidates

    def validate_hierarchy(self) -> bool:
        """Validate parent-child relationship integrity.

        Returns:
            True if hierarchy is valid, False otherwise
        """
        # Check bidirectional parent-child links
        if self.parent is not None and self not in self.parent.children:
            return False

        # Check all children have correct parent reference
        for child in self.children:
            if child.parent != self:
                return False
            # Recursively validate children
            if not child.validate_hierarchy():
                return False

        return True


@dataclass
class SceneGraph:
    """Hierarchical graph representing the entire scene.

    Attributes:
        root: Root node of the graph (typically represents the room/scene)
        nodes: Dictionary mapping node IDs to SceneNode instances
        global_anchors: List of global anchor points (walls, corners, center)
    """

    root: SceneNode = field(default_factory=lambda: SceneNode(
        id="root",
        object_type="scene_root",
        metadata={"description": "Scene root node"}
    ))
    nodes: Dict[str, SceneNode] = field(default_factory=dict)
    global_anchors: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize the graph with the root node."""
        self.nodes[self.root.id] = self.root

    def add_object(
        self,
        object_id: str,
        object_type: str,
        parent: Optional[SceneNode] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        anchor_points: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SceneNode:
        """Add a new object to the scene graph.

        Args:
            object_id: Unique identifier for the object
            object_type: Type/category of the object
            parent: Parent node (defaults to root if None)
            constraints: Spatial constraints for this object
            anchor_points: Anchor points provided by this object
            metadata: Additional object data

        Returns:
            The newly created SceneNode

        Raises:
            ValueError: If object_id already exists in the graph
        """
        if object_id in self.nodes:
            raise ValueError(f"Object with id '{object_id}' already exists in scene graph")

        # Create new node
        node = SceneNode(
            id=object_id,
            object_type=object_type,
            constraints=constraints or [],
            anchor_points=anchor_points or [],
            metadata=metadata or {}
        )

        # Add to nodes dictionary
        self.nodes[object_id] = node

        # Establish parent-child relationship
        parent_node = parent if parent is not None else self.root
        parent_node.add_child(node)

        return node

    def get_node(self, object_id: str) -> Optional[SceneNode]:
        """Get a node by its ID.

        Args:
            object_id: The ID of the node to retrieve

        Returns:
            The SceneNode if found, None otherwise
        """
        return self.nodes.get(object_id)

    def remove_object(self, object_id: str) -> bool:
        """Remove an object from the scene graph.

        Args:
            object_id: The ID of the object to remove

        Returns:
            True if object was removed, False if not found
        """
        node = self.nodes.get(object_id)
        if node is None:
            return False

        # Remove from parent's children
        if node.parent is not None:
            node.parent.remove_child(node)

        # Recursively remove all descendants
        for child in list(node.children):
            self.remove_object(child.id)

        # Remove from nodes dictionary
        del self.nodes[object_id]

        return True

    def get_descendants(self, object_id: str) -> List[SceneNode]:
        """Get all descendants of a specific object.

        Args:
            object_id: The ID of the object

        Returns:
            List of descendant nodes, empty list if object not found
        """
        node = self.nodes.get(object_id)
        if node is None:
            return []
        return node.get_descendants()

    def find_anchor_candidates(
        self,
        object_id: Optional[str] = None,
        anchor_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find available anchor points in the scene.

        Args:
            object_id: Optional ID of object to search from (searches from root if None)
            anchor_type: Optional filter for specific anchor type

        Returns:
            List of available anchor points
        """
        candidates = []

        # Add global anchors
        for anchor in self.global_anchors:
            if anchor.get("available", True):
                if anchor_type is None or anchor.get("anchor_type") == anchor_type:
                    candidates.append(anchor)

        # Add local anchors from specific node or all nodes
        if object_id is not None:
            node = self.nodes.get(object_id)
            if node is not None:
                candidates.extend(node.find_anchor_candidates(anchor_type))
        else:
            # Search all nodes for anchor points
            for node in self.nodes.values():
                for anchor in node.anchor_points:
                    if anchor.get("available", True):
                        if anchor_type is None or anchor.get("anchor_type") == anchor_type:
                            candidates.append(anchor)

        return candidates

    def validate_integrity(self) -> bool:
        """Validate the integrity of the entire scene graph.

        Returns:
            True if graph is valid, False otherwise
        """
        # Check root node exists
        if self.root.id not in self.nodes:
            return False

        # Validate all nodes are reachable from root
        reachable = {self.root.id}
        to_visit = [self.root]

        while to_visit:
            current = to_visit.pop()
            for child in current.children:
                if child.id in reachable:
                    # Cycle detected
                    return False
                reachable.add(child.id)
                to_visit.append(child)

        # Check all nodes in dictionary are reachable
        if set(self.nodes.keys()) != reachable:
            return False

        # Validate each node's hierarchy
        for node in self.nodes.values():
            if not node.validate_hierarchy():
                return False

        return True

    def get_all_objects(self) -> List[SceneNode]:
        """Get all objects in the scene (excluding root).

        Returns:
            List of all SceneNode objects except the root
        """
        return [node for node in self.nodes.values() if node != self.root]

    def get_objects_by_type(self, object_type: str) -> List[SceneNode]:
        """Get all objects of a specific type.

        Args:
            object_type: The type of objects to retrieve

        Returns:
            List of SceneNode objects matching the type
        """
        return [
            node for node in self.nodes.values()
            if node.object_type == object_type and node != self.root
        ]
