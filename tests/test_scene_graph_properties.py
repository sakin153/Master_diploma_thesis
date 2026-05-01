"""Property-based tests for Scene Graph integrity.

Feature: universal-scene-placement-system
Property 1: Целостность иерархического графа сцены

This module validates Requirements 1.2, 1.4, 1.5:
- Bidirectional parent-child relationships
- Correctness of get_descendants()
- Hierarchical integrity preservation
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from typing import List, Tuple

from creator.placement.scene_graph import SceneGraph, SceneNode


# ============================================================================
# Hypothesis Strategies for generating test data
# ============================================================================

@st.composite
def scene_node_data(draw):
    """Generate data for creating a SceneNode."""
    node_id = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
        min_size=1,
        max_size=20
    ))
    object_type = draw(st.sampled_from([
        'table', 'chair', 'sofa', 'lamp', 'shelf', 'box', 'plant'
    ]))
    return node_id, object_type


@st.composite
def scene_graph_with_objects(draw, min_objects=1, max_objects=20):
    """Generate a SceneGraph with random objects.
    
    Returns a tuple of (SceneGraph, list of added node IDs).
    """
    graph = SceneGraph()
    num_objects = draw(st.integers(min_value=min_objects, max_value=max_objects))
    
    added_nodes = []
    node_ids = set()
    
    for i in range(num_objects):
        # Generate unique node ID
        node_id = draw(st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
            min_size=1,
            max_size=20
        ))
        
        # Ensure uniqueness
        if node_id in node_ids or node_id == 'root':
            node_id = f"node_{i}_{node_id}"
        
        node_ids.add(node_id)
        
        object_type = draw(st.sampled_from([
            'table', 'chair', 'sofa', 'lamp', 'shelf', 'box', 'plant'
        ]))
        
        # Choose parent: either root or one of the already added nodes
        if added_nodes and draw(st.booleans()):
            parent_id = draw(st.sampled_from(added_nodes))
            parent = graph.get_node(parent_id)
        else:
            parent = graph.root
        
        # Add object to graph
        try:
            node = graph.add_object(
                object_id=node_id,
                object_type=object_type,
                parent=parent
            )
            added_nodes.append(node_id)
        except ValueError:
            # Skip if duplicate ID somehow generated
            continue
    
    return graph, added_nodes


# ============================================================================
# Property 1: Целостность иерархического графа сцены
# ============================================================================

@given(scene_graph_with_objects())
@settings(max_examples=100, deadline=None)
def test_property_1_bidirectional_parent_child_links(graph_data):
    """Property 1a: Parent-child links are bidirectional.
    
    For any node in the graph:
    - If node has a parent, then node must be in parent's children list
    - For each child in node's children, child's parent must be this node
    
    Validates: Requirement 1.2
    """
    graph, node_ids = graph_data
    
    # Check all nodes in the graph
    for node_id in node_ids:
        node = graph.get_node(node_id)
        assert node is not None, f"Node {node_id} should exist in graph"
        
        # Check parent -> child direction
        if node.parent is not None:
            assert node in node.parent.children, \
                f"Node {node_id} has parent {node.parent.id}, " \
                f"but is not in parent's children list"
        
        # Check child -> parent direction
        for child in node.children:
            assert child.parent == node, \
                f"Child {child.id} is in {node_id}'s children, " \
                f"but its parent is {child.parent.id if child.parent else None}"


@given(scene_graph_with_objects())
@settings(max_examples=100, deadline=None)
def test_property_1_get_descendants_correctness(graph_data):
    """Property 1b: get_descendants() returns complete subtree.
    
    For any node in the graph:
    - get_descendants() should return all nodes in the subtree
    - Each descendant should have a path back to the node through parent links
    - No node should appear twice in descendants list
    
    Validates: Requirement 1.4
    """
    graph, node_ids = graph_data
    
    for node_id in node_ids:
        node = graph.get_node(node_id)
        descendants = node.get_descendants()
        
        # Check no duplicates
        descendant_ids = [d.id for d in descendants]
        assert len(descendant_ids) == len(set(descendant_ids)), \
            f"get_descendants() returned duplicates for node {node_id}"
        
        # Check each descendant has path back to node
        for descendant in descendants:
            current = descendant
            found_ancestor = False
            visited = set()
            
            # Walk up the tree to find the node
            while current is not None:
                if current.id in visited:
                    # Cycle detected - should not happen
                    pytest.fail(f"Cycle detected in hierarchy at {current.id}")
                visited.add(current.id)
                
                if current == node:
                    found_ancestor = True
                    break
                current = current.parent
            
            assert found_ancestor, \
                f"Descendant {descendant.id} does not have path back to {node_id}"
        
        # Check completeness: all children and their descendants should be included
        expected_descendants = set()
        for child in node.children:
            expected_descendants.add(child.id)
            for grandchild in child.get_descendants():
                expected_descendants.add(grandchild.id)
        
        actual_descendants = set(d.id for d in descendants)
        assert expected_descendants == actual_descendants, \
            f"get_descendants() incomplete for {node_id}. " \
            f"Expected: {expected_descendants}, Got: {actual_descendants}"


@given(scene_graph_with_objects())
@settings(max_examples=100, deadline=None)
def test_property_1_hierarchy_integrity_preservation(graph_data):
    """Property 1c: Hierarchy integrity is preserved across operations.
    
    For any graph:
    - validate_hierarchy() should return True for all nodes
    - validate_integrity() should return True for the entire graph
    - All nodes should be reachable from root
    - No cycles should exist
    
    Validates: Requirement 1.5
    """
    graph, node_ids = graph_data
    
    # Check each node's local hierarchy
    for node_id in node_ids:
        node = graph.get_node(node_id)
        assert node.validate_hierarchy(), \
            f"Node {node_id} failed hierarchy validation"
    
    # Check root's hierarchy
    assert graph.root.validate_hierarchy(), \
        "Root node failed hierarchy validation"
    
    # Check overall graph integrity
    assert graph.validate_integrity(), \
        "Graph failed integrity validation"
    
    # Verify all nodes are reachable from root
    reachable_from_root = {graph.root.id}
    to_visit = [graph.root]
    
    while to_visit:
        current = to_visit.pop()
        for child in current.children:
            if child.id in reachable_from_root:
                # This would indicate a cycle or duplicate
                pytest.fail(f"Node {child.id} encountered twice - possible cycle")
            reachable_from_root.add(child.id)
            to_visit.append(child)
    
    # All added nodes should be reachable
    all_node_ids = set(node_ids) | {'root'}
    assert all_node_ids == reachable_from_root, \
        f"Not all nodes reachable from root. " \
        f"Missing: {all_node_ids - reachable_from_root}"


@given(scene_graph_with_objects(min_objects=2, max_objects=10))
@settings(max_examples=50, deadline=None)
def test_property_1_add_remove_preserves_integrity(graph_data):
    """Property 1d: Adding and removing nodes preserves integrity.
    
    After adding and removing nodes:
    - Hierarchy integrity should be maintained
    - Removed nodes should not be reachable
    - Parent-child links should remain bidirectional
    
    Validates: Requirements 1.2, 1.5
    """
    graph, node_ids = graph_data
    
    # Skip if too few nodes
    assume(len(node_ids) >= 2)
    
    # Pick a node to remove (not root)
    node_to_remove = node_ids[0]
    node = graph.get_node(node_to_remove)
    
    # Record descendants before removal
    descendants_before = [d.id for d in node.get_descendants()]
    
    # Remove the node
    success = graph.remove_object(node_to_remove)
    assert success, f"Failed to remove node {node_to_remove}"
    
    # Verify node and its descendants are gone
    assert graph.get_node(node_to_remove) is None, \
        f"Node {node_to_remove} still exists after removal"
    
    for desc_id in descendants_before:
        assert graph.get_node(desc_id) is None, \
            f"Descendant {desc_id} still exists after parent removal"
    
    # Verify remaining graph integrity
    assert graph.validate_integrity(), \
        "Graph integrity violated after node removal"
    
    # Verify all remaining nodes have valid hierarchy
    for remaining_id in node_ids[1:]:
        remaining_node = graph.get_node(remaining_id)
        if remaining_node is not None:  # May have been a descendant
            assert remaining_node.validate_hierarchy(), \
                f"Node {remaining_id} hierarchy invalid after removal"


@given(scene_graph_with_objects(min_objects=1, max_objects=10))
@settings(max_examples=50, deadline=None)
def test_property_1_reparenting_preserves_integrity(graph_data):
    """Property 1e: Reparenting nodes preserves integrity.
    
    When moving a node to a new parent:
    - Old parent should no longer list it as child
    - New parent should list it as child
    - Node's parent reference should update
    - Hierarchy integrity should be maintained
    
    Validates: Requirements 1.2, 1.5
    """
    graph, node_ids = graph_data
    
    # Need at least 2 nodes to reparent
    assume(len(node_ids) >= 2)
    
    # Pick a node to reparent and a new parent
    node_to_move_id = node_ids[0]
    new_parent_id = node_ids[1]
    
    node_to_move = graph.get_node(node_to_move_id)
    new_parent = graph.get_node(new_parent_id)
    
    # Make sure we're not creating a cycle (child becoming parent of ancestor)
    ancestors = set()
    current = new_parent
    while current is not None:
        ancestors.add(current.id)
        current = current.parent
    
    assume(node_to_move_id not in ancestors)
    
    # Record old parent
    old_parent = node_to_move.parent
    
    # Reparent by removing from old parent and adding to new parent
    if old_parent is not None:
        old_parent.remove_child(node_to_move)
    
    new_parent.add_child(node_to_move)
    
    # Verify old parent no longer has the node
    if old_parent is not None:
        assert node_to_move not in old_parent.children, \
            f"Node {node_to_move_id} still in old parent's children"
    
    # Verify new parent has the node
    assert node_to_move in new_parent.children, \
        f"Node {node_to_move_id} not in new parent's children"
    
    # Verify node's parent reference updated
    assert node_to_move.parent == new_parent, \
        f"Node {node_to_move_id} parent not updated"
    
    # Verify hierarchy integrity
    assert node_to_move.validate_hierarchy(), \
        f"Node {node_to_move_id} hierarchy invalid after reparenting"
    
    assert graph.validate_integrity(), \
        "Graph integrity violated after reparenting"


# ============================================================================
# Additional edge case tests
# ============================================================================

def test_empty_graph_integrity():
    """Test that an empty graph (only root) maintains integrity."""
    graph = SceneGraph()
    
    assert graph.validate_integrity()
    assert graph.root.validate_hierarchy()
    assert len(graph.root.get_descendants()) == 0


def test_single_object_graph():
    """Test graph with single object maintains integrity."""
    graph = SceneGraph()
    node = graph.add_object("obj1", "table")
    
    assert graph.validate_integrity()
    assert node.validate_hierarchy()
    assert node.parent == graph.root
    assert node in graph.root.children
    assert len(node.get_descendants()) == 0


def test_deep_hierarchy():
    """Test deeply nested hierarchy maintains integrity."""
    graph = SceneGraph()
    
    # Create a chain: root -> n1 -> n2 -> n3 -> n4 -> n5
    parent = graph.root
    nodes = []
    for i in range(5):
        node = graph.add_object(f"node{i}", "box", parent=parent)
        nodes.append(node)
        parent = node
    
    # Verify integrity
    assert graph.validate_integrity()
    
    # Verify root's descendants include all nodes
    root_descendants = graph.root.get_descendants()
    assert len(root_descendants) == 5
    assert all(n in root_descendants for n in nodes)
    
    # Verify first node's descendants
    assert len(nodes[0].get_descendants()) == 4
    
    # Verify last node has no descendants
    assert len(nodes[4].get_descendants()) == 0


def test_wide_hierarchy():
    """Test wide hierarchy (many children of root) maintains integrity."""
    graph = SceneGraph()
    
    # Create many children of root
    nodes = []
    for i in range(20):
        node = graph.add_object(f"node{i}", "chair", parent=graph.root)
        nodes.append(node)
    
    # Verify integrity
    assert graph.validate_integrity()
    
    # Verify all are children of root
    assert len(graph.root.children) == 20
    assert all(n.parent == graph.root for n in nodes)
    
    # Verify root's descendants
    root_descendants = graph.root.get_descendants()
    assert len(root_descendants) == 20
    assert all(n in root_descendants for n in nodes)


def test_duplicate_id_rejected():
    """Test that duplicate object IDs are rejected."""
    graph = SceneGraph()
    
    graph.add_object("obj1", "table")
    
    with pytest.raises(ValueError, match="already exists"):
        graph.add_object("obj1", "chair")


def test_remove_nonexistent_node():
    """Test removing non-existent node returns False."""
    graph = SceneGraph()
    
    result = graph.remove_object("nonexistent")
    assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
