"""Unit tests for Semantic Plan Parser.

These tests verify that the SemanticPlanParser correctly parses JSON plans,
validates structure, and converts semantic relationships to geometric constraints.
"""

import json
import pytest

from creator.placement.parser import (
    SemanticPlanParser,
    ParseError,
    ValidationError,
    SchemaVersion,
    SemanticRelationship,
)
from creator.placement.constraints import (
    ConstraintType,
    DistanceConstraint,
    AlignmentConstraint,
    FacingConstraint,
    ContainmentConstraint,
)


class TestBasicParsing:
    """Test basic JSON parsing functionality."""

    def test_parse_valid_json_string(self):
        """Test parsing a valid JSON string."""
        parser = SemanticPlanParser()

        json_str = json.dumps({
            "schema_version": "1.0",
            "objects": [
                {"id": "table1", "type": "table"},
                {"id": "chair1", "type": "chair"}
            ]
        })

        plan = parser.parse(json_str)

        assert plan.schema_version == SchemaVersion.V1_0
        assert len(plan.objects) == 2
        assert plan.objects[0]["id"] == "table1"

    def test_parse_valid_dict(self):
        """Test parsing a valid dictionary."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.0",
            "objects": [
                {"id": "table1", "type": "table"}
            ]
        }

        plan = parser.parse(plan_dict)

        assert plan.schema_version == SchemaVersion.V1_0
        assert len(plan.objects) == 1

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON string."""
        parser = SemanticPlanParser()

        invalid_json = "{invalid json"

        with pytest.raises(ParseError, match="Invalid JSON"):
            parser.parse(invalid_json)

    def test_parse_non_dict(self):
        """Test parsing non-dictionary data."""
        parser = SemanticPlanParser()

        with pytest.raises(ValidationError, match="must be a dictionary"):
            parser.parse([1, 2, 3])


class TestSchemaVersioning:
    """Test schema version handling."""

    def test_parse_v1_0_schema(self):
        """Test parsing schema version 1.0."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.0",
            "objects": [{"id": "obj1", "type": "table"}]
        }

        plan = parser.parse(plan_dict)

        assert plan.schema_version == SchemaVersion.V1_0

    def test_parse_v1_1_schema(self):
        """Test parsing schema version 1.1."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [{"id": "obj1", "type": "table"}],
            "relationships": []
        }

        plan = parser.parse(plan_dict)

        assert plan.schema_version == SchemaVersion.V1_1

    def test_parse_unsupported_version(self):
        """Test parsing unsupported schema version."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "2.0",
            "objects": [{"id": "obj1", "type": "table"}]
        }

        with pytest.raises(ValidationError, match="Unsupported schema version"):
            parser.parse(plan_dict)

    def test_parse_default_version(self):
        """Test parsing with default schema version."""
        parser = SemanticPlanParser()

        plan_dict = {
            "objects": [{"id": "obj1", "type": "table"}]
        }

        plan = parser.parse(plan_dict)

        assert plan.schema_version == SchemaVersion.V1_0


class TestStructureValidation:
    """Test structure validation."""

    def test_missing_required_field_objects(self):
        """Test validation fails when objects field is missing."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.0"
        }

        with pytest.raises(ValidationError, match="Missing required field: objects"):
            parser.parse(plan_dict)

    def test_missing_required_field_relationships_v1_1(self):
        """Test validation fails when relationships field is missing in v1.1."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [{"id": "obj1", "type": "table"}]
        }

        with pytest.raises(ValidationError, match="Missing required field: relationships"):
            parser.parse(plan_dict)

    def test_wrong_type_objects(self):
        """Test validation fails when objects is not a list."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.0",
            "objects": "not a list"
        }

        with pytest.raises(ValidationError, match="must be of type list"):
            parser.parse(plan_dict)

    def test_empty_objects_list(self):
        """Test validation fails when objects list is empty."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.0",
            "objects": []
        }

        with pytest.raises(ValidationError, match="must contain at least one object"):
            parser.parse(plan_dict)


class TestObjectValidation:
    """Test object specification validation."""

    def test_object_missing_id(self):
        """Test validation fails when object is missing id."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.0",
            "objects": [
                {"type": "table"}
            ]
        }

        with pytest.raises(ValidationError, match="missing required field 'id'"):
            parser.parse(plan_dict)

    def test_object_missing_type(self):
        """Test validation fails when object is missing type."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.0",
            "objects": [
                {"id": "obj1"}
            ]
        }

        with pytest.raises(ValidationError, match="missing required field 'type'"):
            parser.parse(plan_dict)

    def test_duplicate_object_ids(self):
        """Test validation fails when objects have duplicate IDs."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.0",
            "objects": [
                {"id": "obj1", "type": "table"},
                {"id": "obj1", "type": "chair"}
            ]
        }

        with pytest.raises(ValidationError, match="Duplicate object ID"):
            parser.parse(plan_dict)

    def test_object_not_dict(self):
        """Test validation fails when object is not a dictionary."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.0",
            "objects": [
                "not a dict"
            ]
        }

        with pytest.raises(ValidationError, match="must be a dictionary"):
            parser.parse(plan_dict)


class TestRelationshipParsing:
    """Test semantic relationship parsing."""

    def test_parse_simple_relationship(self):
        """Test parsing a simple relationship."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [
                {"id": "chair1", "type": "chair"},
                {"id": "table1", "type": "table"}
            ],
            "relationships": [
                {
                    "type": "near",
                    "source": "chair1",
                    "target": "table1"
                }
            ]
        }

        plan = parser.parse(plan_dict)

        assert len(plan.relationships) == 1
        assert plan.relationships[0].relationship_type == "near"
        assert plan.relationships[0].source_object == "chair1"
        assert plan.relationships[0].target_object == "table1"

    def test_parse_relationship_with_parameters(self):
        """Test parsing relationship with parameters."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [
                {"id": "chair1", "type": "chair"},
                {"id": "table1", "type": "table"}
            ],
            "relationships": [
                {
                    "type": "near",
                    "source": "chair1",
                    "target": "table1",
                    "parameters": {
                        "min_distance": 0.5,
                        "max_distance": 1.5
                    }
                }
            ]
        }

        plan = parser.parse(plan_dict)

        assert len(plan.relationships) == 1
        assert plan.relationships[0].parameters["min_distance"] == 0.5
        assert plan.relationships[0].parameters["max_distance"] == 1.5

    def test_parse_relationship_without_target(self):
        """Test parsing relationship without target (absolute positioning)."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [
                {"id": "table1", "type": "table"}
            ],
            "relationships": [
                {
                    "type": "within",
                    "source": "table1"
                }
            ]
        }

        plan = parser.parse(plan_dict)

        assert len(plan.relationships) == 1
        assert plan.relationships[0].target_object is None

    def test_relationship_missing_type(self):
        """Test validation fails when relationship is missing type."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [{"id": "obj1", "type": "table"}],
            "relationships": [
                {
                    "source": "obj1"
                }
            ]
        }

        with pytest.raises(ValidationError, match="missing required field 'type'"):
            parser.parse(plan_dict)

    def test_relationship_missing_source(self):
        """Test validation fails when relationship is missing source."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [{"id": "obj1", "type": "table"}],
            "relationships": [
                {
                    "type": "near"
                }
            ]
        }

        with pytest.raises(ValidationError, match="missing required field 'source'"):
            parser.parse(plan_dict)


class TestConstraintConversion:
    """Test conversion of relationships to constraints."""

    def test_convert_near_to_distance_constraint(self):
        """Test converting 'near' relationship to distance constraint."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [
                {"id": "chair1", "type": "chair"},
                {"id": "table1", "type": "table"}
            ],
            "relationships": [
                {
                    "type": "near",
                    "source": "chair1",
                    "target": "table1"
                }
            ]
        }

        plan = parser.parse(plan_dict)

        assert len(plan.constraints) == 1
        constraint = plan.constraints[0]
        assert isinstance(constraint, DistanceConstraint)
        assert constraint.source_object == "chair1"
        assert constraint.target_object == "table1"
        assert constraint.min_distance == 0.2
        assert constraint.max_distance == 1.0

    def test_convert_aligned_with_to_alignment_constraint(self):
        """Test converting 'aligned_with' relationship to alignment constraint."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [
                {"id": "chair1", "type": "chair"},
                {"id": "table1", "type": "table"}
            ],
            "relationships": [
                {
                    "type": "aligned_with",
                    "source": "chair1",
                    "target": "table1",
                    "parameters": {
                        "axis": "x",
                        "alignment_type": "center"
                    }
                }
            ]
        }

        plan = parser.parse(plan_dict)

        assert len(plan.constraints) == 1
        constraint = plan.constraints[0]
        assert isinstance(constraint, AlignmentConstraint)
        assert constraint.source_object == "chair1"
        assert constraint.target_object == "table1"

    def test_convert_facing_to_facing_constraint(self):
        """Test converting 'facing' relationship to facing constraint."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [
                {"id": "chair1", "type": "chair"},
                {"id": "table1", "type": "table"}
            ],
            "relationships": [
                {
                    "type": "facing",
                    "source": "chair1",
                    "target": "table1"
                }
            ]
        }

        plan = parser.parse(plan_dict)

        assert len(plan.constraints) == 1
        constraint = plan.constraints[0]
        assert isinstance(constraint, FacingConstraint)
        assert constraint.source_object == "chair1"
        assert constraint.target_object == "table1"

    def test_convert_inside_to_containment_constraint(self):
        """Test converting 'inside' relationship to containment constraint."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [
                {"id": "apple1", "type": "apple"},
                {"id": "box1", "type": "box"}
            ],
            "relationships": [
                {
                    "type": "inside",
                    "source": "apple1",
                    "target": "box1"
                }
            ]
        }

        plan = parser.parse(plan_dict)

        assert len(plan.constraints) == 1
        constraint = plan.constraints[0]
        assert isinstance(constraint, ContainmentConstraint)
        assert constraint.source_object == "apple1"
        assert constraint.target_object == "box1"

    def test_convert_on_to_support_constraint(self):
        """Test converting 'on' relationship to support constraint."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [
                {"id": "book1", "type": "book"},
                {"id": "table1", "type": "table"}
            ],
            "relationships": [
                {
                    "type": "on",
                    "source": "book1",
                    "target": "table1"
                }
            ]
        }

        plan = parser.parse(plan_dict)

        assert len(plan.constraints) == 1
        constraint = plan.constraints[0]
        assert constraint.constraint_type == ConstraintType.SUPPORT
        assert constraint.source_object == "book1"
        assert constraint.target_object == "table1"

    def test_convert_custom_distance_parameters(self):
        """Test converting relationship with custom distance parameters."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [
                {"id": "chair1", "type": "chair"},
                {"id": "table1", "type": "table"}
            ],
            "relationships": [
                {
                    "type": "near",
                    "source": "chair1",
                    "target": "table1",
                    "parameters": {
                        "min_distance": 0.3,
                        "max_distance": 0.8,
                        "priority": 2.0,
                        "is_hard": False
                    }
                }
            ]
        }

        plan = parser.parse(plan_dict)

        constraint = plan.constraints[0]
        assert isinstance(constraint, DistanceConstraint)
        assert constraint.min_distance == 0.3
        assert constraint.max_distance == 0.8
        assert constraint.priority == 2.0
        assert constraint.is_hard is False

    def test_unknown_relationship_type_skipped(self):
        """Test that unknown relationship types are skipped."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [
                {"id": "obj1", "type": "table"}
            ],
            "relationships": [
                {
                    "type": "unknown_relationship",
                    "source": "obj1"
                }
            ]
        }

        plan = parser.parse(plan_dict)

        # Unknown relationship should be parsed but not converted to constraint
        assert len(plan.relationships) == 1
        assert len(plan.constraints) == 0


class TestExplicitConstraints:
    """Test parsing explicit constraints."""

    def test_parse_explicit_distance_constraint(self):
        """Test parsing explicit distance constraint."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.0",
            "objects": [
                {"id": "chair1", "type": "chair"},
                {"id": "table1", "type": "table"}
            ],
            "constraints": [
                {
                    "id": "c1",
                    "constraint_type": "distance",
                    "source_object": "chair1",
                    "target_object": "table1",
                    "min_distance": 0.5,
                    "max_distance": 1.0
                }
            ]
        }

        plan = parser.parse(plan_dict)

        assert len(plan.constraints) == 1
        constraint = plan.constraints[0]
        assert isinstance(constraint, DistanceConstraint)
        assert constraint.id == "c1"
        assert constraint.min_distance == 0.5

    def test_parse_mixed_relationships_and_constraints(self):
        """Test parsing both relationships and explicit constraints."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [
                {"id": "chair1", "type": "chair"},
                {"id": "table1", "type": "table"}
            ],
            "relationships": [
                {
                    "type": "near",
                    "source": "chair1",
                    "target": "table1"
                }
            ],
            "constraints": [
                {
                    "id": "c1",
                    "constraint_type": "alignment",
                    "source_object": "chair1",
                    "target_object": "table1",
                    "alignment_axis": "x",
                    "alignment_type": "center"
                }
            ]
        }

        plan = parser.parse(plan_dict)

        # Should have 1 constraint from relationship + 1 explicit constraint
        assert len(plan.constraints) == 2


class TestPlanValidation:
    """Test plan validation."""

    def test_validate_valid_plan(self):
        """Test validation of a valid plan."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [
                {"id": "chair1", "type": "chair"},
                {"id": "table1", "type": "table"}
            ],
            "relationships": [
                {
                    "type": "near",
                    "source": "chair1",
                    "target": "table1"
                }
            ]
        }

        plan = parser.parse(plan_dict)
        warnings = parser.validate_plan(plan)

        assert len(warnings) == 0

    def test_validate_unknown_source_object(self):
        """Test validation detects unknown source object."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.0",
            "objects": [
                {"id": "table1", "type": "table"}
            ],
            "constraints": [
                {
                    "id": "c1",
                    "constraint_type": "distance",
                    "source_object": "unknown_obj",
                    "target_object": "table1",
                    "min_distance": 0.5
                }
            ]
        }

        plan = parser.parse(plan_dict)
        warnings = parser.validate_plan(plan)

        assert len(warnings) > 0
        assert "unknown source object" in warnings[0].lower()

    def test_validate_unknown_target_object(self):
        """Test validation detects unknown target object."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.0",
            "objects": [
                {"id": "chair1", "type": "chair"}
            ],
            "constraints": [
                {
                    "id": "c1",
                    "constraint_type": "distance",
                    "source_object": "chair1",
                    "target_object": "unknown_obj",
                    "min_distance": 0.5
                }
            ]
        }

        plan = parser.parse(plan_dict)
        warnings = parser.validate_plan(plan)

        assert len(warnings) > 0
        assert "unknown target object" in warnings[0].lower()

    def test_validate_conflicting_distance_constraints(self):
        """Test validation detects conflicting distance constraints."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.0",
            "objects": [
                {"id": "chair1", "type": "chair"},
                {"id": "table1", "type": "table"}
            ],
            "constraints": [
                {
                    "id": "c1",
                    "constraint_type": "distance",
                    "source_object": "chair1",
                    "target_object": "table1",
                    "min_distance": 0.5,
                    "max_distance": 1.0,
                    "is_hard": True
                },
                {
                    "id": "c2",
                    "constraint_type": "distance",
                    "source_object": "chair1",
                    "target_object": "table1",
                    "min_distance": 2.0,
                    "max_distance": 3.0,
                    "is_hard": True
                }
            ]
        }

        plan = parser.parse(plan_dict)
        warnings = parser.validate_plan(plan)

        assert len(warnings) > 0
        assert "conflicting" in warnings[0].lower()


class TestMetadata:
    """Test metadata handling."""

    def test_parse_metadata(self):
        """Test parsing plan metadata."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.0",
            "objects": [{"id": "obj1", "type": "table"}],
            "metadata": {
                "author": "LLM",
                "timestamp": "2024-01-01T00:00:00Z",
                "description": "Test scene"
            }
        }

        plan = parser.parse(plan_dict)

        assert plan.metadata["author"] == "LLM"
        assert plan.metadata["timestamp"] == "2024-01-01T00:00:00Z"
        assert plan.metadata["description"] == "Test scene"

    def test_parse_without_metadata(self):
        """Test parsing plan without metadata."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.0",
            "objects": [{"id": "obj1", "type": "table"}]
        }

        plan = parser.parse(plan_dict)

        assert plan.metadata == {}


class TestConstraintIDGeneration:
    """Test constraint ID generation."""

    def test_unique_constraint_ids(self):
        """Test that generated constraint IDs are unique."""
        parser = SemanticPlanParser()

        plan_dict = {
            "schema_version": "1.1",
            "objects": [
                {"id": "obj1", "type": "table"},
                {"id": "obj2", "type": "chair"},
                {"id": "obj3", "type": "lamp"}
            ],
            "relationships": [
                {"type": "near", "source": "obj1", "target": "obj2"},
                {"type": "near", "source": "obj2", "target": "obj3"},
                {"type": "near", "source": "obj3", "target": "obj1"}
            ]
        }

        plan = parser.parse(plan_dict)

        constraint_ids = [c.id for c in plan.constraints]
        assert len(constraint_ids) == len(set(constraint_ids))  # All unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
