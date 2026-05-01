"""Semantic Plan Parser for LLM-generated placement plans.

This module provides parsing and validation of JSON-formatted semantic plans
from LLM, converting them into geometric constraints for the Layout Solver.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from creator.placement.constraints import (
    Constraint,
    ConstraintType,
    DistanceConstraint,
    AlignmentConstraint,
    FacingConstraint,
    ContainmentConstraint,
    DistanceMeasurement,
    Axis,
    AlignmentType,
    FacingType,
    create_constraint,
)


class SchemaVersion(Enum):
    """Supported schema versions for semantic plans."""
    V1_0 = "1.0"
    V1_1 = "1.1"


class ParseError(Exception):
    """Exception raised when parsing fails."""
    pass


class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass


@dataclass
class SemanticRelationship:
    """Semantic relationship between objects.

    Attributes:
        relationship_type: Type of relationship (on, near, beside, etc.)
        source_object: ID of source object
        target_object: ID of target object (None for absolute relationships)
        parameters: Additional relationship parameters
    """
    relationship_type: str
    source_object: str
    target_object: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedPlan:
    """Parsed semantic plan.

    Attributes:
        schema_version: Version of the plan schema
        objects: List of object specifications
        relationships: List of semantic relationships
        constraints: List of geometric constraints
        metadata: Additional plan metadata
    """
    schema_version: SchemaVersion
    objects: List[Dict[str, Any]]
    relationships: List[SemanticRelationship]
    constraints: List[Constraint]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticPlanParser:
    """Parser for LLM-generated semantic placement plans.

    This parser validates JSON structure, extracts semantic relationships,
    and converts them into geometric constraints for the Layout Solver.
    """

    # Required fields for different schema versions
    REQUIRED_FIELDS_V1_0 = {
        "schema_version": str,
        "objects": list,
    }

    REQUIRED_FIELDS_V1_1 = {
        "schema_version": str,
        "objects": list,
        "relationships": list,
    }

    # Mapping from semantic relationships to constraint types
    RELATIONSHIP_TO_CONSTRAINT = {
        "on": ConstraintType.SUPPORT,
        "above": ConstraintType.DISTANCE,
        "below": ConstraintType.DISTANCE,
        "near": ConstraintType.DISTANCE,
        "beside": ConstraintType.DISTANCE,
        "next_to": ConstraintType.DISTANCE,
        "far_from": ConstraintType.DISTANCE,
        "aligned_with": ConstraintType.ALIGNMENT,
        "facing": ConstraintType.FACING,
        "facing_away": ConstraintType.FACING,
        "parallel_to": ConstraintType.FACING,
        "perpendicular_to": ConstraintType.FACING,
        "inside": ConstraintType.CONTAINMENT,
        "within": ConstraintType.CONTAINMENT,
        "against": ConstraintType.DISTANCE,
    }

    # Default parameters for semantic relationships
    DEFAULT_RELATIONSHIP_PARAMS = {
        "near": {"min_distance": 0.2, "max_distance": 1.0},
        "beside": {"min_distance": 0.1, "max_distance": 0.5},
        "next_to": {"min_distance": 0.05, "max_distance": 0.3},
        "far_from": {"min_distance": 2.0, "max_distance": float('inf')},
        "against": {"min_distance": 0.0, "max_distance": 0.05},
    }

    def __init__(self):
        """Initialize the parser."""
        self._constraint_id_counter = 0

    def parse(self, json_data: Union[str, Dict[str, Any]]) -> ParsedPlan:
        """Parse a semantic plan from JSON.

        Args:
            json_data: JSON string or dictionary containing the plan

        Returns:
            ParsedPlan object with extracted data

        Raises:
            ParseError: If JSON is malformed
            ValidationError: If plan structure is invalid
        """
        # Parse JSON if string
        if isinstance(json_data, str):
            try:
                plan_dict = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise ParseError(f"Invalid JSON: {e}")
        else:
            plan_dict = json_data

        # Validate structure
        self._validate_structure(plan_dict)

        # Determine schema version
        schema_version = self._parse_schema_version(plan_dict.get("schema_version", "1.0"))

        # Extract objects
        objects = plan_dict.get("objects", [])
        self._validate_objects(objects)

        # Extract relationships
        relationships = self._parse_relationships(plan_dict.get("relationships", []))

        # Convert relationships to constraints
        constraints = self._relationships_to_constraints(relationships)

        # Add explicit constraints if present
        if "constraints" in plan_dict:
            explicit_constraints = self._parse_explicit_constraints(
                plan_dict["constraints"]
            )
            constraints.extend(explicit_constraints)

        # Extract metadata
        metadata = plan_dict.get("metadata", {})

        return ParsedPlan(
            schema_version=schema_version,
            objects=objects,
            relationships=relationships,
            constraints=constraints,
            metadata=metadata,
        )

    def _validate_structure(self, plan_dict: Dict[str, Any]) -> None:
        """Validate the basic structure of the plan.

        Args:
            plan_dict: Plan dictionary to validate

        Raises:
            ValidationError: If structure is invalid
        """
        if not isinstance(plan_dict, dict):
            raise ValidationError("Plan must be a dictionary")

        # Get schema version to determine required fields
        schema_version_str = plan_dict.get("schema_version", "1.0")
        try:
            schema_version = SchemaVersion(schema_version_str)
        except ValueError:
            raise ValidationError(
                f"Unsupported schema version: {schema_version_str}. "
                f"Supported versions: {[v.value for v in SchemaVersion]}"
            )

        # Select required fields based on version
        if schema_version == SchemaVersion.V1_0:
            required_fields = self.REQUIRED_FIELDS_V1_0
        else:  # V1_1
            required_fields = self.REQUIRED_FIELDS_V1_1

        # Check required fields
        for field_name, field_type in required_fields.items():
            if field_name not in plan_dict:
                raise ValidationError(f"Missing required field: {field_name}")

            if not isinstance(plan_dict[field_name], field_type):
                raise ValidationError(
                    f"Field '{field_name}' must be of type {field_type.__name__}, "
                    f"got {type(plan_dict[field_name]).__name__}"
                )

    def _parse_schema_version(self, version_str: str) -> SchemaVersion:
        """Parse schema version string.

        Args:
            version_str: Version string (e.g., "1.0")

        Returns:
            SchemaVersion enum value

        Raises:
            ValidationError: If version is unsupported
        """
        try:
            return SchemaVersion(version_str)
        except ValueError:
            raise ValidationError(
                f"Unsupported schema version: {version_str}. "
                f"Supported versions: {[v.value for v in SchemaVersion]}"
            )

    def _validate_objects(self, objects: List[Dict[str, Any]]) -> None:
        """Validate object specifications.

        Args:
            objects: List of object dictionaries

        Raises:
            ValidationError: If objects are invalid
        """
        if not objects:
            raise ValidationError("Plan must contain at least one object")

        seen_ids = set()
        for i, obj in enumerate(objects):
            if not isinstance(obj, dict):
                raise ValidationError(f"Object at index {i} must be a dictionary")

            # Check required fields
            if "id" not in obj:
                raise ValidationError(f"Object at index {i} missing required field 'id'")

            # Make 'type' optional with default value
            if "type" not in obj:
                obj["type"] = "furniture"  # Default type for backward compatibility

            # Check for duplicate IDs
            obj_id = obj["id"]
            if obj_id in seen_ids:
                raise ValidationError(f"Duplicate object ID: {obj_id}")
            seen_ids.add(obj_id)

    def _parse_relationships(
        self, relationships_data: List[Dict[str, Any]]
    ) -> List[SemanticRelationship]:
        """Parse semantic relationships from plan data.

        Args:
            relationships_data: List of relationship dictionaries

        Returns:
            List of SemanticRelationship objects

        Raises:
            ValidationError: If relationships are invalid
        """
        relationships = []

        for i, rel_data in enumerate(relationships_data):
            if not isinstance(rel_data, dict):
                raise ValidationError(f"Relationship at index {i} must be a dictionary")

            # Check required fields
            if "type" not in rel_data:
                raise ValidationError(
                    f"Relationship at index {i} missing required field 'type'"
                )

            if "source" not in rel_data:
                raise ValidationError(
                    f"Relationship at index {i} missing required field 'source'"
                )

            relationship = SemanticRelationship(
                relationship_type=rel_data["type"],
                source_object=rel_data["source"],
                target_object=rel_data.get("target"),
                parameters=rel_data.get("parameters", {}),
            )

            relationships.append(relationship)

        return relationships

    def _relationships_to_constraints(
        self, relationships: List[SemanticRelationship]
    ) -> List[Constraint]:
        """Convert semantic relationships to geometric constraints.

        Args:
            relationships: List of semantic relationships

        Returns:
            List of Constraint objects
        """
        constraints = []

        for rel in relationships:
            constraint = self._relationship_to_constraint(rel)
            if constraint is not None:
                constraints.append(constraint)

        return constraints

    def _relationship_to_constraint(
        self, relationship: SemanticRelationship
    ) -> Optional[Constraint]:
        """Convert a single semantic relationship to a constraint.

        Args:
            relationship: Semantic relationship to convert

        Returns:
            Constraint object or None if conversion not possible
        """
        rel_type = relationship.relationship_type.lower()

        # Get constraint type
        constraint_type = self.RELATIONSHIP_TO_CONSTRAINT.get(rel_type)
        if constraint_type is None:
            # Unknown relationship type - skip
            return None

        # Generate unique constraint ID
        constraint_id = self._generate_constraint_id()

        # Convert based on constraint type
        if constraint_type == ConstraintType.DISTANCE:
            return self._create_distance_constraint(constraint_id, relationship)

        elif constraint_type == ConstraintType.ALIGNMENT:
            return self._create_alignment_constraint(constraint_id, relationship)

        elif constraint_type == ConstraintType.FACING:
            return self._create_facing_constraint(constraint_id, relationship)

        elif constraint_type == ConstraintType.CONTAINMENT:
            return self._create_containment_constraint(constraint_id, relationship)

        elif constraint_type == ConstraintType.SUPPORT:
            # Support is a special case of distance constraint (on top of)
            return self._create_support_constraint(constraint_id, relationship)

        return None

    def _create_distance_constraint(
        self, constraint_id: str, relationship: SemanticRelationship
    ) -> DistanceConstraint:
        """Create a distance constraint from a relationship.

        Args:
            constraint_id: Unique constraint ID
            relationship: Semantic relationship

        Returns:
            DistanceConstraint object
        """
        rel_type = relationship.relationship_type.lower()

        # Get default parameters for this relationship type
        default_params = self.DEFAULT_RELATIONSHIP_PARAMS.get(rel_type, {})

        # Merge with relationship parameters
        params = {**default_params, **relationship.parameters}

        # Extract distance parameters
        min_distance = params.get("min_distance", 0.0)
        max_distance = params.get("max_distance", float('inf'))

        # Get measurement type
        measurement_str = params.get("measurement_type", "center_to_center")
        try:
            measurement_type = DistanceMeasurement(measurement_str)
        except ValueError:
            measurement_type = DistanceMeasurement.CENTER_TO_CENTER

        # Get priority and hard/soft flag
        priority = params.get("priority", 1.0)
        is_hard = params.get("is_hard", True)

        return DistanceConstraint(
            id=constraint_id,
            constraint_type=ConstraintType.DISTANCE,
            source_object=relationship.source_object,
            target_object=relationship.target_object,
            min_distance=min_distance,
            max_distance=max_distance,
            measurement_type=measurement_type,
            priority=priority,
            is_hard=is_hard,
        )

    def _create_alignment_constraint(
        self, constraint_id: str, relationship: SemanticRelationship
    ) -> AlignmentConstraint:
        """Create an alignment constraint from a relationship.

        Args:
            constraint_id: Unique constraint ID
            relationship: Semantic relationship

        Returns:
            AlignmentConstraint object
        """
        params = relationship.parameters

        # Get alignment axis
        axis_str = params.get("axis", "x")
        try:
            alignment_axis = Axis(axis_str.lower())
        except ValueError:
            alignment_axis = Axis.X

        # Get alignment type
        alignment_type_str = params.get("alignment_type", "center")
        try:
            alignment_type = AlignmentType(alignment_type_str.lower())
        except ValueError:
            alignment_type = AlignmentType.CENTER

        # Get tolerance
        tolerance = params.get("tolerance", 0.01)

        # Get priority and hard/soft flag
        priority = params.get("priority", 1.0)
        is_hard = params.get("is_hard", True)

        return AlignmentConstraint(
            id=constraint_id,
            constraint_type=ConstraintType.ALIGNMENT,
            source_object=relationship.source_object,
            target_object=relationship.target_object,
            alignment_axis=alignment_axis,
            alignment_type=alignment_type,
            tolerance=tolerance,
            priority=priority,
            is_hard=is_hard,
        )

    def _create_facing_constraint(
        self, constraint_id: str, relationship: SemanticRelationship
    ) -> FacingConstraint:
        """Create a facing constraint from a relationship.

        Args:
            constraint_id: Unique constraint ID
            relationship: Semantic relationship

        Returns:
            FacingConstraint object
        """
        params = relationship.parameters
        rel_type = relationship.relationship_type.lower()

        # Determine facing type
        if rel_type == "facing":
            facing_type = FacingType.TOWARDS
        elif rel_type == "facing_away":
            facing_type = FacingType.AWAY_FROM
        elif rel_type == "parallel_to":
            facing_type = FacingType.PARALLEL
        elif rel_type == "perpendicular_to":
            facing_type = FacingType.PERPENDICULAR
        else:
            facing_type = FacingType.TOWARDS

        # Get facing direction
        facing_direction = tuple(params.get("facing_direction", [1.0, 0.0, 0.0]))

        # Get angle tolerance
        angle_tolerance = params.get("angle_tolerance", 10.0)

        # Get priority and hard/soft flag
        priority = params.get("priority", 1.0)
        is_hard = params.get("is_hard", True)

        return FacingConstraint(
            id=constraint_id,
            constraint_type=ConstraintType.FACING,
            source_object=relationship.source_object,
            target_object=relationship.target_object,
            facing_direction=facing_direction,
            angle_tolerance=angle_tolerance,
            facing_type=facing_type,
            priority=priority,
            is_hard=is_hard,
        )

    def _create_containment_constraint(
        self, constraint_id: str, relationship: SemanticRelationship
    ) -> ContainmentConstraint:
        """Create a containment constraint from a relationship.

        Args:
            constraint_id: Unique constraint ID
            relationship: Semantic relationship

        Returns:
            ContainmentConstraint object
        """
        params = relationship.parameters

        # Get boundary type
        boundary_type = params.get("boundary_type", "box")

        # Get boundary parameters
        boundary_params = params.get("boundary_params", {})

        # Get margin
        margin = params.get("margin", 0.0)

        # Get allow_partial flag
        allow_partial = params.get("allow_partial", False)

        # Get priority and hard/soft flag
        priority = params.get("priority", 1.0)
        is_hard = params.get("is_hard", True)

        return ContainmentConstraint(
            id=constraint_id,
            constraint_type=ConstraintType.CONTAINMENT,
            source_object=relationship.source_object,
            target_object=relationship.target_object,
            boundary_type=boundary_type,
            boundary_params=boundary_params,
            margin=margin,
            allow_partial=allow_partial,
            priority=priority,
            is_hard=is_hard,
        )

    def _create_support_constraint(
        self, constraint_id: str, relationship: SemanticRelationship
    ) -> Constraint:
        """Create a support constraint (object on top of another).

        Args:
            constraint_id: Unique constraint ID
            relationship: Semantic relationship

        Returns:
            Constraint object
        """
        params = relationship.parameters

        # Get priority and hard/soft flag
        priority = params.get("priority", 1.0)
        is_hard = params.get("is_hard", True)

        # Support is represented as a base constraint with SUPPORT type
        return Constraint(
            id=constraint_id,
            constraint_type=ConstraintType.SUPPORT,
            source_object=relationship.source_object,
            target_object=relationship.target_object,
            parameters=params,
            priority=priority,
            is_hard=is_hard,
        )

    def _parse_explicit_constraints(
        self, constraints_data: List[Dict[str, Any]]
    ) -> List[Constraint]:
        """Parse explicit constraints from plan data.

        Args:
            constraints_data: List of constraint dictionaries

        Returns:
            List of Constraint objects
        """
        constraints = []

        for constraint_dict in constraints_data:
            try:
                constraint = create_constraint(constraint_dict)
                constraints.append(constraint)
            except (ValueError, KeyError) as e:
                # Skip invalid constraints but log the error
                # In production, this should use proper logging
                print(f"Warning: Failed to parse constraint: {e}")
                continue

        return constraints

    def _generate_constraint_id(self) -> str:
        """Generate a unique constraint ID.

        Returns:
            Unique constraint ID string
        """
        constraint_id = f"constraint_{self._constraint_id_counter}"
        self._constraint_id_counter += 1
        return constraint_id

    def validate_plan(self, plan: ParsedPlan) -> List[str]:
        """Validate a parsed plan for semantic correctness.

        Args:
            plan: Parsed plan to validate

        Returns:
            List of validation warnings (empty if valid)
        """
        warnings = []

        # Check that all constraint references point to valid objects
        object_ids = {obj["id"] for obj in plan.objects}

        for constraint in plan.constraints:
            if constraint.source_object not in object_ids:
                warnings.append(
                    f"Constraint {constraint.id} references unknown source object: "
                    f"{constraint.source_object}"
                )

            if (
                constraint.target_object is not None
                and constraint.target_object not in object_ids
            ):
                warnings.append(
                    f"Constraint {constraint.id} references unknown target object: "
                    f"{constraint.target_object}"
                )

        # Check for conflicting hard constraints
        # (This is a simplified check - full conflict detection is in Constraint Engine)
        distance_constraints = {}
        for constraint in plan.constraints:
            if isinstance(constraint, DistanceConstraint) and constraint.is_hard:
                key = (constraint.source_object, constraint.target_object)
                if key in distance_constraints:
                    prev = distance_constraints[key]
                    # Check if ranges overlap
                    if (
                        constraint.min_distance > prev.max_distance
                        or constraint.max_distance < prev.min_distance
                    ):
                        warnings.append(
                            f"Conflicting distance constraints between "
                            f"{constraint.source_object} and {constraint.target_object}"
                        )
                else:
                    distance_constraints[key] = constraint

        return warnings
