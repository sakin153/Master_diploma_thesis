"""Schema validation for semantic plans.

This module validates that semantic plans follow the defined JSON schema
before processing. It checks required fields, data types, and references.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ValidationResult:
    """Result of schema validation.
    
    Attributes:
        success: True if validation passed, False otherwise
        errors: List of error messages (empty if success=True)
    """
    success: bool
    errors: List[str]


class SchemaValidator:
    """Validates semantic plans against the defined schema.
    
    The validator checks:
    - Required top-level fields (schema_version, objects, room_size)
    - Required object fields (id, Model, type, size, is_static)
    - Position specification (either absolute OR relative, not both/neither)
    - Orientation specification (either absolute OR relative, not both/neither)
    - Reference validity (relative_to and facing reference existing object IDs)
    
    Note: model_loc is NOT required - the system adds it automatically after validation.
    """
    
    SUPPORTED_SCHEMA_VERSIONS = ["1.0"]
    
    def validate_plan(self, semantic_plan: Dict[str, Any]) -> ValidationResult:
        """Validate semantic plan against schema.
        
        Args:
            semantic_plan: Semantic plan dictionary to validate
            
        Returns:
            ValidationResult with success flag and error messages
        """
        errors = []
        
        # Validate top-level fields
        if not isinstance(semantic_plan, dict):
            return ValidationResult(success=False, errors=["Semantic plan must be a dictionary"])
        
        # Check required top-level fields
        if "schema_version" not in semantic_plan:
            errors.append("Missing required field: schema_version")
        elif semantic_plan["schema_version"] not in self.SUPPORTED_SCHEMA_VERSIONS:
            errors.append(
                f"Unsupported schema version: {semantic_plan['schema_version']}. "
                f"Supported versions: {self.SUPPORTED_SCHEMA_VERSIONS}"
            )
        
        if "objects" not in semantic_plan:
            errors.append("Missing required field: objects")
        elif not isinstance(semantic_plan["objects"], list):
            errors.append("Field 'objects' must be a list")
        elif len(semantic_plan["objects"]) == 0:
            errors.append("Field 'objects' must contain at least one object")
        
        if "room_size" not in semantic_plan:
            errors.append("Missing required field: room_size")
        elif not isinstance(semantic_plan["room_size"], dict):
            errors.append("Field 'room_size' must be a dictionary")
        else:
            # Validate room_size has required dimensions
            room_size = semantic_plan["room_size"]
            for dim in ["width", "length", "height"]:
                if dim not in room_size:
                    errors.append(f"room_size missing required dimension: {dim}")
        
        # If top-level validation failed, return early
        if errors:
            return ValidationResult(success=False, errors=errors)
        
        # Validate objects
        objects = semantic_plan["objects"]
        object_ids = set()
        
        for i, obj in enumerate(objects):
            obj_errors = self._validate_object(obj, i, object_ids)
            errors.extend(obj_errors)
            
            # Collect object IDs for reference validation
            if isinstance(obj, dict) and "id" in obj:
                object_ids.add(obj["id"])
        
        # Validate references (relative_to and facing must reference existing IDs)
        for i, obj in enumerate(objects):
            if not isinstance(obj, dict):
                continue
            
            # Validate relative position references
            if "position" in obj and isinstance(obj["position"], dict):
                if "relative" in obj["position"] and isinstance(obj["position"]["relative"], dict):
                    relative = obj["position"]["relative"]
                    if "relative_to" in relative:
                        target_id = relative["relative_to"]
                        if target_id not in object_ids:
                            errors.append(
                                f"Object at index {i}: relative position references "
                                f"non-existent object ID '{target_id}'"
                            )
            
            # Validate relative orientation references
            if "orientation" in obj and isinstance(obj["orientation"], dict):
                if "relative" in obj["orientation"] and isinstance(obj["orientation"]["relative"], dict):
                    relative = obj["orientation"]["relative"]
                    if "facing" in relative:
                        target_id = relative["facing"]
                        if target_id not in object_ids:
                            errors.append(
                                f"Object at index {i}: relative orientation references "
                                f"non-existent object ID '{target_id}'"
                            )
        
        if errors:
            return ValidationResult(success=False, errors=errors)
        
        return ValidationResult(success=True, errors=[])
    
    def _validate_object(
        self,
        obj: Any,
        index: int,
        seen_ids: set
    ) -> List[str]:
        """Validate a single object specification.
        
        Args:
            obj: Object dictionary to validate
            index: Object index in the list (for error messages)
            seen_ids: Set of already seen object IDs (for duplicate detection)
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        if not isinstance(obj, dict):
            errors.append(f"Object at index {index} must be a dictionary")
            return errors
        
        # Check required fields
        required_fields = ["id", "Model", "type", "size", "is_static", "position", "orientation"]
        for field in required_fields:
            if field not in obj:
                errors.append(f"Object at index {index} missing required field: {field}")
        
        # If basic fields are missing, return early
        if errors:
            return errors
        
        # Validate ID uniqueness
        obj_id = obj["id"]
        if obj_id in seen_ids:
            errors.append(f"Duplicate object ID: {obj_id}")
        
        # Validate size
        if not isinstance(obj["size"], dict):
            errors.append(f"Object at index {index}: 'size' must be a dictionary")
        else:
            for dim in ["width", "length", "height"]:
                if dim not in obj["size"]:
                    errors.append(f"Object at index {index}: size missing dimension '{dim}'")
        
        # Validate position (must be either absolute OR relative, not both/neither)
        position = obj["position"]
        if not isinstance(position, dict):
            errors.append(f"Object at index {index}: 'position' must be a dictionary")
        else:
            has_absolute = "absolute" in position and position["absolute"] is not None
            has_relative = "relative" in position and position["relative"] is not None
            
            if not has_absolute and not has_relative:
                errors.append(
                    f"Object at index {index}: position must specify either 'absolute' or 'relative'"
                )
            elif has_absolute and has_relative:
                errors.append(
                    f"Object at index {index}: position cannot specify both 'absolute' and 'relative'"
                )
        
        # Validate orientation (must be either absolute OR relative, not both/neither)
        orientation = obj["orientation"]
        if not isinstance(orientation, dict):
            errors.append(f"Object at index {index}: 'orientation' must be a dictionary")
        else:
            has_absolute = "absolute" in orientation and orientation["absolute"] is not None
            has_relative = "relative" in orientation and orientation["relative"] is not None
            
            if not has_absolute and not has_relative:
                errors.append(
                    f"Object at index {index}: orientation must specify either 'absolute' or 'relative'"
                )
            elif has_absolute and has_relative:
                errors.append(
                    f"Object at index {index}: orientation cannot specify both 'absolute' and 'relative'"
                )
        
        return errors
