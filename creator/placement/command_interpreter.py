"""Command Interpreter for spatial commands.

This module provides lightweight validation and extraction of precise numerical
constraints from natural language commands. It works as a validator on top of
LLM-generated plans to ensure precise specifications are not lost.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any


class PositioningType(Enum):
    """Type of positioning specification."""
    PRECISE = "precise"  # Exact numbers: "1.5 метра", "45 градусов"
    APPROXIMATE = "approximate"  # Relative: "рядом", "недалеко"
    ANGULAR = "angular"  # Corner positions: "в углу", "в правом верхнем углу"


@dataclass
class PreciseConstraint:
    """Extracted precise numerical constraint.
    
    Attributes:
        constraint_type: Type of constraint (distance, angle, position)
        value: Numerical value extracted
        unit: Unit of measurement (meters, degrees, etc.)
        original_text: Original text fragment
        source_object: Source object (if specified)
        target_object: Target object (if specified)
    """
    constraint_type: str
    value: float
    unit: str
    original_text: str
    source_object: Optional[str] = None
    target_object: Optional[str] = None


@dataclass
class SpatialCommand:
    """Parsed spatial command.
    
    Attributes:
        original_text: Original command text
        objects: List of mentioned objects
        positioning_type: Type of positioning (precise/approximate/angular)
        precise_constraints: List of extracted precise constraints
        relative_positions: List of relative position keywords
        metadata: Additional extracted information
    """
    original_text: str
    objects: List[str]
    positioning_type: PositioningType
    precise_constraints: List[PreciseConstraint]
    relative_positions: List[str]
    metadata: Dict[str, Any]


class CommandInterpreter:
    """Lightweight interpreter for spatial commands.
    
    This class extracts precise numerical constraints from natural language
    and validates that LLM-generated plans include them as hard constraints.
    It does NOT replace LLM - it works as a safety layer on top.
    """
    
    # Regex patterns for extracting precise measurements
    DISTANCE_PATTERNS = [
        # Russian patterns
        r'(?:на\s+расстоянии\s+)?(\d+(?:[.,]\d+)?)\s*(?:метр|м|метра|метров)',
        r'(?:в\s+)?(\d+(?:[.,]\d+)?)\s*(?:метр|м|метра|метров)(?:\s+от)?',
        r'(\d+(?:[.,]\d+)?)\s*(?:см|сантиметр|сантиметра|сантиметров)',
        # English patterns
        r'(?:at\s+)?(\d+(?:[.,]\d+)?)\s*(?:meter|meters|m)',
        r'(\d+(?:[.,]\d+)?)\s*(?:cm|centimeter|centimeters)',
    ]
    
    ANGLE_PATTERNS = [
        # Russian patterns
        r'(?:под\s+углом\s+)?(\d+(?:[.,]\d+)?)\s*(?:градус|градуса|градусов|°)',
        r'(?:повернуть\s+на\s+)?(\d+(?:[.,]\d+)?)\s*(?:градус|градуса|градусов|°)',
        # English patterns
        r'(?:at\s+)?(\d+(?:[.,]\d+)?)\s*(?:degree|degrees|°)',
        r'(?:rotate\s+)?(\d+(?:[.,]\d+)?)\s*(?:degree|degrees|°)',
    ]
    
    # Relative position keywords
    RELATIVE_KEYWORDS = {
        # Russian
        'справа': 'right',
        'слева': 'left',
        'спереди': 'front',
        'сзади': 'back',
        'рядом': 'near',
        'около': 'near',
        'возле': 'near',
        'недалеко': 'near',
        'по центру': 'center',
        'в центре': 'center',
        'в углу': 'corner',
        'у стены': 'wall',
        # English
        'right': 'right',
        'left': 'left',
        'front': 'front',
        'back': 'back',
        'near': 'near',
        'beside': 'near',
        'next to': 'near',
        'center': 'center',
        'corner': 'corner',
        'wall': 'wall',
    }
    
    # Angular position keywords
    ANGULAR_KEYWORDS = [
        'в углу', 'в правом углу', 'в левом углу',
        'в верхнем углу', 'в нижнем углу',
        'в правом верхнем углу', 'в правом нижнем углу',
        'в левом верхнем углу', 'в левом нижнем углу',
        'corner', 'right corner', 'left corner',
        'top corner', 'bottom corner',
        'top right corner', 'top left corner',
        'bottom right corner', 'bottom left corner',
    ]
    
    def parse_spatial_command(self, text: str) -> SpatialCommand:
        """Parse spatial command from natural language.
        
        This method extracts:
        - Precise numerical constraints (distances, angles)
        - Relative position keywords
        - Positioning type classification
        
        Args:
            text: Natural language command
            
        Returns:
            SpatialCommand with extracted information
        """
        text_lower = text.lower()
        
        # Extract precise constraints
        precise_constraints = self.extract_precise_constraints(text)
        
        # Extract relative positions
        relative_positions = []
        for keyword, normalized in self.RELATIVE_KEYWORDS.items():
            if keyword in text_lower:
                relative_positions.append(normalized)
        
        # Classify positioning type
        positioning_type = self.classify_positioning_type(text_lower)
        
        # Extract object mentions (simple heuristic - can be improved)
        objects = self._extract_objects(text_lower)
        
        return SpatialCommand(
            original_text=text,
            objects=objects,
            positioning_type=positioning_type,
            precise_constraints=precise_constraints,
            relative_positions=list(set(relative_positions)),  # Remove duplicates
            metadata={}
        )
    
    def extract_precise_constraints(
        self,
        text: str
    ) -> List[PreciseConstraint]:
        """Extract precise numerical constraints from text.
        
        Finds patterns like:
        - "0.5 метра", "1.5м", "50 см"
        - "45 градусов", "90°"
        
        Args:
            text: Text to extract from
            
        Returns:
            List of PreciseConstraint objects
        """
        constraints = []
        
        # Extract distance constraints
        for pattern in self.DISTANCE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value_str = match.group(1).replace(',', '.')
                value = float(value_str)
                
                # Determine unit
                unit = 'meters'
                if 'см' in match.group(0).lower() or 'cm' in match.group(0).lower():
                    unit = 'centimeters'
                    value = value / 100.0  # Convert to meters
                
                constraints.append(PreciseConstraint(
                    constraint_type='distance',
                    value=value,
                    unit=unit,
                    original_text=match.group(0)
                ))
        
        # Extract angle constraints
        for pattern in self.ANGLE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value_str = match.group(1).replace(',', '.')
                value = float(value_str)
                
                constraints.append(PreciseConstraint(
                    constraint_type='angle',
                    value=value,
                    unit='degrees',
                    original_text=match.group(0)
                ))
        
        return constraints
    
    def classify_positioning_type(self, text: str) -> PositioningType:
        """Classify the type of positioning specification.
        
        Args:
            text: Text to classify (should be lowercase)
            
        Returns:
            PositioningType enum value
        """
        # Check for angular positions first (most specific)
        for keyword in self.ANGULAR_KEYWORDS:
            if keyword in text:
                return PositioningType.ANGULAR
        
        # Check for precise specifications (numbers)
        has_distance = any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in self.DISTANCE_PATTERNS
        )
        has_angle = any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in self.ANGLE_PATTERNS
        )
        
        if has_distance or has_angle:
            return PositioningType.PRECISE
        
        # Default to approximate
        return PositioningType.APPROXIMATE
    
    def validate_llm_plan(
        self,
        original_text: str,
        llm_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate that LLM plan includes all precise constraints.
        
        This method checks if the LLM-generated plan includes all precise
        numerical constraints from the original text. If any are missing,
        they are added as hard constraints.
        
        Args:
            original_text: Original user command
            llm_plan: LLM-generated plan (dict with 'constraints' key)
            
        Returns:
            Updated plan with missing precise constraints added
        """
        # Parse original command
        command = self.parse_spatial_command(original_text)
        
        # If no precise constraints, return plan as-is
        if not command.precise_constraints:
            return llm_plan
        
        # Check which precise constraints are missing
        plan_constraints = llm_plan.get('constraints', [])
        
        for precise in command.precise_constraints:
            # Check if this constraint is already in the plan
            found = False
            for plan_constraint in plan_constraints:
                if self._constraint_matches(precise, plan_constraint):
                    # Ensure it's marked as hard constraint
                    plan_constraint['is_hard'] = True
                    found = True
                    break
            
            # If not found, add it
            if not found:
                new_constraint = self._precise_to_constraint(precise)
                plan_constraints.append(new_constraint)
        
        llm_plan['constraints'] = plan_constraints
        return llm_plan
    
    def _extract_objects(self, text: str) -> List[str]:
        """Extract object mentions from text (simple heuristic).
        
        This is a placeholder - in practice, LLM does this better.
        """
        # Common object types
        objects = [
            'стол', 'стул', 'диван', 'кровать', 'шкаф', 'полка',
            'лампа', 'растение', 'коробка', 'ящик',
            'table', 'chair', 'sofa', 'bed', 'shelf', 'lamp',
            'plant', 'box', 'crate'
        ]
        
        found = []
        for obj in objects:
            if obj in text:
                found.append(obj)
        
        return found
    
    def _constraint_matches(
        self,
        precise: PreciseConstraint,
        plan_constraint: Dict[str, Any]
    ) -> bool:
        """Check if a plan constraint matches a precise constraint."""
        if precise.constraint_type == 'distance':
            if plan_constraint.get('type') == 'distance':
                # Check if values are close (within 1cm tolerance)
                plan_value = plan_constraint.get('distance', 0)
                return abs(plan_value - precise.value) < 0.01
        
        elif precise.constraint_type == 'angle':
            if plan_constraint.get('type') == 'orientation':
                # Check if angles are close (within 1 degree tolerance)
                plan_value = plan_constraint.get('angle', 0)
                return abs(plan_value - precise.value) < 1.0
        
        return False
    
    def _precise_to_constraint(
        self,
        precise: PreciseConstraint
    ) -> Dict[str, Any]:
        """Convert PreciseConstraint to plan constraint dict."""
        if precise.constraint_type == 'distance':
            return {
                'type': 'distance',
                'distance': precise.value,
                'is_hard': True,
                'source': 'command_interpreter',
                'original_text': precise.original_text
            }
        
        elif precise.constraint_type == 'angle':
            return {
                'type': 'orientation',
                'angle': precise.value,
                'is_hard': True,
                'source': 'command_interpreter',
                'original_text': precise.original_text
            }
        
        return {}
