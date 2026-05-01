"""Unit tests for Command Interpreter.

These tests verify that the Command Interpreter correctly extracts
precise numerical constraints from natural language commands.
"""

import pytest

from creator.placement.command_interpreter import (
    CommandInterpreter,
    PositioningType,
    PreciseConstraint,
)


class TestPreciseConstraintExtraction:
    """Test extraction of precise numerical constraints."""
    
    def test_extract_distance_meters_russian(self):
        """Test extraction of distance in meters (Russian)."""
        interpreter = CommandInterpreter()
        
        text = "поставь стол на расстоянии 1.5 метра от стены"
        constraints = interpreter.extract_precise_constraints(text)
        
        assert len(constraints) == 1
        assert constraints[0].constraint_type == 'distance'
        assert constraints[0].value == 1.5
        assert constraints[0].unit == 'meters'
    
    def test_extract_distance_centimeters_russian(self):
        """Test extraction of distance in centimeters (Russian)."""
        interpreter = CommandInterpreter()
        
        text = "стул в 50 см от стола"
        constraints = interpreter.extract_precise_constraints(text)
        
        assert len(constraints) == 1
        assert constraints[0].constraint_type == 'distance'
        assert constraints[0].value == 0.5  # Converted to meters
        assert constraints[0].unit == 'centimeters'
    
    def test_extract_distance_meters_english(self):
        """Test extraction of distance in meters (English)."""
        interpreter = CommandInterpreter()
        
        text = "place table at 2.5 meters from wall"
        constraints = interpreter.extract_precise_constraints(text)
        
        assert len(constraints) == 1
        assert constraints[0].constraint_type == 'distance'
        assert constraints[0].value == 2.5
        assert constraints[0].unit == 'meters'
    
    def test_extract_angle_russian(self):
        """Test extraction of angle (Russian)."""
        interpreter = CommandInterpreter()
        
        text = "повернуть стол на 45 градусов"
        constraints = interpreter.extract_precise_constraints(text)
        
        assert len(constraints) == 1
        assert constraints[0].constraint_type == 'angle'
        assert constraints[0].value == 45.0
        assert constraints[0].unit == 'degrees'
    
    def test_extract_angle_english(self):
        """Test extraction of angle (English)."""
        interpreter = CommandInterpreter()
        
        text = "rotate chair 90 degrees"
        constraints = interpreter.extract_precise_constraints(text)
        
        assert len(constraints) == 1
        assert constraints[0].constraint_type == 'angle'
        assert constraints[0].value == 90.0
        assert constraints[0].unit == 'degrees'
    
    def test_extract_multiple_constraints(self):
        """Test extraction of multiple constraints."""
        interpreter = CommandInterpreter()
        
        text = "стол на расстоянии 1 метр от стены под углом 45 градусов"
        constraints = interpreter.extract_precise_constraints(text)
        
        assert len(constraints) == 2
        
        distance_constraints = [c for c in constraints if c.constraint_type == 'distance']
        angle_constraints = [c for c in constraints if c.constraint_type == 'angle']
        
        assert len(distance_constraints) == 1
        assert distance_constraints[0].value == 1.0
        
        assert len(angle_constraints) == 1
        assert angle_constraints[0].value == 45.0
    
    def test_extract_no_constraints(self):
        """Test text with no precise constraints."""
        interpreter = CommandInterpreter()
        
        text = "поставь стол рядом с диваном"
        constraints = interpreter.extract_precise_constraints(text)
        
        assert len(constraints) == 0


class TestPositioningTypeClassification:
    """Test classification of positioning types."""
    
    def test_classify_precise_distance(self):
        """Test classification of precise distance specification."""
        interpreter = CommandInterpreter()
        
        text = "стол на расстоянии 1.5 метра от стены"
        pos_type = interpreter.classify_positioning_type(text.lower())
        
        assert pos_type == PositioningType.PRECISE
    
    def test_classify_precise_angle(self):
        """Test classification of precise angle specification."""
        interpreter = CommandInterpreter()
        
        text = "повернуть на 45 градусов"
        pos_type = interpreter.classify_positioning_type(text.lower())
        
        assert pos_type == PositioningType.PRECISE
    
    def test_classify_angular_russian(self):
        """Test classification of angular position (Russian)."""
        interpreter = CommandInterpreter()
        
        text = "стол в правом верхнем углу"
        pos_type = interpreter.classify_positioning_type(text.lower())
        
        assert pos_type == PositioningType.ANGULAR
    
    def test_classify_angular_english(self):
        """Test classification of angular position (English)."""
        interpreter = CommandInterpreter()
        
        text = "table in top right corner"
        pos_type = interpreter.classify_positioning_type(text.lower())
        
        assert pos_type == PositioningType.ANGULAR
    
    def test_classify_approximate(self):
        """Test classification of approximate position."""
        interpreter = CommandInterpreter()
        
        text = "стол рядом с диваном"
        pos_type = interpreter.classify_positioning_type(text.lower())
        
        assert pos_type == PositioningType.APPROXIMATE


class TestSpatialCommandParsing:
    """Test full spatial command parsing."""
    
    def test_parse_precise_command(self):
        """Test parsing command with precise constraints."""
        interpreter = CommandInterpreter()
        
        text = "поставь стол на расстоянии 1.5 метра справа от дивана"
        command = interpreter.parse_spatial_command(text)
        
        assert command.original_text == text
        assert command.positioning_type == PositioningType.PRECISE
        assert len(command.precise_constraints) == 1
        assert command.precise_constraints[0].value == 1.5
        assert 'right' in command.relative_positions
    
    def test_parse_approximate_command(self):
        """Test parsing command with approximate position."""
        interpreter = CommandInterpreter()
        
        text = "стул рядом со столом"
        command = interpreter.parse_spatial_command(text)
        
        assert command.positioning_type == PositioningType.APPROXIMATE
        assert len(command.precise_constraints) == 0
        assert 'near' in command.relative_positions
    
    def test_parse_angular_command(self):
        """Test parsing command with angular position."""
        interpreter = CommandInterpreter()
        
        text = "диван в левом нижнем углу комнаты"
        command = interpreter.parse_spatial_command(text)
        
        assert command.positioning_type == PositioningType.ANGULAR
        assert 'corner' in command.relative_positions
    
    def test_parse_combined_command(self):
        """Test parsing command with multiple specifications."""
        interpreter = CommandInterpreter()
        
        text = "стол по центру комнаты на расстоянии 2 метра от стены"
        command = interpreter.parse_spatial_command(text)
        
        assert command.positioning_type == PositioningType.PRECISE
        assert len(command.precise_constraints) == 1
        assert 'center' in command.relative_positions


class TestLLMPlanValidation:
    """Test validation of LLM-generated plans."""
    
    def test_validate_plan_with_missing_constraint(self):
        """Test that missing precise constraints are added."""
        interpreter = CommandInterpreter()
        
        original_text = "стол на расстоянии 1.5 метра от стены"
        llm_plan = {
            'objects': ['стол', 'стена'],
            'constraints': []
        }
        
        validated_plan = interpreter.validate_llm_plan(original_text, llm_plan)
        
        assert len(validated_plan['constraints']) == 1
        assert validated_plan['constraints'][0]['type'] == 'distance'
        assert validated_plan['constraints'][0]['distance'] == 1.5
        assert validated_plan['constraints'][0]['is_hard'] is True
    
    def test_validate_plan_with_existing_constraint(self):
        """Test that existing constraints are marked as hard."""
        interpreter = CommandInterpreter()
        
        original_text = "стол на расстоянии 1.5 метра от стены"
        llm_plan = {
            'objects': ['стол', 'стена'],
            'constraints': [{
                'type': 'distance',
                'distance': 1.5,
                'is_hard': False
            }]
        }
        
        validated_plan = interpreter.validate_llm_plan(original_text, llm_plan)
        
        assert len(validated_plan['constraints']) == 1
        assert validated_plan['constraints'][0]['is_hard'] is True
    
    def test_validate_plan_no_precise_constraints(self):
        """Test that plans without precise constraints are unchanged."""
        interpreter = CommandInterpreter()
        
        original_text = "стол рядом с диваном"
        llm_plan = {
            'objects': ['стол', 'диван'],
            'constraints': [{
                'type': 'proximity',
                'relation': 'near'
            }]
        }
        
        validated_plan = interpreter.validate_llm_plan(original_text, llm_plan)
        
        # Plan should be unchanged
        assert validated_plan == llm_plan


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_text(self):
        """Test parsing empty text."""
        interpreter = CommandInterpreter()
        
        command = interpreter.parse_spatial_command("")
        
        assert command.original_text == ""
        assert len(command.precise_constraints) == 0
        assert len(command.relative_positions) == 0
    
    def test_decimal_comma_format(self):
        """Test parsing numbers with comma as decimal separator."""
        interpreter = CommandInterpreter()
        
        text = "стол на расстоянии 1,5 метра"
        constraints = interpreter.extract_precise_constraints(text)
        
        assert len(constraints) == 1
        assert constraints[0].value == 1.5
    
    def test_multiple_same_constraints(self):
        """Test text with multiple instances of same constraint."""
        interpreter = CommandInterpreter()
        
        text = "стол 1 метр от стены и стул 2 метра от окна"
        constraints = interpreter.extract_precise_constraints(text)
        
        assert len(constraints) == 2
        assert constraints[0].value == 1.0
        assert constraints[1].value == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
