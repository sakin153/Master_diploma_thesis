# Implementation Plan: Semantic Plan Enforcement

## Overview

This implementation plan breaks down the Semantic Plan Enforcement feature into discrete coding tasks. The system enforces LLM-generated semantic plans through a strict pipeline: Schema Validation → Distance Resolution → Orientation Resolution → Placement Execution → Scene Assembly → Validation.

The implementation follows a bottom-up approach: core data models and utilities first, then individual pipeline components, then integration and validation. Each task builds on previous work, with checkpoints to ensure correctness before proceeding.

## Tasks

- [x] 1. Set up project structure and core data models
  - [x] 1.1 Create semantic plan enforcement module structure
    - Create `creator/placement/semantic_enforcement/` directory
    - Create `__init__.py` with module exports
    - Create `models.py` for data models
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_
  
  - [x] 1.2 Implement core data models in `models.py`
    - Implement `Position` dataclass (absolute and relative variants)
    - Implement `Orientation` dataclass (absolute and relative variants)
    - Implement `SemanticObject` dataclass with all required fields
    - Implement `SemanticPlan` dataclass with schema_version, room_size, objects
    - Implement `PlacedObject` dataclass with resolved absolute coordinates
    - Implement `PlacementSolution` dataclass
    - Implement `ValidationReport` dataclass
    - Add type hints and docstrings for all models
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 6.1, 6.2, 6.3, 6.4_
  
  - [ ]* 1.3 Write unit tests for data models
    - Test dataclass instantiation with valid data
    - Test dataclass validation (if using pydantic)
    - Test serialization/deserialization to/from JSON
    - Test edge cases (empty lists, None values)
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 2. Implement Schema Validator
  - [x] 2.1 Create `schema_validator.py` with SchemaValidator class
    - Implement `validate_plan()` method
    - Implement validation for required top-level fields (schema_version, objects, room_size)
    - Implement validation for each object's required fields (id, Model, type, size, is_static, model_loc)
    - Implement validation that position is either absolute OR relative (not both, not neither)
    - Implement validation that orientation is either absolute OR relative (not both, not neither)
    - Implement validation that relative position references existing object IDs
    - Implement validation that relative orientation references existing object IDs
    - Return `ValidationResult` with success flag and detailed error messages
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_
  
  - [ ]* 2.2 Write unit tests for SchemaValidator
    - Test valid semantic plans pass validation
    - Test missing required fields fail with correct error messages
    - Test invalid position specifications (both absolute and relative, or neither)
    - Test invalid orientation specifications (both absolute and relative, or neither)
    - Test invalid references (relative_to non-existent object)
    - Test edge cases (empty objects list, invalid schema_version)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [x] 3. Implement Distance Resolver
  - [x] 3.1 Create `distance_resolver.py` with DistanceResolver class
    - Implement `resolve_positions()` main method
    - Implement dependency graph building (which objects depend on which)
    - Implement topological sort for dependency resolution order
    - Implement circular dependency detection with clear error messages
    - Implement `_calculate_direction_vector()` for all direction types (front, back, left, right, front_left, front_right, back_left, back_right, above, below)
    - Implement `_get_reference_point()` for all reference point types (center, front_edge, back_edge, left_edge, right_edge, top_surface)
    - Implement position calculation: reference_point + (direction_vector * distance)
    - Preserve z-coordinate based on object type (floor objects at z=0, table-top objects at table surface height)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_
  
  - [ ]* 3.2 Write unit tests for DistanceResolver
    - Test absolute positions remain unchanged
    - Test relative positions correctly calculated for all direction types
    - Test reference points correctly calculated (center, edges, corners)
    - Test dependency resolution order (absolute first, then relative)
    - Test multi-level dependencies (A depends on B, B depends on C)
    - Test circular dependency detection
    - Test z-coordinate preservation for floor vs table-top objects
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 9.1, 9.2, 9.3, 9.4, 9.7_

- [x] 4. Implement Orientation Resolver
  - [x] 4.1 Create `orientation_resolver.py` with OrientationResolver class
    - Implement `resolve_orientations()` main method
    - Implement `_calculate_facing_angle()` for all facing directions (front, back, left_side, right_side)
    - Calculate angle using atan2 from source position to target position
    - Adjust angle based on target's orientation and facing_direction
    - Support facing_away modifier (add 180° to calculated angle)
    - Normalize angles to [0, 360) range
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_
  
  - [ ]* 4.2 Write unit tests for OrientationResolver
    - Test absolute orientations remain unchanged
    - Test relative orientations correctly calculated for all facing directions
    - Test facing_away modifier works correctly
    - Test angle normalization to [0, 360) range
    - Test orientation calculation accounts for target's orientation
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [x] 5. Checkpoint - Ensure core resolvers work correctly
  - Run all unit tests for SchemaValidator, DistanceResolver, OrientationResolver
  - Verify dependency resolution handles complex scenarios
  - Ensure all tests pass, ask the user if questions arise

- [x] 6. Implement Placement Executor
  - [x] 6.1 Create `placement_executor.py` with PlacementExecutor class
    - Implement `execute_placement()` method
    - Place ALL objects from semantic plan without omissions
    - Use exact positions from plan (no collision avoidance modifications)
    - Use exact orientations from plan (no automatic adjustments)
    - Preserve all metadata (uuid, model_loc, size, is_static) from SemanticPlan to PlacementSolution
    - Log warnings for potential collisions but still place objects as instructed
    - Verify PlacementSolution contains exactly the same number of objects as SemanticPlan with matching IDs
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_
  
  - [ ]* 6.2 Write unit tests for PlacementExecutor
    - Test all objects from semantic plan are placed
    - Test positions exactly match plan (no modifications)
    - Test orientations exactly match plan (no modifications)
    - Test metadata is preserved (uuid, model_loc, size, is_static)
    - Test object count and IDs match between input and output
    - Test warnings are logged for potential collisions
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_

- [x] 7. Implement Scene Assembly
  - [x] 7.1 Create `scene_assembly.py` with SceneAssembly class
    - Implement `assemble_scene()` method
    - Create MuJoCo XML root with worldbody element
    - Add floor and walls based on room_size
    - For each object in PlacementSolution: create body element with position (x, y, z) and orientation (euler angles)
    - Verify model_loc file exists before adding object to XML
    - Include 3D model via `<include file="..."/>` tag
    - Add `<freejoint/>` for dynamic objects (is_static=False)
    - Preserve object ID as name attribute in body element
    - Log error and skip object if model_loc file does not exist
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_
  
  - [ ]* 7.2 Write unit tests for SceneAssembly
    - Test valid MuJoCo XML is generated
    - Test all objects from PlacementSolution are included in XML
    - Test positions and orientations are correctly set in XML
    - Test model includes are present with correct paths
    - Test freejoint is added for dynamic objects only
    - Test object IDs are preserved as name attributes
    - Test error handling for missing model files
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

- [x] 8. Implement Pipeline Tracker
  - [x] 8.1 Create `pipeline_tracker.py` with PipelineTracker class
    - Implement `log_stage()` method to record object data at each pipeline stage
    - Implement `generate_trace()` method to create complete pipeline trace in JSON format
    - Implement `detect_changes()` method to identify added, removed, and modified objects between stages
    - Log object count and list of object IDs at each stage
    - Log object metadata (id, Model, position, orientation, size) at each stage
    - Include timestamps for each stage
    - Save pipeline trace file to configurable location with timestamp in filename
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_
  
  - [ ]* 8.2 Write unit tests for PipelineTracker
    - Test log_stage() records object data correctly
    - Test generate_trace() creates valid JSON with all stages
    - Test detect_changes() identifies added, removed, and modified objects
    - Test timestamps are included for each stage
    - Test trace file is saved with correct filename format
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.7_

- [x] 9. Checkpoint - Ensure pipeline components work correctly
  - Run all unit tests for PlacementExecutor, SceneAssembly, PipelineTracker
  - Verify objects are tracked correctly through all stages
  - Ensure all tests pass, ask the user if questions arise

- [ ] 10. Implement Constraint Validator
  - [ ] 10.1 Create `constraint_validator.py` with ConstraintValidator class
    - Implement `validate_scene()` method
    - Parse MuJoCo XML to extract object positions and orientations
    - Compare final XML with original SemanticPlan
    - Verify object count matches between plan and XML
    - Verify all planned objects exist in XML with matching IDs
    - Calculate position error for each object (distance between planned and actual in meters)
    - Calculate orientation error for each object (angular difference in degrees)
    - Mark objects with position error > 0.1m as "position mismatch"
    - Mark objects with orientation error > 5° as "orientation mismatch"
    - Calculate overall compliance score (percentage of objects placed correctly)
    - Generate ValidationReport with detailed errors and summary
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9_
  
  - [ ]* 10.2 Write unit tests for ConstraintValidator
    - Test object count validation
    - Test missing object detection
    - Test position error calculation
    - Test orientation error calculation
    - Test position mismatch threshold (0.1m)
    - Test orientation mismatch threshold (5°)
    - Test compliance score calculation
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9_

- [ ] 11. Implement Placement Report Generator
  - [ ] 11.1 Create `report_generator.py` with ReportGenerator class
    - Implement `generate_placement_report()` method
    - Include SemanticPlan summary (object count, room size)
    - Include PlacementSolution summary (placed objects, failed objects)
    - Include SceneAssembly summary (XML objects, missing objects)
    - For each object: include id, Model, planned position, actual position, position error, planned orientation, actual orientation, orientation error
    - Include list of objects that failed to place with reason for failure
    - Include list of objects missing from final XML with reason
    - Include overall statistics (total objects planned, total objects placed, total objects in XML, average position error, average orientation error)
    - Save report to configurable location with timestamp in filename
    - Generate report in JSON format
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8_
  
  - [ ]* 11.2 Write unit tests for ReportGenerator
    - Test report includes all required sections
    - Test object-level details are correct
    - Test statistics are calculated correctly
    - Test report is saved with correct filename format
    - Test JSON format is valid
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

- [x] 12. Integrate components into main pipeline
  - [x] 12.1 Create `pipeline.py` with SemanticEnforcementPipeline class
    - Implement `execute_pipeline()` method that orchestrates all components
    - Pipeline flow: SemanticPlan → SchemaValidator → DistanceResolver → OrientationResolver → PlacementExecutor → SceneAssembly → ConstraintValidator
    - Integrate PipelineTracker to log objects at each stage
    - Handle errors at each stage with appropriate error messages
    - Return final MuJoCo XML, ValidationReport, and PlacementReport
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.1, 3.1, 4.1, 5.1, 7.1, 8.1, 10.1_
  
  - [x] 12.2 Add pipeline integration to existing World Creator system
    - Identify where semantic plan is generated in existing codebase
    - Replace or augment existing placement logic with SemanticEnforcementPipeline
    - Ensure backward compatibility with existing scene generation
    - Add configuration option to enable/disable semantic enforcement
    - _Requirements: 1.1, 4.1, 5.1_

- [ ] 13. Checkpoint - Ensure full pipeline works end-to-end
  - Create test semantic plans with various complexity levels
  - Run full pipeline for each test plan
  - Verify all objects are placed correctly
  - Verify validation reports show high compliance scores
  - Ensure all tests pass, ask the user if questions arise

- [x] 14. Update LLM prompts for semantic plan generation
  - [x] 14.1 Create or update LLM prompt template for semantic plan generation
    - Define LLM role as "scene composer and architect"
    - Include JSON schema definition with all required fields
    - Provide examples of semantic plans for common scenarios (table with chairs, room with furniture)
    - Emphasize explicit instructions for every object (position, orientation, distances)
    - Include examples of relative positioning and orientation
    - Add instructions for handling user queries with relationships ("у каждого стола по 4 стула")
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_
  
  - [x] 14.2 Integrate semantic plan generation into LLM workflow
    - Update `creator/contexts_prompts/place.py` or relevant prompt file
    - Add semantic plan schema to LLM context
    - Configure LLM to output structured JSON matching SemanticPlan schema
    - Add validation feedback loop: if schema validation fails, send error back to LLM for correction
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 6.7_

- [ ] 15. Add error handling and logging
  - [ ] 15.1 Implement comprehensive error handling
    - Add try-except blocks for all critical operations
    - Define custom exception classes for different error types (SchemaValidationError, ResolutionError, AssemblyError, ValidationError)
    - Implement fail-fast strategy for critical errors (schema validation failure, circular dependencies, missing model files)
    - Implement warning strategy for non-critical issues (potential collisions, objects outside bounds)
    - Add detailed error messages with object IDs, expected vs actual values, and suggested fixes
    - _Requirements: 6.7, 9.4, 9.5, 9.8_
  
  - [ ] 15.2 Implement comprehensive logging
    - Add logging at DEBUG level for detailed calculations (position/orientation resolution)
    - Add logging at INFO level for pipeline stage transitions and object counts
    - Add logging at WARNING level for potential issues (collisions, out-of-bounds objects)
    - Add logging at ERROR level for failures (schema validation, missing files)
    - Configure logging to write to file and console
    - _Requirements: 4.6, 5.7, 7.2, 7.6_

- [ ] 16. Integration testing with real scenes
  - [ ]* 16.1 Write integration tests for simple scenes
    - Test scene with 2 objects (table and chair with relative positioning)
    - Test scene with absolute positioning only
    - Test scene with relative positioning only
    - Verify all objects present in final XML
    - Verify positions within 0.1m tolerance
    - Verify orientations within 5° tolerance
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 8.1_
  
  - [ ]* 16.2 Write integration tests for complex scenes
    - Test scene with 20+ objects (4 tables with 4 chairs each)
    - Test scene with multi-level dependencies (A depends on B, B depends on C)
    - Test scene with mixed absolute and relative positioning
    - Test universal relative placement (any object type relative to any other)
    - Verify all objects present in final XML
    - Verify high compliance score (>95%)
    - _Requirements: 1.4, 2.1, 3.1, 4.1, 5.1, 8.1, 9.6_
  
  - [ ]* 16.3 Write integration tests for edge cases
    - Test scene with circular dependencies (should fail with clear error)
    - Test scene with missing model files (should skip objects with errors)
    - Test scene with invalid references (should fail schema validation)
    - Test scene with objects outside room boundaries (should log warnings)
    - _Requirements: 5.6, 5.7, 6.5, 6.6, 9.4, 9.5, 9.8_

- [ ] 17. Final checkpoint and validation
  - Run full test suite (unit tests + integration tests)
  - Test with real user queries from requirements ("Простая столовая где есть 4 стола и 16 стульев")
  - Verify semantic plan is generated correctly by LLM
  - Verify all 20 objects (4 tables + 16 chairs) are placed correctly
  - Verify chairs face their respective tables
  - Verify validation report shows high compliance
  - Generate and review placement reports and pipeline traces
  - Ensure all tests pass, ask the user if questions arise

- [ ] 18. Documentation and cleanup
  - [ ] 18.1 Write module documentation
    - Add README.md in `creator/placement/semantic_enforcement/` explaining the system
    - Document JSON schema for SemanticPlan
    - Document pipeline flow and component responsibilities
    - Add usage examples with code snippets
    - Document configuration options
    - _Requirements: All_
  
  - [ ] 18.2 Add inline code documentation
    - Ensure all classes have docstrings
    - Ensure all public methods have docstrings with parameter and return type descriptions
    - Add comments for complex algorithms (dependency resolution, angle calculations)
    - _Requirements: All_
  
  - [ ] 18.3 Code cleanup and optimization
    - Remove debug print statements
    - Remove unused imports
    - Format code with black or similar formatter
    - Run linter (flake8, pylint) and fix issues
    - Optimize performance if needed (caching, parallel processing)
    - _Requirements: All_

## Notes

- Tasks marked with `*` are optional testing tasks and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- The implementation follows a bottom-up approach: data models → individual components → integration → LLM integration
- All components are designed to be modular and independently testable
- The system is designed as a "dumb executor" that faithfully implements LLM instructions without modifications
- Universal relative placement algorithm works for any object type combinations (no special-case logic)
- Property-based testing is NOT used because this is an infrastructure-like system with deterministic transformations
