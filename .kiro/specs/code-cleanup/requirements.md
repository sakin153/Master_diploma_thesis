# Requirements Document: Legacy Code Cleanup

## Introduction

This document specifies requirements for cleaning up legacy code, temporary files, and outdated documentation from the world-creator project. The world-creator project is a 3D scene generation system using the Universal Placement System. Over time, the project has accumulated legacy documentation, temporary test scripts, unused code modules, and obsolete directories that need to be removed to improve maintainability and clarity.

The cleanup will remove all legacy artifacts while preserving the active codebase, official test suite, current documentation, and project infrastructure.

## Glossary

- **Cleanup_System**: The automated system responsible for identifying and removing legacy files
- **Legacy_File**: A file that is no longer used in the current project workflow (outdated documentation, temporary test scripts, or unused code)
- **Active_Code**: Code that is currently used in the production system (main.py, creator/runner.py, Universal Placement System modules)
- **Official_Test_Suite**: The test files located in the tests/ directory that are part of the formal testing infrastructure
- **Temporary_Test_Script**: A test script in the root directory that was created for ad-hoc testing and is not part of the Official_Test_Suite
- **Legacy_Documentation**: Markdown files containing outdated information that has been superseded by current documentation
- **Current_Documentation**: Active documentation files (README_RUN.md, QUICKSTART.md, USAGE_GUIDE.md, RUN_TEST.md)
- **Project_Infrastructure**: Essential project files and directories (.git/, .venv/, .github/, configuration files)
- **Legacy_Directory**: A directory containing only outdated or unused content (project/, docs/, examples/, .claude/)
- **Backup_Archive**: A compressed archive of removed files for recovery purposes

## Requirements

### Requirement 1: Identify Legacy Documentation Files

**User Story:** As a developer, I want to remove outdated documentation files, so that the project only contains current and accurate documentation.

#### Acceptance Criteria

1. THE Cleanup_System SHALL identify all Legacy_Documentation files in the root directory
2. THE Cleanup_System SHALL classify the following files as Legacy_Documentation: CHANGES_is_static.md, CHEATSHEET.md, CLAUDE.md, FIX_SUMMARY.md, INTEGRATION_PLAN.md, INTEGRATION_STATUS.md, MUJOCO_INTEGRATION.md, MUJOCO_LOG.TXT, PLACEMENT_FIXES.md, QUICK_TEST_GUIDE.md, SIMPLE_INTEGRATION.md, TASK_2_COMPLETION_SUMMARY.md, TASK_7.1_IMPLEMENTATION.md, TASK_7.1_SUMMARY.md, BUGFIX_STATUS.md, PHYSICS_BUG_ANALYSIS.md
3. THE Cleanup_System SHALL classify the following files as Legacy_Documentation: pipeline.md, integration_check_result.txt, list_items_objaverse.txt, scene_graph_latest.json, test_results.txt, уровень_работы.txt
4. THE Cleanup_System SHALL preserve Current_Documentation files: README_RUN.md, QUICKSTART.md, USAGE_GUIDE.md, RUN_TEST.md
5. FOR ALL identified Legacy_Documentation files, THE Cleanup_System SHALL verify they are not referenced by Active_Code before removal

### Requirement 2: Identify Temporary Test Scripts

**User Story:** As a developer, I want to remove temporary test scripts from the root directory, so that only the Official_Test_Suite remains for testing.

#### Acceptance Criteria

1. THE Cleanup_System SHALL identify all Temporary_Test_Script files in the root directory
2. THE Cleanup_System SHALL classify the following files as Temporary_Test_Script: example_usage.py, main_universal.py, mujoco_simulation_demo.py, quick_test.py, run_mujoco_viewer.py, run_property_tests.sh, run_tests.py, simple_test.py, test_apples_debug.py, test_final.py, test_integration.py, test_placement_fixes.py, test_quick.py, test_robot.py, test_scene_simple.py, test_size_preservation.py, test_with_mujoco.py, verify_integration.py, verify_tests.py, visualize_scene_graph.py
3. THE Cleanup_System SHALL preserve the Official_Test_Suite in the tests/ directory
4. THE Cleanup_System SHALL preserve save_scene_for_analysis.py as it is referenced by main.py
5. FOR ALL identified Temporary_Test_Script files, THE Cleanup_System SHALL verify they are not imported by Active_Code before removal

### Requirement 3: Identify Legacy Code Modules

**User Story:** As a developer, I want to remove duplicate and unused code modules, so that the codebase contains only active implementation.

#### Acceptance Criteria

1. THE Cleanup_System SHALL identify legacy code modules in the creator/ directory
2. THE Cleanup_System SHALL classify creator/runner_universal.py as a legacy module duplicate of creator/runner.py
3. THE Cleanup_System SHALL preserve creator/runner.py as the active runner implementation
4. WHEN a module is classified as legacy, THE Cleanup_System SHALL verify no Active_Code imports it
5. IF Active_Code imports a legacy module, THEN THE Cleanup_System SHALL report the dependency and skip removal

### Requirement 4: Identify Legacy Directories

**User Story:** As a developer, I want to remove obsolete directories, so that the project structure is clean and understandable.

#### Acceptance Criteria

1. THE Cleanup_System SHALL identify Legacy_Directory entries in the project root
2. THE Cleanup_System SHALL classify the following as Legacy_Directory: project/, docs/, examples/, .claude/
3. THE Cleanup_System SHALL preserve Project_Infrastructure directories: .git/, .venv/, .github/, .hypothesis/, .pytest_cache/, .kiro/
4. THE Cleanup_System SHALL preserve active resource directories: assets/, robot_assets/, creator/, tests/
5. FOR ALL Legacy_Directory entries, THE Cleanup_System SHALL verify no Active_Code references files within them before removal

### Requirement 5: Identify Temporary Data Directories

**User Story:** As a developer, I want to clean temporary cache and old test data, so that the project does not accumulate unnecessary files.

#### Acceptance Criteria

1. THE Cleanup_System SHALL identify temporary data directories
2. THE Cleanup_System SHALL classify .cache_ciare/ as a temporary cache directory
3. THE Cleanup_System SHALL identify saved_scenes/ as containing temporary test data
4. WHEN cleaning saved_scenes/, THE Cleanup_System SHALL preserve the directory structure but remove old scene subdirectories
5. THE Cleanup_System SHALL define "old scene" as a scene subdirectory older than 30 days

### Requirement 6: Create Backup Before Removal

**User Story:** As a developer, I want a backup of removed files, so that I can recover them if needed.

#### Acceptance Criteria

1. WHEN removing Legacy_File entries, THE Cleanup_System SHALL create a Backup_Archive before deletion
2. THE Backup_Archive SHALL contain all files to be removed with their original directory structure preserved
3. THE Backup_Archive SHALL be named with the format "legacy_backup_YYYYMMDD_HHMMSS.tar.gz"
4. THE Backup_Archive SHALL be stored in the project root directory
5. WHEN the Backup_Archive is created successfully, THE Cleanup_System SHALL log the archive path and size

### Requirement 7: Remove Legacy Files Safely

**User Story:** As a developer, I want legacy files removed safely, so that no active code is broken.

#### Acceptance Criteria

1. WHEN all Legacy_File entries are identified and verified, THE Cleanup_System SHALL remove them from the filesystem
2. THE Cleanup_System SHALL remove files only after Backup_Archive creation succeeds
3. IF a file removal fails, THEN THE Cleanup_System SHALL log the error and continue with remaining files
4. THE Cleanup_System SHALL remove empty directories after file removal
5. WHEN removal is complete, THE Cleanup_System SHALL generate a removal report listing all deleted files

### Requirement 8: Verify Active Code Integrity

**User Story:** As a developer, I want to verify that active code still works after cleanup, so that I know the cleanup was safe.

#### Acceptance Criteria

1. WHEN file removal is complete, THE Cleanup_System SHALL verify Active_Code integrity
2. THE Cleanup_System SHALL execute "python -m pytest tests/" to run the Official_Test_Suite
3. IF the Official_Test_Suite passes, THEN THE Cleanup_System SHALL report cleanup success
4. IF the Official_Test_Suite fails, THEN THE Cleanup_System SHALL report which tests failed and recommend restoring from Backup_Archive
5. THE Cleanup_System SHALL verify that main.py can be imported without errors

### Requirement 9: Generate Cleanup Report

**User Story:** As a developer, I want a detailed cleanup report, so that I know exactly what was removed and the project state.

#### Acceptance Criteria

1. WHEN cleanup is complete, THE Cleanup_System SHALL generate a cleanup report
2. THE cleanup report SHALL list all removed Legacy_Documentation files with their sizes
3. THE cleanup report SHALL list all removed Temporary_Test_Script files with their sizes
4. THE cleanup report SHALL list all removed Legacy_Directory entries with their total sizes
5. THE cleanup report SHALL include total space freed in megabytes
6. THE cleanup report SHALL include the Backup_Archive location and size
7. THE cleanup report SHALL include the Official_Test_Suite execution result
8. THE cleanup report SHALL be saved as "CLEANUP_REPORT.md" in the project root

### Requirement 10: Preserve Project Configuration

**User Story:** As a developer, I want all project configuration files preserved, so that the development environment remains functional.

#### Acceptance Criteria

1. THE Cleanup_System SHALL preserve all configuration files in the project root
2. THE Cleanup_System SHALL preserve the following files: .flake8, .gitignore, pyproject.toml, requirements-main.txt
3. THE Cleanup_System SHALL preserve main.py as the primary entry point
4. THE Cleanup_System SHALL preserve all files in .kiro/specs/ directory
5. THE Cleanup_System SHALL preserve all files in Project_Infrastructure directories
