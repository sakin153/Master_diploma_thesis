# Implementation Plan: Legacy Code Cleanup System

## Overview

This implementation plan creates an automated cleanup system that safely removes legacy files, temporary scripts, and obsolete directories from the world-creator project. The system follows a classify-verify-backup-remove workflow with comprehensive safety checks and backup creation before any file removal.

## Tasks

- [ ] 1. Set up project structure and configuration
  - Create cleanup_system/ directory for all cleanup modules
  - Create cleanup_config.py with all classification rules from requirements
  - Define all file lists (legacy documentation, temporary scripts, legacy modules, etc.)
  - Set up logging configuration
  - _Requirements: 1.2, 1.3, 2.2, 3.2, 4.2, 5.2, 10.2_

- [ ] 2. Implement data models
  - [ ] 2.1 Create data model classes
    - Implement ClassificationResult dataclass with all file categories
    - Implement DependencyReport dataclass for dependency analysis results
    - Implement BackupResult dataclass for backup operation results
    - Implement RemovalResult dataclass for removal operation results
    - Implement TestResult dataclass for test execution results
    - Implement CleanupResult dataclass for overall cleanup results
    - Add helper methods to each dataclass (all_files_to_remove, total_count, etc.)
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1_

  - [ ]* 2.2 Write unit tests for data models
    - Test dataclass instantiation and field validation
    - Test helper methods (all_files_to_remove, space_freed_mb, etc.)
    - Test edge cases (empty lists, zero values)
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1_

- [ ] 3. Implement File Classifier component
  - [ ] 3.1 Create FileClassifier class
    - Implement classify_files method to scan project and categorize all files
    - Implement is_legacy_documentation method using config file list
    - Implement is_temporary_test_script method using config file list
    - Implement is_legacy_module method using config file list
    - Implement is_legacy_directory method using config file list
    - Implement is_temporary_data method for cache and old scenes
    - Return ClassificationResult with all categorized files
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 3.2 Write unit tests for FileClassifier
    - Test classification of each file category
    - Test preservation of active files and current documentation
    - Test handling of missing files and permission errors
    - Test edge cases (empty directories, symlinks)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4_

- [ ] 4. Implement Dependency Analyzer component
  - [ ] 4.1 Create DependencyAnalyzer class
    - Implement find_imports method using AST parsing to extract import statements
    - Implement resolve_import_path method to map import names to file paths
    - Implement has_dependencies method to check if file is imported by active code
    - Implement analyze_dependencies method to verify files safe to remove
    - Scan all Python files in active code directories (creator/, tests/, main.py)
    - Build dependency graph and flag any files marked for removal that are imported
    - Return DependencyReport with safe and blocked removals
    - _Requirements: 1.5, 2.5, 3.4, 3.5, 4.5_

  - [ ]* 4.2 Write unit tests for DependencyAnalyzer
    - Test import extraction from Python files with various import styles
    - Test import path resolution for relative and absolute imports
    - Test dependency graph construction
    - Test detection of files with dependencies
    - Test handling of invalid Python syntax
    - _Requirements: 1.5, 2.5, 3.4, 3.5_

- [ ] 5. Checkpoint - Ensure classification and analysis work correctly
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement Backup Manager component
  - [ ] 6.1 Create BackupManager class
    - Implement generate_backup_name method with timestamp format
    - Implement create_backup method to create tar.gz archive
    - Preserve directory structure within archive
    - Include metadata file listing all archived files
    - Implement verify_backup method to check archive integrity
    - Check available disk space before creating archive
    - Return BackupResult with archive path, size, and success status
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 6.2 Write unit tests for BackupManager
    - Test backup archive creation with temporary test files
    - Test backup name generation with timestamp format
    - Test archive integrity verification
    - Test handling of large files
    - Test handling of special characters in filenames
    - Test disk space checking
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 7. Implement Removal Engine component
  - [ ] 7.1 Create RemovalEngine class
    - Implement remove_files method to delete files from filesystem
    - Verify backup exists before any removal
    - Log each file removal operation
    - Continue on individual failures and collect errors
    - Implement remove_empty_directories method to clean up after file removal
    - Never remove directories with remaining files
    - Implement clean_old_scenes method for date-based scene cleanup (>30 days)
    - Return RemovalResult with removed files, failures, and space freed
    - _Requirements: 5.4, 5.5, 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 7.2 Write unit tests for RemovalEngine
    - Test file removal using temporary test directories
    - Test empty directory removal
    - Test old scene cleanup with date-based filtering
    - Test error handling for permission denied
    - Test preservation of non-empty directories
    - Test backup verification before removal
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 8. Implement Reporter component
  - [ ] 8.1 Create Reporter class
    - Implement calculate_space_freed method to sum file sizes
    - Implement format_file_list method for Markdown formatting
    - Implement generate_cleanup_report method to create comprehensive report
    - Include summary statistics (files removed, space freed)
    - Include categorized lists of removed files with sizes
    - Include backup archive location and size
    - Include test suite execution results
    - Include any errors or warnings encountered
    - Save report as CLEANUP_REPORT.md in project root
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_

  - [ ]* 8.2 Write unit tests for Reporter
    - Test report generation with various inputs
    - Test space calculation with different file sizes
    - Test file list formatting with sizes
    - Test Markdown formatting
    - Test report content completeness
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_

- [ ] 9. Checkpoint - Ensure all components work in isolation
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Implement Cleanup Orchestrator
  - [ ] 10.1 Create CleanupOrchestrator class
    - Initialize all component instances (classifier, analyzer, backup_manager, removal_engine, reporter)
    - Implement execute_cleanup method to coordinate full workflow
    - Execute classification phase and handle errors
    - Execute dependency analysis phase and check for blockers
    - Execute backup phase and abort if backup fails (CRITICAL)
    - Execute removal phase and collect all errors
    - Execute verification phase by running pytest
    - Generate final cleanup report
    - Return CleanupResult with all operation results
    - Support dry-run mode that skips actual file removal
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 6.2, 7.1, 7.2, 8.1, 9.1_

  - [ ] 10.2 Implement verify_integrity method
    - Run "python -m pytest tests/" using subprocess
    - Capture test output and execution time
    - Parse test results to identify passed/failed tests
    - Verify main.py can be imported without errors
    - Return TestResult with pass/fail status and output
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 10.3 Write unit tests for CleanupOrchestrator
    - Test workflow coordination with mocked components
    - Test error handling in each phase
    - Test abort on backup failure
    - Test continuation on partial removal failure
    - Test dry-run mode
    - _Requirements: 6.2, 7.2, 7.3, 8.1, 8.2, 8.3, 8.4_

- [ ] 11. Implement error handling and custom exceptions
  - [ ] 11.1 Create custom exception classes
    - Implement CleanupError base exception
    - Implement BackupFailedError for backup failures (abort cleanup)
    - Implement RemovalError for file removal failures (continue with others)
    - Implement VerificationError for post-cleanup verification failures
    - Add error messages and context to each exception
    - _Requirements: 6.2, 7.3, 8.4_

  - [ ]* 11.2 Write unit tests for error handling
    - Test exception raising and catching
    - Test error message formatting
    - Test error recovery procedures
    - _Requirements: 6.2, 7.3, 8.4_

- [ ] 12. Create command-line interface
  - [ ] 12.1 Create cleanup_cli.py script
    - Add argparse for command-line arguments (--dry-run, --verbose, --help)
    - Initialize CleanupOrchestrator with project root and dry-run flag
    - Execute cleanup workflow
    - Display progress messages during execution
    - Display final cleanup report
    - Handle keyboard interrupts gracefully
    - Exit with appropriate status codes
    - _Requirements: 7.1, 7.2, 9.1_

  - [ ]* 12.2 Write integration tests for CLI
    - Test CLI execution with dry-run mode
    - Test CLI execution with verbose mode
    - Test CLI help output
    - Test CLI error handling
    - _Requirements: 7.1, 7.2, 9.1_

- [ ] 13. Checkpoint - Ensure end-to-end workflow works
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Create integration tests
  - [ ]* 14.1 Write end-to-end cleanup test
    - Create mock project structure with legacy and active files
    - Run complete cleanup workflow
    - Verify correct files removed
    - Verify backup created and contains correct files
    - Verify active files preserved
    - Verify report generated with correct content
    - Clean up test artifacts
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1_

  - [ ]* 14.2 Write dependency analysis integration test
    - Create test project with import dependencies
    - Verify files with dependencies are not removed
    - Verify files without dependencies are removed
    - Verify dependency report accuracy
    - _Requirements: 1.5, 2.5, 3.4, 3.5, 4.5_

  - [ ]* 14.3 Write backup and restore test
    - Create backup archive of test files
    - Remove test files
    - Restore from backup archive
    - Verify all files restored correctly with original structure
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [ ]* 14.4 Write test suite verification test
    - Run cleanup on test project
    - Verify pytest execution
    - Verify test results captured correctly
    - Verify main.py import verification
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 15. Add documentation and usage instructions
  - [ ] 15.1 Create README for cleanup system
    - Document cleanup system purpose and workflow
    - Document command-line usage with examples
    - Document dry-run mode usage
    - Document backup restoration procedure
    - Document configuration file format
    - Document safety mechanisms and error handling
    - Add troubleshooting section
    - _Requirements: 6.5, 7.2, 8.4, 9.1_

  - [ ] 15.2 Add inline code documentation
    - Add docstrings to all classes and methods
    - Add type hints to all function signatures
    - Add comments for complex logic sections
    - Document error handling strategies
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1_

- [ ] 16. Final checkpoint and validation
  - Run full test suite to ensure all tests pass
  - Run cleanup system in dry-run mode on actual project
  - Review generated report for accuracy
  - Verify backup archive creation and integrity
  - Verify no active files marked for removal
  - Ask the user if questions arise before proceeding with actual cleanup

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- The system uses Python with pathlib, tarfile, ast, subprocess, datetime, and shutil
- Backup creation is CRITICAL - if backup fails, entire cleanup must abort
- Removal phase continues even if individual files fail, collecting all errors
- Post-cleanup verification runs pytest to ensure system integrity
- Dry-run mode is recommended for initial testing before actual cleanup
