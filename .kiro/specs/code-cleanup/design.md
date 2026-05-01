# Design Document: Legacy Code Cleanup System

## Overview

The Legacy Code Cleanup System is an automated tool for safely removing outdated files, temporary scripts, and obsolete directories from the world-creator project while preserving active code and creating recovery backups. The system follows a classify-verify-backup-remove workflow to ensure safe cleanup operations.

### Design Goals

1. **Safety First**: Never remove files that are actively used by the codebase
2. **Recoverability**: Create comprehensive backups before any deletion
3. **Transparency**: Provide detailed reporting of all cleanup operations
4. **Verification**: Validate system integrity after cleanup
5. **Configurability**: Allow easy modification of cleanup rules

### Key Design Decisions

- **Static Classification**: Use predefined file lists rather than heuristic detection to avoid false positives
- **Dependency Analysis**: Parse Python imports to verify no active code depends on files marked for removal
- **Atomic Backup**: Create complete backup archive before any file removal begins
- **Fail-Safe Operation**: Continue cleanup even if individual file removals fail, logging all errors
- **Post-Cleanup Validation**: Run official test suite to verify system integrity

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Cleanup Orchestrator                      │
│  (Coordinates overall cleanup workflow)                      │
└────────────┬────────────────────────────────────────────────┘
             │
             ├──────────────┬──────────────┬──────────────┐
             │              │              │              │
             ▼              ▼              ▼              ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
    │   File     │  │ Dependency │  │   Backup   │  │  Removal   │
    │ Classifier │  │  Analyzer  │  │  Manager   │  │  Engine    │
    └────────────┘  └────────────┘  └────────────┘  └────────────┘
             │              │              │              │
             └──────────────┴──────────────┴──────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │    Reporter    │
                   │  (Generates    │
                   │   reports)     │
                   └────────────────┘
```

### Component Responsibilities

1. **Cleanup Orchestrator**: Main entry point that coordinates the cleanup workflow
2. **File Classifier**: Identifies and categorizes files based on predefined rules
3. **Dependency Analyzer**: Verifies that files marked for removal are not imported by active code
4. **Backup Manager**: Creates compressed archives of files before removal
5. **Removal Engine**: Safely removes files and empty directories
6. **Reporter**: Generates detailed cleanup reports

### Workflow Sequence

```
1. Initialize Cleanup
   ↓
2. Classify Files (File Classifier)
   ↓
3. Analyze Dependencies (Dependency Analyzer)
   ↓
4. Create Backup Archive (Backup Manager)
   ↓
5. Remove Files (Removal Engine)
   ↓
6. Verify Integrity (Run pytest)
   ↓
7. Generate Report (Reporter)
```

## Components and Interfaces

### 1. File Classifier

**Purpose**: Categorize files into legacy and active based on predefined rules.

**Interface**:
```python
class FileClassifier:
    def classify_files(self, root_dir: Path) -> ClassificationResult:
        """Classify all files in the project."""
        pass
    
    def is_legacy_documentation(self, file_path: Path) -> bool:
        """Check if file is legacy documentation."""
        pass
    
    def is_temporary_test_script(self, file_path: Path) -> bool:
        """Check if file is a temporary test script."""
        pass
    
    def is_legacy_module(self, file_path: Path) -> bool:
        """Check if file is a legacy code module."""
        pass
    
    def is_legacy_directory(self, dir_path: Path) -> bool:
        """Check if directory is obsolete."""
        pass
    
    def is_temporary_data(self, dir_path: Path) -> bool:
        """Check if directory contains temporary data."""
        pass
```

**Classification Rules**:
- Legacy documentation: Hardcoded list from requirements
- Temporary test scripts: Root directory .py files not in active code list
- Legacy modules: Duplicate implementations (e.g., runner_universal.py)
- Legacy directories: project/, docs/, examples/, .claude/
- Temporary data: .cache_ciare/, old saved_scenes/ subdirectories

### 2. Dependency Analyzer

**Purpose**: Verify that files marked for removal are not imported by active code.

**Interface**:
```python
class DependencyAnalyzer:
    def analyze_dependencies(
        self, 
        files_to_remove: List[Path],
        active_code_dirs: List[Path]
    ) -> DependencyReport:
        """Analyze if any active code depends on files to be removed."""
        pass
    
    def find_imports(self, python_file: Path) -> Set[str]:
        """Extract all import statements from a Python file."""
        pass
    
    def resolve_import_path(self, import_name: str, project_root: Path) -> Optional[Path]:
        """Resolve an import name to a file path."""
        pass
    
    def has_dependencies(self, file_path: Path, active_imports: Set[str]) -> bool:
        """Check if a file is imported by active code."""
        pass
```

**Analysis Strategy**:
- Parse all Python files in active code directories (creator/, tests/, main.py)
- Extract import statements using AST parsing
- Build dependency graph
- Flag any files marked for removal that appear in the dependency graph

### 3. Backup Manager

**Purpose**: Create compressed archives of files before removal.

**Interface**:
```python
class BackupManager:
    def create_backup(
        self, 
        files_to_backup: List[Path],
        backup_dir: Path
    ) -> BackupResult:
        """Create a tar.gz archive of files to be removed."""
        pass
    
    def generate_backup_name(self) -> str:
        """Generate timestamped backup filename."""
        pass
    
    def verify_backup(self, archive_path: Path) -> bool:
        """Verify backup archive integrity."""
        pass
```

**Backup Format**:
- Archive name: `legacy_backup_YYYYMMDD_HHMMSS.tar.gz`
- Preserve directory structure within archive
- Store in project root directory
- Include metadata file listing all archived files

### 4. Removal Engine

**Purpose**: Safely remove files and directories from the filesystem.

**Interface**:
```python
class RemovalEngine:
    def remove_files(
        self, 
        files_to_remove: List[Path],
        dry_run: bool = False
    ) -> RemovalResult:
        """Remove files from filesystem."""
        pass
    
    def remove_empty_directories(self, root_dir: Path) -> List[Path]:
        """Remove empty directories after file removal."""
        pass
    
    def clean_old_scenes(self, saved_scenes_dir: Path, days_old: int = 30) -> List[Path]:
        """Remove scene subdirectories older than specified days."""
        pass
```

**Safety Mechanisms**:
- Verify backup exists before any removal
- Log each file removal operation
- Continue on individual failures, collect errors
- Never remove directories with remaining files
- Preserve directory structure for active directories

### 5. Reporter

**Purpose**: Generate detailed reports of cleanup operations.

**Interface**:
```python
class Reporter:
    def generate_cleanup_report(
        self,
        classification: ClassificationResult,
        backup_result: BackupResult,
        removal_result: RemovalResult,
        test_result: TestResult
    ) -> str:
        """Generate comprehensive cleanup report in Markdown format."""
        pass
    
    def calculate_space_freed(self, removed_files: List[Path]) -> int:
        """Calculate total bytes freed by cleanup."""
        pass
    
    def format_file_list(self, files: List[Path], include_sizes: bool = True) -> str:
        """Format file list for report."""
        pass
```

**Report Contents**:
- Summary statistics (files removed, space freed)
- Categorized lists of removed files with sizes
- Backup archive location and size
- Test suite execution results
- Any errors or warnings encountered

### 6. Cleanup Orchestrator

**Purpose**: Coordinate the entire cleanup workflow.

**Interface**:
```python
class CleanupOrchestrator:
    def __init__(
        self,
        project_root: Path,
        dry_run: bool = False
    ):
        self.classifier = FileClassifier()
        self.analyzer = DependencyAnalyzer()
        self.backup_manager = BackupManager()
        self.removal_engine = RemovalEngine()
        self.reporter = Reporter()
    
    def execute_cleanup(self) -> CleanupResult:
        """Execute the complete cleanup workflow."""
        pass
    
    def verify_integrity(self) -> TestResult:
        """Run pytest to verify system integrity."""
        pass
```

## Data Models

### ClassificationResult

```python
@dataclass
class ClassificationResult:
    legacy_documentation: List[Path]
    temporary_test_scripts: List[Path]
    legacy_modules: List[Path]
    legacy_directories: List[Path]
    temporary_data_dirs: List[Path]
    preserved_files: List[Path]
    
    def all_files_to_remove(self) -> List[Path]:
        """Get all files marked for removal."""
        pass
    
    def total_count(self) -> int:
        """Get total number of files to remove."""
        pass
```

### DependencyReport

```python
@dataclass
class DependencyReport:
    analyzed_files: List[Path]
    dependencies_found: Dict[Path, List[Path]]  # file -> files that import it
    safe_to_remove: List[Path]
    blocked_removals: List[Path]
    
    def has_blockers(self) -> bool:
        """Check if any files are blocked from removal."""
        pass
```

### BackupResult

```python
@dataclass
class BackupResult:
    archive_path: Path
    archive_size_bytes: int
    files_backed_up: List[Path]
    success: bool
    error_message: Optional[str] = None
    
    def archive_size_mb(self) -> float:
        """Get archive size in megabytes."""
        pass
```

### RemovalResult

```python
@dataclass
class RemovalResult:
    removed_files: List[Path]
    removed_directories: List[Path]
    failed_removals: Dict[Path, str]  # file -> error message
    space_freed_bytes: int
    
    def success_count(self) -> int:
        """Get number of successfully removed files."""
        pass
    
    def failure_count(self) -> int:
        """Get number of failed removals."""
        pass
    
    def space_freed_mb(self) -> float:
        """Get space freed in megabytes."""
        pass
```

### TestResult

```python
@dataclass
class TestResult:
    passed: bool
    output: str
    failed_tests: List[str]
    execution_time_seconds: float
    
    def summary(self) -> str:
        """Get test result summary."""
        pass
```

### CleanupResult

```python
@dataclass
class CleanupResult:
    classification: ClassificationResult
    dependency_report: DependencyReport
    backup_result: BackupResult
    removal_result: RemovalResult
    test_result: TestResult
    report_path: Path
    overall_success: bool
```

## Error Handling

### Error Categories

1. **Classification Errors**: File not found, permission denied during scanning
2. **Dependency Analysis Errors**: Invalid Python syntax, import resolution failures
3. **Backup Errors**: Insufficient disk space, archive creation failure
4. **Removal Errors**: Permission denied, file in use, directory not empty
5. **Verification Errors**: Test suite failures, import errors

### Error Handling Strategy

**Classification Phase**:
- Log missing files but continue classification
- Skip files with permission errors
- Report all errors in final report

**Dependency Analysis Phase**:
- Skip files with syntax errors (log warning)
- Continue analysis even if some imports can't be resolved
- Conservative approach: if uncertain, mark as "not safe to remove"

**Backup Phase**:
- **CRITICAL**: If backup fails, abort entire cleanup operation
- Check available disk space before creating archive
- Verify archive integrity after creation
- Provide clear error message if backup fails

**Removal Phase**:
- Continue removing files even if individual removals fail
- Collect all errors for reporting
- Never remove directories that still contain files
- Log each successful and failed removal

**Verification Phase**:
- Run pytest even if some removals failed
- Capture full test output
- If tests fail, recommend restoring from backup
- Verify main.py can be imported

### Error Recovery

```python
class CleanupError(Exception):
    """Base exception for cleanup operations."""
    pass

class BackupFailedError(CleanupError):
    """Raised when backup creation fails - abort cleanup."""
    pass

class RemovalError(CleanupError):
    """Raised when file removal fails - continue with others."""
    pass

class VerificationError(CleanupError):
    """Raised when post-cleanup verification fails."""
    pass
```

**Recovery Procedures**:
1. **Backup Failure**: Abort cleanup, report error, no files removed
2. **Partial Removal Failure**: Continue cleanup, report failures in final report
3. **Test Failure**: Report failure, provide backup restoration instructions
4. **Import Error**: Report error, recommend backup restoration

## Testing Strategy

### Unit Tests

**File Classifier Tests**:
- Test classification of each file category
- Test preservation of active files
- Test handling of missing files
- Test edge cases (empty directories, symlinks)

**Dependency Analyzer Tests**:
- Test import extraction from Python files
- Test import path resolution
- Test dependency graph construction
- Test detection of circular dependencies
- Test handling of invalid Python syntax

**Backup Manager Tests**:
- Test backup archive creation
- Test backup name generation
- Test archive integrity verification
- Test handling of large files
- Test handling of special characters in filenames

**Removal Engine Tests**:
- Test file removal (use temporary test directories)
- Test empty directory removal
- Test old scene cleanup (date-based filtering)
- Test error handling for permission denied
- Test preservation of non-empty directories

**Reporter Tests**:
- Test report generation with various inputs
- Test space calculation
- Test file list formatting
- Test Markdown formatting

### Integration Tests

**End-to-End Cleanup Test**:
- Create mock project structure with legacy and active files
- Run complete cleanup workflow
- Verify correct files removed
- Verify backup created
- Verify active files preserved
- Verify report generated

**Dependency Analysis Integration Test**:
- Create test project with import dependencies
- Verify files with dependencies are not removed
- Verify files without dependencies are removed

**Backup and Restore Test**:
- Create backup archive
- Remove files
- Restore from backup
- Verify all files restored correctly

**Test Suite Verification Test**:
- Run cleanup on test project
- Verify pytest execution
- Verify test results captured correctly

### Test Data

Create fixture directories for testing:
```
test_fixtures/
├── mock_project/
│   ├── active_code/
│   │   ├── main.py
│   │   └── module.py
│   ├── legacy_docs/
│   │   └── old_readme.md
│   ├── temp_scripts/
│   │   └── test_temp.py
│   └── tests/
│       └── test_active.py
```

### Testing Approach

- **Unit tests**: Test each component in isolation with mocks
- **Integration tests**: Test component interactions with temporary file systems
- **No property-based testing**: This feature involves file system side effects, external dependencies (pytest), and deterministic configuration-based behavior, making it unsuitable for property-based testing
- **Manual testing**: Run on actual project with dry-run mode first

### Test Coverage Goals

- Unit test coverage: >90% for all components
- Integration test coverage: All major workflows
- Error path coverage: All error handling branches
- Edge case coverage: Empty directories, missing files, permission errors

## Implementation Notes

### Dependencies

- **pathlib**: File path manipulation
- **tarfile**: Backup archive creation
- **ast**: Python import parsing
- **subprocess**: Running pytest
- **datetime**: Timestamp generation
- **shutil**: File operations

### Configuration

Store classification rules in a configuration file for easy modification:

```python
# cleanup_config.py
LEGACY_DOCUMENTATION = [
    "CHANGES_is_static.md",
    "CHEATSHEET.md",
    "CLAUDE.md",
    # ... full list from requirements
]

TEMPORARY_TEST_SCRIPTS = [
    "example_usage.py",
    "main_universal.py",
    # ... full list from requirements
]

LEGACY_MODULES = [
    "creator/runner_universal.py",
]

LEGACY_DIRECTORIES = [
    "project/",
    "docs/",
    "examples/",
    ".claude/",
]

TEMPORARY_DATA_DIRS = [
    ".cache_ciare/",
]

PRESERVED_DOCUMENTATION = [
    "README_RUN.md",
    "QUICKSTART.md",
    "USAGE_GUIDE.md",
    "RUN_TEST.md",
]

ACTIVE_CODE_DIRS = [
    "creator/",
    "tests/",
]

ACTIVE_CODE_FILES = [
    "main.py",
    "save_scene_for_analysis.py",
]

OLD_SCENE_DAYS = 30
```

### Dry Run Mode

Implement a dry-run mode that:
- Performs all analysis steps
- Creates backup archive
- **Does not** remove any files
- Generates report showing what would be removed
- Useful for testing and verification before actual cleanup

### Logging

Use Python logging module with multiple levels:
- **DEBUG**: Detailed operation logs
- **INFO**: Major workflow steps
- **WARNING**: Non-critical issues (skipped files, unresolved imports)
- **ERROR**: Critical failures (backup failure, removal errors)

### Performance Considerations

- **Lazy evaluation**: Only analyze files that need to be checked
- **Parallel processing**: Consider using multiprocessing for large projects
- **Incremental backup**: For very large cleanups, consider incremental backups
- **Memory efficiency**: Stream large files during backup rather than loading into memory

### Future Enhancements

1. **Interactive mode**: Prompt user to confirm each category of removals
2. **Selective cleanup**: Allow user to choose which categories to clean
3. **Scheduled cleanup**: Integrate with cron for periodic cleanup
4. **Cleanup history**: Track previous cleanups and allow rollback
5. **Smart detection**: Use heuristics to detect legacy files automatically
6. **Git integration**: Use git history to identify unused files

