"""Pipeline tracker for monitoring objects through semantic enforcement stages.

This module provides tracking and tracing capabilities for the semantic plan
enforcement pipeline. It logs object data at each stage, detects changes between
stages, and generates comprehensive trace files for debugging and validation.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StageLog:
    """Log entry for a single pipeline stage.
    
    Attributes:
        name: Stage name (e.g., "semantic_plan", "distance_resolver")
        timestamp: ISO 8601 timestamp when stage was logged
        object_count: Number of objects at this stage
        objects: List of object data dictionaries
    """
    name: str
    timestamp: str
    object_count: int
    objects: List[Dict[str, Any]]


@dataclass
class StageChange:
    """Changes detected between two pipeline stages.
    
    Attributes:
        from_stage: Source stage name
        to_stage: Target stage name
        added: List of object IDs that were added
        removed: List of object IDs that were removed
        modified: List of modifications with details
    """
    from_stage: str
    to_stage: str
    added: List[str]
    removed: List[str]
    modified: List[Dict[str, Any]]


class PipelineTracker:
    """Tracks objects through all pipeline stages.
    
    This class logs object data at each stage of the semantic enforcement
    pipeline and provides methods to generate traces and detect changes.
    
    Stages tracked:
    - semantic_plan: Initial plan from LLM
    - schema_validation: After schema validation
    - distance_resolver: After resolving relative positions
    - orientation_resolver: After resolving relative orientations
    - placement_executor: After placement execution
    - scene_assembly: After MuJoCo XML generation
    
    Example:
        tracker = PipelineTracker(output_dir="logs/traces")
        tracker.log_stage("semantic_plan", semantic_plan.objects)
        tracker.log_stage("distance_resolver", resolved_objects)
        trace = tracker.generate_trace()
        tracker.save_trace(trace)
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize pipeline tracker.
        
        Args:
            output_dir: Directory to save trace files. If None, uses "analysis_logs".
        """
        self.output_dir = Path(output_dir) if output_dir else Path("analysis_logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stages: List[StageLog] = []
        self.pipeline_start_time = datetime.utcnow().isoformat()
    
    def log_stage(
        self,
        stage_name: str,
        objects: List[Any],
    ) -> None:
        """Log object data at a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            objects: List of objects (can be SemanticObject, PlacedObject, or dicts)
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Convert objects to dictionaries for serialization
        object_dicts = []
        for obj in objects:
            if hasattr(obj, '__dict__'):
                # Handle dataclass or regular class instances
                obj_dict = asdict(obj) if hasattr(obj, '__dataclass_fields__') else vars(obj)
            elif isinstance(obj, dict):
                obj_dict = obj
            else:
                # Fallback: try to convert to dict
                obj_dict = {"data": str(obj)}
            
            object_dicts.append(obj_dict)
        
        stage_log = StageLog(
            name=stage_name,
            timestamp=timestamp,
            object_count=len(objects),
            objects=object_dicts,
        )
        
        self.stages.append(stage_log)
        
        # Log summary
        print(f"[PipelineTracker] Stage '{stage_name}': {len(objects)} objects logged")
    
    def generate_trace(self) -> Dict[str, Any]:
        """Generate complete pipeline trace.
        
        Returns:
            Dictionary containing:
            - pipeline_start_time: ISO 8601 timestamp
            - total_stages: Number of stages logged
            - stages: List of stage logs with object data
            - changes: List of detected changes between consecutive stages
        """
        # Generate changes between consecutive stages
        changes = []
        for i in range(len(self.stages) - 1):
            change = self.detect_changes(
                self.stages[i].name,
                self.stages[i + 1].name,
            )
            if change:
                changes.append(asdict(change))
        
        trace = {
            "pipeline_trace": {
                "pipeline_start_time": self.pipeline_start_time,
                "total_stages": len(self.stages),
                "stages": [asdict(stage) for stage in self.stages],
                "changes": changes,
            }
        }
        
        return trace
    
    def detect_changes(
        self,
        stage_a: str,
        stage_b: str,
    ) -> Optional[StageChange]:
        """Detect changes between two stages.
        
        Args:
            stage_a: Source stage name
            stage_b: Target stage name
        
        Returns:
            StageChange object with added, removed, and modified objects,
            or None if either stage is not found.
        """
        # Find stage logs
        stage_a_log = None
        stage_b_log = None
        
        for stage in self.stages:
            if stage.name == stage_a:
                stage_a_log = stage
            if stage.name == stage_b:
                stage_b_log = stage
        
        if not stage_a_log or not stage_b_log:
            return None
        
        # Build object ID maps
        objects_a = {obj.get("id"): obj for obj in stage_a_log.objects if "id" in obj}
        objects_b = {obj.get("id"): obj for obj in stage_b_log.objects if "id" in obj}
        
        ids_a = set(objects_a.keys())
        ids_b = set(objects_b.keys())
        
        # Detect added and removed objects
        added = list(ids_b - ids_a)
        removed = list(ids_a - ids_b)
        
        # Detect modified objects
        modified = []
        for obj_id in ids_a & ids_b:
            obj_a = objects_a[obj_id]
            obj_b = objects_b[obj_id]
            
            # Check for modifications in key fields
            modifications = self._compare_objects(obj_id, obj_a, obj_b)
            if modifications:
                modified.extend(modifications)
        
        return StageChange(
            from_stage=stage_a,
            to_stage=stage_b,
            added=added,
            removed=removed,
            modified=modified,
        )
    
    def _compare_objects(
        self,
        obj_id: str,
        obj_a: Dict[str, Any],
        obj_b: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Compare two object versions and detect modifications.
        
        Args:
            obj_id: Object ID
            obj_a: Object data from stage A
            obj_b: Object data from stage B
        
        Returns:
            List of modification dictionaries with field, old, and new values.
        """
        modifications = []
        
        # Fields to track for changes
        tracked_fields = ["position", "orientation", "size", "Model", "model_loc"]
        
        for field_name in tracked_fields:
            if field_name in obj_a and field_name in obj_b:
                if obj_a[field_name] != obj_b[field_name]:
                    modifications.append({
                        "id": obj_id,
                        "field": field_name,
                        "old": obj_a[field_name],
                        "new": obj_b[field_name],
                    })
        
        return modifications
    
    def save_trace(
        self,
        trace: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """Save pipeline trace to JSON file.
        
        Args:
            trace: Trace dictionary. If None, generates trace automatically.
            filename: Output filename. If None, generates timestamped filename.
        
        Returns:
            Path to saved trace file.
        """
        if trace is None:
            trace = self.generate_trace()
        
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_trace_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
        
        print(f"[PipelineTracker] Trace saved to: {output_path}")
        return output_path
    
    def get_stage_summary(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get summary information for a specific stage.
        
        Args:
            stage_name: Name of the stage
        
        Returns:
            Dictionary with stage summary or None if stage not found.
        """
        for stage in self.stages:
            if stage.name == stage_name:
                object_ids = [obj.get("id") for obj in stage.objects if "id" in obj]
                return {
                    "name": stage.name,
                    "timestamp": stage.timestamp,
                    "object_count": stage.object_count,
                    "object_ids": object_ids,
                }
        return None
    
    def log_warning(
        self,
        stage_name: str,
        message: str,
        obj_id: Optional[str] = None,
    ) -> None:
        """Log a warning for unexpected changes or issues.
        
        Args:
            stage_name: Stage where warning occurred
            message: Warning message
            obj_id: Optional object ID related to warning
        """
        timestamp = datetime.utcnow().isoformat()
        warning_msg = f"[WARNING] {timestamp} | Stage: {stage_name}"
        if obj_id:
            warning_msg += f" | Object: {obj_id}"
        warning_msg += f" | {message}"
        print(warning_msg)
