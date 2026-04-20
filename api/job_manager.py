import glob
import os
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Optional

from api.models import GenerateRequest, JobFiles, JobState, JobStatus


class JobManager:
    """Single-GPU sequential job queue with in-memory state."""

    def __init__(self, output_root: str = "outputs/jobs"):
        self.output_root = output_root
        self._jobs: OrderedDict[str, JobStatus] = OrderedDict()
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._event = threading.Event()
        self._worker.start()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def submit(self, request: GenerateRequest) -> JobStatus:
        job_id = str(uuid.uuid4())
        name = request.name or request.prompt.split()[0].lower().replace(",", "")
        job = JobStatus(
            job_id=job_id,
            status=JobState.queued,
            prompt=request.prompt,
            name=name,
            model=request.model,
            created_at=datetime.utcnow(),
        )
        with self._lock:
            self._jobs[job_id] = job
            self._queue.append(job_id)
            job.queue_position = len(self._queue)
        self._event.set()
        return job

    def get(self, job_id: str) -> Optional[JobStatus]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status == JobState.queued:
                try:
                    job.queue_position = self._queue.index(job_id) + 1
                except ValueError:
                    pass
            return job

    def list_all(self) -> list[JobStatus]:
        with self._lock:
            return list(reversed(list(self._jobs.values())))

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _process_loop(self):
        while True:
            self._event.wait()
            self._event.clear()
            while True:
                with self._lock:
                    if not self._queue:
                        break
                    job_id = self._queue[0]

                self._run_job(job_id)

                with self._lock:
                    if self._queue and self._queue[0] == job_id:
                        self._queue.pop(0)

    def _run_job(self, job_id: str):
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobState.processing
            job.started_at = datetime.utcnow()
            job.queue_position = None
            request = GenerateRequest(
                prompt=job.prompt,
                name=job.name,
                model=job.model,
            )

        try:
            files = self._execute(job_id, request)
            with self._lock:
                job.status = JobState.completed
                job.finished_at = datetime.utcnow()
                job.files = files
        except Exception as exc:
            with self._lock:
                job.status = JobState.failed
                job.finished_at = datetime.utcnow()
                job.error = str(exc)

    def _execute(self, job_id: str, request: GenerateRequest) -> JobFiles:
        # Import here so models are loaded once at server startup, not import time
        import os
        os.environ["TEXT_MODEL"] = request.model.value

        from embodied_gen.data.asset_converter import cvt_embodiedgen_asset_to_anysim
        from embodied_gen.scripts.textto3d import text_to_3d
        from embodied_gen.utils.enum import AssetType

        output_root = os.path.join(self.output_root, job_id)
        os.makedirs(output_root, exist_ok=True)

        results = text_to_3d(
            prompts=[request.prompt],
            asset_names=[request.name],
            output_root=output_root,
            seed_img=request.seed_img,
            seed_3d=request.seed_3d,
            n_image_retry=request.n_image_retry,
            n_asset_retry=request.n_asset_retry,
            n_pipe_retry=1,
        )

        asset_rel = results.get("assets", {}).get(request.name)
        if not asset_rel:
            raise RuntimeError("Pipeline produced no output asset")

        asset_dir = os.path.join(output_root, asset_rel)

        # Find URDF
        urdf_matches = glob.glob(os.path.join(asset_dir, "**", "*.urdf"), recursive=True)
        if not urdf_matches:
            raise RuntimeError(f"URDF not found in {asset_dir}")
        urdf_path = next(
            (p for p in urdf_matches if os.path.basename(p) == f"{request.name}.urdf"),
            urdf_matches[0],
        )

        # Find OBJ
        obj_matches = glob.glob(os.path.join(asset_dir, "mesh", "*.obj"))
        obj_path = obj_matches[0] if obj_matches else None

        # Convert URDF → MJCF
        mjcf_dir = os.path.join(os.path.dirname(urdf_path), "mjcf")
        asset_paths = cvt_embodiedgen_asset_to_anysim(
            urdf_files=[urdf_path],
            target_dirs=[mjcf_dir],
            target_type=AssetType.MJCF,
            source_type=AssetType.MESH,
        )
        mjcf_path = asset_paths.get(urdf_path)

        return JobFiles(
            obj=obj_path,
            urdf=urdf_path,
            mjcf=mjcf_path,
        )


job_manager = JobManager(
    output_root=os.environ.get("OUTPUT_ROOT", "outputs/jobs")
)
