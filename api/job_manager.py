import glob as glob_module
import io
import json
import logging
import os
import sys
import threading
import traceback
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Optional

from api.models import (
    GenerateRequest,
    ItemFiles,
    JobState,
    JobStatus,
)

MAX_LOG_LINES = 2000


class _JobLogHandler(logging.Handler):
    def __init__(self, append_fn):
        super().__init__()
        self._append = append_fn

    def emit(self, record):
        try:
            self._append(self.format(record))
        except Exception:
            pass


class _TeeStream(io.TextIOBase):
    def __init__(self, original, append_fn):
        self._original = original
        self._append = append_fn
        self._buf = ""

    def write(self, s):
        self._original.write(s)
        self._original.flush()
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._append(line)
        return len(s)

    def flush(self):
        self._original.flush()


class JobManager:
    """Single-GPU sequential job queue with disk-persisted state."""

    def __init__(self, output_root: str = "outputs/jobs"):
        self.output_root = output_root
        self._jobs: OrderedDict[str, JobStatus] = OrderedDict()
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._worker = threading.Thread(
            target=self._process_loop, daemon=True
        )
        self._event = threading.Event()
        self._load_jobs()
        self._worker.start()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _job_path(self, job_id: str) -> str:
        return os.path.join(self.output_root, job_id, "job.json")

    def _save_job(self, job: JobStatus):
        path = self._job_path(job.job_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(job.model_dump_json(indent=2))

    def _load_jobs(self):
        os.makedirs(self.output_root, exist_ok=True)
        paths = sorted(
            glob_module.glob(
                os.path.join(self.output_root, "*", "job.json")
            ),
            key=os.path.getmtime,
        )
        for path in paths:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                job = JobStatus(**data)
                if job.status == JobState.processing:
                    job.status = JobState.failed
                    job.error = (
                        (job.error or "") + " [server restarted]"
                    )
                    job.finished_at = (
                        job.finished_at or datetime.utcnow()
                    )
                    self._save_job(job)
                self._jobs[job.job_id] = job
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def submit(self, request: GenerateRequest) -> JobStatus:
        job_id = str(uuid.uuid4())
        job = JobStatus(
            job_id=job_id,
            status=JobState.queued,
            item_count=len(request.items),
            model=request.model,
            created_at=datetime.utcnow(),
        )
        with self._lock:
            self._jobs[job_id] = job
            self._queue.append(job_id)
            job.queue_position = len(self._queue)
        self._save_job(job)
        # Store request for later execution
        req_path = os.path.join(
            self.output_root, job_id, "request.json"
        )
        os.makedirs(os.path.dirname(req_path), exist_ok=True)
        with open(req_path, "w") as f:
            f.write(request.model_dump_json(indent=2))
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

    def _append_log(self, job_id: str, line: str):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.logs.append(line)
                if len(job.logs) > MAX_LOG_LINES:
                    job.logs = job.logs[-MAX_LOG_LINES:]

    def _run_job(self, job_id: str):
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobState.processing
            job.started_at = datetime.utcnow()
            job.queue_position = None
        self._save_job(job)

        def append(line: str):
            self._append_log(job_id, line)

        tee_out = _TeeStream(sys.stdout, append)
        tee_err = _TeeStream(sys.stderr, append)
        log_handler = _JobLogHandler(append)
        log_handler.setFormatter(
            logging.Formatter("%(name)s: %(message)s")
        )
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = tee_out, tee_err

        try:
            results = self._execute(job_id)
            with self._lock:
                job.status = JobState.completed
                job.finished_at = datetime.utcnow()
                job.results = results
            self._save_job(job)
        except Exception as exc:
            err_msg = (
                f"{type(exc).__name__}: {exc}"
                if str(exc)
                else type(exc).__name__
            )
            append(f"ERROR: {err_msg}")
            append("Traceback (most recent call last):")
            for line in traceback.format_exc().strip().splitlines()[1:]:
                append(line)
            with self._lock:
                job.status = JobState.failed
                job.finished_at = datetime.utcnow()
                job.error = err_msg
            self._save_job(job)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            root_logger.removeHandler(log_handler)

    def _execute(self, job_id: str) -> dict[str, ItemFiles]:
        from asset_gen.data.asset_converter import (
            cvt_asset_gen_asset_to_anysim,
        )
        from asset_gen.scripts.textto3d import (
            GenerateItem,
            text_to_3d,
        )
        from asset_gen.utils.enum import AssetType

        req_path = os.path.join(
            self.output_root, job_id, "request.json"
        )
        with open(req_path) as f:
            req_data = json.load(f)

        os.environ["TEXT_MODEL"] = req_data.get("model", "sd15")

        n_image_retry = int(req_data.get("n_image_retry", 2))
        n_asset_retry = int(req_data.get("n_asset_retry", 2))
        n_pipe_retry = int(req_data.get("n_pipe_retry", 1))
        img_denoise_step = int(req_data.get("img_denoise_step", 25))
        text_guidance_scale = float(req_data.get("text_guidance_scale", 7.0))
        n_img_sample = int(req_data.get("n_img_sample", 1))
        image_height = int(req_data.get("image_height", 768))
        image_width = int(req_data.get("image_width", 768))

        logging.getLogger(__name__).info(
            (
                "request settings: model=%s image=%sx%s steps=%s "
                "retries=%s/%s/%s samples=%s guidance=%.2f"
            ),
            os.environ["TEXT_MODEL"],
            image_width,
            image_height,
            img_denoise_step,
            n_image_retry,
            n_asset_retry,
            n_pipe_retry,
            n_img_sample,
            text_guidance_scale,
        )

        output_root = os.path.join(self.output_root, job_id)

        # Build GenerateItem list from stored request
        gen_items: list[GenerateItem] = []
        for raw in req_data.get("items", []):
            name = raw.get("name") or (
                raw.get("prompt", "object").split()[0]
                .lower()
                .replace(",", "")
            )
            gen_items.append(
                GenerateItem(
                    name=name,
                    prompt=raw.get("prompt"),
                    image_b64=raw.get("image_b64"),
                    asset_type=raw.get("asset_type"),
                    seed_img=raw.get("seed_img"),
                    seed_3d=raw.get("seed_3d", 0),
                )
            )

        enable_texture = bool(req_data.get("enable_texture", False))
        skip_qa = bool(req_data.get("skip_qa", False))
        disable_decompose_convex = bool(
            req_data.get("disable_decompose_convex", False)
        )

        batch_results = text_to_3d(
            items=gen_items,
            output_root=output_root,
            n_image_retry=n_image_retry,
            n_asset_retry=n_asset_retry,
            n_pipe_retry=n_pipe_retry,
            img_denoise_step=img_denoise_step,
            text_guidance_scale=text_guidance_scale,
            n_img_sample=n_img_sample,
            image_height=image_height,
            image_width=image_width,
            enable_texture=enable_texture,
            skip_qa=skip_qa,
            disable_decompose_convex=disable_decompose_convex,
        )

        # Convert URDF → MJCF and build ItemFiles for each object
        item_files: dict[str, ItemFiles] = {}
        for item in gen_items:
            file_paths = batch_results.get("files", {}).get(item.name)
            if not file_paths:
                item_files[item.name] = ItemFiles()
                continue

            urdf_path = file_paths.get("urdf")
            obj_path = file_paths.get("obj")
            glb_path = file_paths.get("glb")
            mjcf_path = None

            if urdf_path and os.path.exists(urdf_path):
                mjcf_dir = os.path.join(
                    os.path.dirname(urdf_path), "mjcf"
                )
                asset_paths = cvt_asset_gen_asset_to_anysim(
                    urdf_files=[urdf_path],
                    target_dirs=[mjcf_dir],
                    target_type=AssetType.MJCF,
                    source_type=AssetType.MESH,
                )
                mjcf_path = asset_paths.get(urdf_path)

            item_files[item.name] = ItemFiles(
                obj=obj_path,
                glb=glb_path,
                urdf=urdf_path,
                mjcf=mjcf_path,
            )

        return item_files


job_manager = JobManager(
    output_root=os.environ.get("OUTPUT_ROOT", "outputs/jobs")
)
