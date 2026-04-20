import io
import os
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from api.job_manager import job_manager
from api.models import GenerateRequest, GenerateResponse, JobState, JobStatus

app = FastAPI(
    title="EmbodiedGen API",
    description="Text prompt → 3D model (OBJ + URDF + MuJoCo MJCF)",
    version="1.0.0",
)


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}


# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────

@app.post("/api/generate", response_model=GenerateResponse, tags=["Generation"])
def generate(request: GenerateRequest):
    """
    Submit a text prompt for 3D object generation.

    Returns a `job_id` to track progress. Generation takes 3–10 minutes.
    """
    if not request.name:
        request.name = request.prompt.split()[0].lower().replace(",", "").replace(".", "")

    job = job_manager.submit(request)
    return GenerateResponse(
        job_id=job.job_id,
        status=job.status,
        queue_position=job.queue_position or 1,
    )


# ──────────────────────────────────────────────
# Jobs
# ──────────────────────────────────────────────

@app.get("/api/jobs", response_model=list[JobStatus], tags=["Jobs"])
def list_jobs():
    """List all jobs (newest first)."""
    return job_manager.list_all()


@app.get("/api/jobs/{job_id}", response_model=JobStatus, tags=["Jobs"])
def get_job(job_id: str):
    """Get status and result paths for a specific job."""
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/logs", tags=["Jobs"])
def get_job_logs(job_id: str, last: int = 50):
    """Return last N log lines for a running or completed job."""
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": job.status, "logs": job.logs[-last:]}


# ──────────────────────────────────────────────
# Downloads
# ──────────────────────────────────────────────

def _require_completed(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != JobState.completed:
        raise HTTPException(400, f"Job is {job.status}, not completed")
    if not job.files:
        raise HTTPException(500, "Job completed but no files recorded")
    return job


@app.get("/api/jobs/{job_id}/download/obj", tags=["Download"])
def download_obj(job_id: str):
    """Download the generated OBJ mesh."""
    job = _require_completed(job_id)
    path = job.files.obj
    if not path or not Path(path).exists():
        raise HTTPException(404, "OBJ file not found")
    return FileResponse(path, media_type="model/obj", filename=Path(path).name)


@app.get("/api/jobs/{job_id}/download/urdf", tags=["Download"])
def download_urdf(job_id: str):
    """Download the URDF file with physics parameters."""
    job = _require_completed(job_id)
    path = job.files.urdf
    if not path or not Path(path).exists():
        raise HTTPException(404, "URDF file not found")
    return FileResponse(path, media_type="application/xml", filename=Path(path).name)


@app.get("/api/jobs/{job_id}/download/mjcf", tags=["Download"])
def download_mjcf(job_id: str):
    """Download the MuJoCo MJCF XML file."""
    job = _require_completed(job_id)
    path = job.files.mjcf
    if not path or not Path(path).exists():
        raise HTTPException(404, "MJCF file not found")
    return FileResponse(path, media_type="application/xml", filename=Path(path).name)


@app.get("/api/jobs/{job_id}/download/all", tags=["Download"])
def download_all(job_id: str):
    """Download all generated files as a ZIP archive (OBJ + MTL + textures + URDF + MJCF)."""
    job = _require_completed(job_id)

    # Collect the result directory
    urdf_path = job.files.urdf
    if not urdf_path:
        raise HTTPException(404, "Result files not found")

    result_dir = Path(urdf_path).parent

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in result_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(result_dir))
    buf.seek(0)

    filename = f"{job.name}_3d_asset.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
