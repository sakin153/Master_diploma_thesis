import io
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from api.job_manager import job_manager
from api.models import GenerateRequest, GenerateResponse, JobState, JobStatus

app = FastAPI(
    title="asset_generator API",
    description="Text/image → 3D model (OBJ GLB URDF and MuJoCo MJCF)",
    version="2.0.0",
)

# Health
@app.get("/health", tags=["System"])
def health():
    return {"status": "КАЙФ БРАТОК"}


# Generation
@app.post(
    "/api/generate", response_model=GenerateResponse, tags=["Generation"]
)
def generate(request: GenerateRequest):
    """
    Submit a batch of objects for 3D generation.

    Each item can be a text prompt or a base64-encoded image.
    Returns a `job_id` to track progress.
    """
    job = job_manager.submit(request)
    return GenerateResponse(
        job_id=job.job_id,
        status=job.status,
        queue_position=job.queue_position or 1,
    )



# Jobs
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
def get_job_logs(job_id: str, last: int = 200):
    """Return last N log lines for a running or completed job."""
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job.status,
        "error": job.error,
        "logs": job.logs[-last:],
    }


# ──────────────────────────────────────────────
# Downloads (per object within a job)
# ──────────────────────────────────────────────

def _require_completed(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != JobState.completed:
        raise HTTPException(400, f"Job is {job.status}, not completed")
    if not job.results:
        raise HTTPException(500, "Job completed but no files recorded")
    return job


def _get_item_files(job_id: str, name: str):
    job = _require_completed(job_id)
    item = job.results.get(name)
    if not item:
        raise HTTPException(
            404, f"Object '{name}' not found in job. "
            f"Available: {list(job.results.keys())}"
        )
    return item


@app.get(
    "/api/jobs/{job_id}/objects/{name}/download/obj",
    tags=["Download"],
)
def download_obj(job_id: str, name: str):
    """Download OBJ mesh for a specific object."""
    item = _get_item_files(job_id, name)
    path = item.obj
    if not path or not Path(path).exists():
        raise HTTPException(404, "OBJ file not found")
    return FileResponse(
        path, media_type="model/obj", filename=Path(path).name
    )


@app.get(
    "/api/jobs/{job_id}/objects/{name}/download/glb",
    tags=["Download"],
)
def download_glb(job_id: str, name: str):
    """Download GLB mesh for a specific object."""
    item = _get_item_files(job_id, name)
    path = item.glb
    if not path or not Path(path).exists():
        raise HTTPException(404, "GLB file not found")
    return FileResponse(
        path, media_type="model/gltf-binary", filename=Path(path).name
    )


@app.get(
    "/api/jobs/{job_id}/objects/{name}/download/urdf",
    tags=["Download"],
)
def download_urdf(job_id: str, name: str):
    """Download URDF file with physics parameters."""
    item = _get_item_files(job_id, name)
    path = item.urdf
    if not path or not Path(path).exists():
        raise HTTPException(404, "URDF file not found")
    return FileResponse(
        path, media_type="application/xml", filename=Path(path).name
    )


@app.get(
    "/api/jobs/{job_id}/objects/{name}/download/mjcf",
    tags=["Download"],
)
def download_mjcf(job_id: str, name: str):
    """Download MuJoCo MJCF XML file."""
    item = _get_item_files(job_id, name)
    path = item.mjcf
    if not path or not Path(path).exists():
        raise HTTPException(404, "MJCF file not found")
    return FileResponse(
        path, media_type="application/xml", filename=Path(path).name
    )


@app.get(
    "/api/jobs/{job_id}/objects/{name}/download/all",
    tags=["Download"],
)
def download_all_for_object(job_id: str, name: str):
    """Download ZIP with all files for a specific object."""
    item = _get_item_files(job_id, name)
    urdf_path = item.urdf or item.obj
    if not urdf_path:
        raise HTTPException(404, "Result files not found")

    result_dir = Path(urdf_path).parent
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in result_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(result_dir))
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f"attachment; filename={name}_3d_asset.zip"
            )
        },
    )


@app.get("/api/jobs/{job_id}/download/all", tags=["Download"])
def download_all_job(job_id: str):
    """Download ZIP with all files for every object in the job."""
    job = _require_completed(job_id)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for obj_name, item_files in job.results.items():
            ref_path = item_files.urdf or item_files.obj
            if not ref_path:
                continue
            result_dir = Path(ref_path).parent
            for file_path in result_dir.rglob("*"):
                if file_path.is_file():
                    arc_name = Path(obj_name) / file_path.relative_to(
                        result_dir
                    )
                    zf.write(file_path, arc_name)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f"attachment; filename=job_{job_id}_all.zip"
            )
        },
    )
