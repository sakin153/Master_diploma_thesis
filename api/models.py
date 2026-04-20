from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModelChoice(str, Enum):
    sdxl_turbo = "sdxl-turbo"
    kolors = "kolors"
    sd35 = "sd35"


class JobState(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the object")
    name: Optional[str] = Field(None, description="Asset name (defaults to first word of prompt)")
    model: ModelChoice = Field(ModelChoice.sdxl_turbo, description="Text-to-image model")
    seed_img: Optional[int] = Field(None, description="Seed for image generation")
    seed_3d: int = Field(0, description="Seed for 3D generation")
    n_image_retry: int = Field(2, ge=1, le=5)
    n_asset_retry: int = Field(2, ge=1, le=5)


class JobFiles(BaseModel):
    obj: Optional[str] = None
    urdf: Optional[str] = None
    mjcf: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: JobState
    prompt: str
    name: str
    model: str
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    files: Optional[JobFiles] = None
    queue_position: Optional[int] = None
    logs: list[str] = Field(default_factory=list, description="Recent log lines")


class GenerateResponse(BaseModel):
    job_id: str
    status: JobState
    queue_position: int
