from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModelChoice(str, Enum):
    sd15 = "sd15"
    sdxl_turbo = "sdxl-turbo"
    kolors = "kolors"
    sd35 = "sd35"


class JobState(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class GenerateItemRequest(BaseModel):
    """One object to generate — either text prompt or image (base64)."""
    name: Optional[str] = Field(
        None,
        description="Asset name. Defaults to first word of prompt.",
    )
    prompt: Optional[str] = Field(
        None, description="Text description of the object."
    )
    image_b64: Optional[str] = Field(
        None, description="Base64-encoded input image (PNG/JPG)."
    )
    asset_type: Optional[str] = Field(
        None, description="Semantic category hint (e.g. 'chair')."
    )
    seed_img: Optional[int] = Field(
        None, description="Seed for image generation."
    )
    seed_3d: int = Field(0, description="Seed for 3D generation.")


class GenerateRequest(BaseModel):
    """Batch generation request — one job generates N objects."""
    items: list[GenerateItemRequest] = Field(
        ...,
        min_length=1,
        description="List of objects to generate (text or image each).",
    )
    model: ModelChoice = Field(
        ModelChoice.sd15, description="Text-to-image model."
    )
    n_image_retry: int = Field(2, ge=1, le=5)
    n_asset_retry: int = Field(2, ge=1, le=5)


class ItemFiles(BaseModel):
    """Output file paths for one generated object."""
    obj: Optional[str] = None
    glb: Optional[str] = None
    urdf: Optional[str] = None
    mjcf: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: JobState
    item_count: int = 1
    model: str
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    # name → files for each generated object
    results: dict[str, ItemFiles] = Field(default_factory=dict)
    queue_position: Optional[int] = None
    logs: list[str] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    job_id: str
    status: JobState
    queue_position: int
