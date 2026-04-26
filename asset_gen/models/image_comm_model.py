# Compatibility shim — module was renamed to model_image_runtime.py.
from asset_gen.models.model_image_runtime import (  # noqa: F401
    PIPELINE_REGISTRY,
    build_hf_image_pipeline,
)
