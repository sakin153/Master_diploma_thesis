import gc

import torch

from asset_gen.utils.log import logger

__all__ = ["log_vram", "free_vram"]


def log_vram(tag: str = "") -> None:
    """Log current VRAM usage. No-op if CUDA is unavailable."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    label = f" [{tag}]" if tag else ""
    logger.info(f"VRAM{label}: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved")


def free_vram() -> None:
    """Force Python GC + CUDA cache clear."""
    gc.collect()
    gc.collect()  # second pass catches cyclic references missed by first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
