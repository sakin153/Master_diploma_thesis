from enum import Enum

__all__ = ["AssetType", "RenderItems", "SimAssetMapper"]


class RenderItems(str, Enum):
    IMAGE = "image_color"
    ALPHA = "image_mask"
    VIEW_NORMAL = "image_view_normal"
    GLOBAL_NORMAL = "image_global_normal"
    DEPTH = "image_depth"


class AssetType(str):
    MJCF = "mjcf"
    USD = "usd"
    URDF = "urdf"
    MESH = "mesh"


class SimAssetMapper:
    _mapping = dict(
        MUJOCO=AssetType.MJCF,
        GENESIS=AssetType.MJCF,
        ISAACGYM=AssetType.URDF,
        PYBULLET=AssetType.URDF,
    )

    @classmethod
    def __class_getitem__(cls, key: str):
        return cls._mapping[key.upper()]
