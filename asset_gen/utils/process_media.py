import logging
import math
import mimetypes
import os
from typing import Union

import cv2
import imageio
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    "render_asset3d",
    "merge_images_video",
    "filter_small_connected_components",
    "filter_image_small_connected_components",
    "combine_images_to_grid",
    "is_image_file",
    "check_object_edge_truncated",
    "vcat_pil_images",
]


def _render_with_open3d(
    mesh_path: str,
    out_dir: str,
    num_images: int,
    elevation: Union[tuple, float],
) -> list[str]:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    W, H = 512, 512
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    renderer.scene.set_background([0.85, 0.85, 0.85, 1.0])

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", mesh, mat)

    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = float(np.linalg.norm(extent)) * 1.5

    elev_angles = list(elevation) if isinstance(elevation, (tuple, list)) else [elevation]
    azimuths = [i * 360.0 / num_images for i in range(num_images)]

    paths = []
    for i, azimuth in enumerate(azimuths):
        elev = elev_angles[i % len(elev_angles)]
        rad_az = np.radians(azimuth)
        rad_el = np.radians(elev)
        eye = np.array([
            center[0] + radius * np.cos(rad_el) * np.cos(rad_az),
            center[1] + radius * np.cos(rad_el) * np.sin(rad_az),
            center[2] + radius * np.sin(rad_el),
        ])
        renderer.setup_camera(60.0, center.tolist(), eye.tolist(), [0.0, 0.0, 1.0])
        img = renderer.render_to_image()
        path = os.path.join(out_dir, f"view_{i:04d}.png")
        o3d.io.write_image(path, img)
        paths.append(path)

    return paths


def _render_with_matplotlib(
    mesh_path: str,
    out_dir: str,
    num_images: int,
    elevation: Union[tuple, float],
) -> list[str]:
    import trimesh
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, no display needed
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        if geoms:
            mesh = trimesh.util.concatenate(geoms)
        else:
            logger.error("render_with_matplotlib: empty scene")
            return []

    elev_angles = list(elevation) if isinstance(elevation, (tuple, list)) else [elevation]
    azimuths = [i * 360.0 / num_images for i in range(num_images)]
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    half_ext = (bounds[1] - bounds[0]).max() * 0.6

    paths = []
    for i, azimuth in enumerate(azimuths):
        elev = elev_angles[i % len(elev_angles)]
        fig = plt.figure(figsize=(5.12, 5.12), dpi=100, facecolor="#D9D9D9")
        ax = fig.add_subplot(111, projection="3d", facecolor="#D9D9D9")

        verts = mesh.vertices[mesh.faces]
        poly = Poly3DCollection(
            verts, alpha=0.9,
            facecolor="#C8C8C8", edgecolor="#888888", linewidth=0.05,
        )
        ax.add_collection3d(poly)

        ax.set_xlim(center[0] - half_ext, center[0] + half_ext)
        ax.set_ylim(center[1] - half_ext, center[1] + half_ext)
        ax.set_zlim(center[2] - half_ext, center[2] + half_ext)
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azimuth)

        path = os.path.join(out_dir, f"view_{i:04d}.png")
        plt.savefig(path, dpi=100, bbox_inches="tight", facecolor="#D9D9D9")
        plt.close(fig)
        paths.append(path)

    return paths


def render_asset3d(
    mesh_path: str,
    output_root: str,
    num_images: int = 4,
    elevation: Union[tuple, float] = (30.0, -30.0),
    output_subdir: str = "renders",
    no_index_file: bool = False,
    **kwargs,
) -> list[str]:
    """Render a mesh from multiple viewpoints.

    Tries Open3D OffscreenRenderer first; falls back to matplotlib (works on
    Windows where EGL headless is unavailable).

    Saves images to {output_root}/{output_subdir}/image_color/view_XXXX.png
    Returns list of saved image paths.
    """
    out_dir = os.path.join(output_root, output_subdir, "image_color")
    os.makedirs(out_dir, exist_ok=True)

    try:
        paths = _render_with_open3d(mesh_path, out_dir, num_images, elevation)
        logger.info(f"render_asset3d: rendered {len(paths)} views via Open3D")
        return paths
    except Exception as e:
        logger.warning(f"Open3D render failed ({e}), falling back to matplotlib...")

    try:
        paths = _render_with_matplotlib(mesh_path, out_dir, num_images, elevation)
        logger.info(f"render_asset3d: rendered {len(paths)} views via matplotlib")
        return paths
    except Exception as e:
        logger.error(f"render_asset3d: all renderers failed for {mesh_path}: {e}")
        return []


def merge_images_video(
    color_images: list,
    normal_images: list,
    output_path: str,
) -> None:
    """Merge color and normal image lists into a side-by-side video."""
    if not color_images:
        return
    width = color_images[0].shape[1]
    combined = [
        np.hstack([c[:, : width // 2], n[:, width // 2 :]])
        for c, n in zip(color_images, normal_images)
    ]
    imageio.mimsave(output_path, combined, fps=50)


def filter_small_connected_components(
    mask: Union[Image.Image, np.ndarray],
    area_ratio: float,
    connectivity: int = 8,
) -> np.ndarray:
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
    )
    small = np.zeros_like(mask, dtype=np.uint8)
    mask_area = (mask != 0).sum()
    min_area = mask_area // area_ratio
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            small[labels == label] = 255
    return cv2.bitwise_and(mask, cv2.bitwise_not(small))


def filter_image_small_connected_components(
    image: Union[Image.Image, np.ndarray],
    area_ratio: float = 10,
    connectivity: int = 8,
) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGBA"))
    mask = image[..., 3]
    mask = filter_small_connected_components(mask, area_ratio, connectivity)
    image[..., 3] = mask
    return image


def combine_images_to_grid(
    images: list,
    cat_row_col: tuple = None,
    target_wh: tuple = (512, 512),
    image_mode: str = "RGB",
) -> list:
    n = len(images)
    if n == 0:
        return []
    if n == 1:
        return images

    if cat_row_col is None:
        n_col = math.ceil(math.sqrt(n))
        n_row = math.ceil(n / n_col)
    else:
        n_row, n_col = cat_row_col

    imgs = [
        Image.open(p).convert(image_mode) if isinstance(p, str) else p
        for p in images
    ]
    imgs = [img.resize(target_wh) for img in imgs]

    grid_w, grid_h = n_col * target_wh[0], n_row * target_wh[1]
    grid = Image.new(image_mode, (grid_w, grid_h), (0, 0, 0))
    for idx, img in enumerate(imgs):
        row, col = divmod(idx, n_col)
        grid.paste(img, (col * target_wh[0], row * target_wh[1]))

    return [grid]


def is_image_file(filename: str) -> bool:
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type is not None and mime_type.startswith("image")


def check_object_edge_truncated(
    mask: np.ndarray, edge_threshold: int = 5
) -> bool:
    top = mask[:edge_threshold, :].any()
    bottom = mask[-edge_threshold:, :].any()
    left = mask[:, :edge_threshold].any()
    right = mask[:, -edge_threshold:].any()
    return not (top or bottom or left or right)


def vcat_pil_images(
    images: list, image_mode: str = "RGB"
) -> Image.Image:
    widths, heights = zip(*(img.size for img in images))
    total_h = sum(heights)
    max_w = max(widths)
    out = Image.new(image_mode, (max_w, total_h))
    y = 0
    for img in images:
        out.paste(img, (0, y))
        y += img.size[1]
    return out
