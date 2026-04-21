# Project EmbodiedGen
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import argparse
import os
import random
from glob import glob
from shutil import copy, copytree, rmtree

import numpy as np
import trimesh
from PIL import Image
from embodied_gen.data.backproject_v3 import entrypoint as backproject_api
from embodied_gen.data.utils import delete_dir
from embodied_gen.models.gs_model import GaussianOperator
from embodied_gen.models.segment_model import RembgRemover
from embodied_gen.scripts.render_gs import entrypoint as render_gs_api
from embodied_gen.utils.gpt_clients import GPT_CLIENT
from embodied_gen.utils.inference import image3d_model_infer
from embodied_gen.utils.log import logger
from embodied_gen.utils.process_media import (
    combine_images_to_grid,
    merge_images_video,
)
from embodied_gen.utils.tags import VERSION
from embodied_gen.utils.trender import render_video
from embodied_gen.utils.vram_utils import free_vram, log_vram
from embodied_gen.validators.quality_checkers import (
    BaseChecker,
    ImageAestheticChecker,
    ImageSegChecker,
    MeshGeoChecker,
)
from embodied_gen.validators.urdf_convertor import URDFGenerator

# ── Active 3-D model ─────────────────────────────────────────────────────────
# "SAM3D"    — Meta SAM-3D-Objects (GS + mesh, requires ~10-14 GB)
# "TRELLIS"  — Microsoft TRELLIS (GS + mesh, requires ~20-25 GB)
# "HUNYUAN3D"— Tencent Hunyuan3D-2mini (mesh only, ~6-10 GB)
IMAGE3D_MODEL = "HUNYUAN3D"

# Quality checkers: GPT-based, no VRAM, safe at module level.
RBG_REMOVER = RembgRemover()
SEG_CHECKER = ImageSegChecker(GPT_CLIENT)
GEO_CHECKER = MeshGeoChecker(GPT_CLIENT)
AESTHETIC_CHECKER = ImageAestheticChecker()
CHECKERS = [GEO_CHECKER, SEG_CHECKER, AESTHETIC_CHECKER]

# ── Lazy 3-D generation pipeline ─────────────────────────────────────────────
_PIPELINE = None


def _release_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        del _PIPELINE
        _PIPELINE = None
        free_vram()
        log_vram("after 3D pipeline release")


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        log_vram("before 3D pipeline load")
        logger.info(f"Loading {IMAGE3D_MODEL} as Image3D model...")
        if IMAGE3D_MODEL == "TRELLIS":
            from thirdparty.TRELLIS.trellis.pipelines import (
                TrellisImageTo3DPipeline,
            )
            _PIPELINE = TrellisImageTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS-image-large"
            )
        elif IMAGE3D_MODEL == "SAM3D":
            from embodied_gen.models.sam3d import Sam3dInference
            _PIPELINE = Sam3dInference()
        elif IMAGE3D_MODEL == "HUNYUAN3D":
            from embodied_gen.models.hunyuan3d import Hunyuan3DInference
            _PIPELINE = Hunyuan3DInference()
        log_vram("after 3D pipeline load")
    return _PIPELINE


# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Image to 3D pipeline args.")
    parser.add_argument(
        "--image_path", type=str, nargs="+", help="Path to the input images."
    )
    parser.add_argument(
        "--image_root", type=str, help="Path to the input images folder."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        help="Root directory for saving outputs.",
    )
    parser.add_argument(
        "--height_range",
        type=str,
        default=None,
        help="The hight in meter to restore the mesh real size.",
    )
    parser.add_argument(
        "--mass_range",
        type=str,
        default=None,
        help="The mass in kg to restore the mesh real weight.",
    )
    parser.add_argument("--asset_type", type=str, nargs="+", default=None)
    parser.add_argument("--skip_exists", action="store_true")
    parser.add_argument("--version", type=str, default=VERSION)
    parser.add_argument("--keep_intermediate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_retry", type=int, default=3)
    parser.add_argument("--disable_decompose_convex", action="store_true")
    parser.add_argument("--texture_size", type=int, default=1024)
    args, unknown = parser.parse_known_args()
    return args


def entrypoint(**kwargs):
    args = parse_args()
    for k, v in kwargs.items():
        if hasattr(args, k) and v is not None:
            setattr(args, k, v)

    assert (
        args.image_path or args.image_root
    ), "Please provide either --image_path or --image_root."
    if not args.image_path:
        args.image_path = glob(os.path.join(args.image_root, "*.png"))
        args.image_path += glob(os.path.join(args.image_root, "*.jpg"))
        args.image_path += glob(os.path.join(args.image_root, "*.jpeg"))

    for idx, image_path in enumerate(args.image_path):
        try:
            filename = os.path.basename(image_path).split(".")[0]
            output_root = args.output_root
            if args.image_root is not None or len(args.image_path) > 1:
                output_root = os.path.join(output_root, filename)
            os.makedirs(output_root, exist_ok=True)

            mesh_out = f"{output_root}/{filename}.obj"
            if args.skip_exists and os.path.exists(mesh_out):
                logger.warning(
                    f"Skip {image_path}, already processed in {mesh_out}"
                )
                continue

            image = Image.open(image_path)
            image.save(f"{output_root}/{filename}_raw.png")

            seg_path = f"{output_root}/{filename}_cond.png"
            seg_image = (
                RBG_REMOVER(image) if image.mode != "RGBA" else image
            )
            seg_image.save(seg_path)

            seed = args.seed
            asset_node = "unknown"
            gs_model = None
            mesh_model = None
            trimesh_result = None
            aligned_gs_path = None

            if isinstance(args.asset_type, list) and args.asset_type[idx]:
                asset_node = args.asset_type[idx]

            # ── Stage 1: Image → 3D ───────────────────────────────────────
            rot_matrix = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
            gs_add_rot = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
            mesh_add_rot = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]

            for try_idx in range(args.n_retry):
                logger.info(
                    f"Try: {try_idx + 1}/{args.n_retry}, "
                    f"Seed: {seed}, Prompt: {seg_path}"
                )
                try:
                    outputs = image3d_model_infer(
                        _get_pipeline(), seg_image, seed
                    )
                except Exception as e:
                    logger.error(
                        f"[Image3D Failed] {image_path}: {e}, "
                        f"retry {try_idx+1}/{args.n_retry}"
                    )
                    seed = (
                        random.randint(0, 100000)
                        if seed is not None
                        else None
                    )
                    continue

                gs_model = outputs["gaussian"][0]   # None for Hunyuan3D
                mesh_model = outputs["mesh"][0]
                trimesh_result = outputs.get("trimesh", [None])[0]

                # ── GS path (SAM3D / TRELLIS only) ───────────────────────
                if gs_model is not None:
                    gs_path = mesh_out.replace(".obj", "_gs.ply")
                    gs_model.save_ply(gs_path)

                    gs_rot = np.array(gs_add_rot) @ np.array(rot_matrix)
                    pose = GaussianOperator.trans_to_quatpose(gs_rot)
                    aligned_gs_path = gs_path.replace(".ply", "_aligned.ply")
                    GaussianOperator.resave_ply(
                        in_ply=gs_path,
                        out_ply=aligned_gs_path,
                        instance_pose=pose,
                        device="cpu",
                    )
                    color_path = os.path.join(output_root, "color.png")
                    render_gs_api(
                        input_gs=aligned_gs_path,
                        output_path=color_path,
                        elevation=[30, -30],
                        num_images=4,
                    )
                    color_img = Image.open(color_path)
                    geo_flag, geo_result = GEO_CHECKER(
                        [color_img], text=asset_node
                    )
                    logger.warning(
                        f"{GEO_CHECKER.__class__.__name__}: "
                        f"{geo_result} for {seg_path}"
                    )
                    if geo_flag is True or geo_flag is None:
                        break
                else:
                    # Hunyuan3D: no GS quality check — accept first result.
                    logger.info(
                        "Hunyuan3D: skipping GS quality check, "
                        "using first result."
                    )
                    break

                seed = (
                    random.randint(0, 100000)
                    if seed is not None
                    else None
                )

            if mesh_model is None:
                logger.error(
                    f"Exceed image3d retry num, skip {image_path}."
                )
                continue

            # ── Stage 2: Video + mesh export ──────────────────────────────
            if gs_model is not None:
                color_images = render_video(gs_model, r=1.85)["color"]
            else:
                color_images = render_video(mesh_model, r=1.85).get(
                    "color", []
                )
            normal_images = render_video(mesh_model, r=1.85).get(
                "normal", []
            )
            video_path = os.path.join(output_root, "gs_mesh.mp4")
            if color_images or normal_images:
                merge_images_video(color_images, normal_images, video_path)

            if trimesh_result is not None:
                # Hunyuan3D already provides a complete trimesh;
                # apply the same axis rotation used by SAM3D path.
                mesh = trimesh_result
                mesh.vertices = (
                    mesh.vertices
                    @ np.array(mesh_add_rot)
                    @ np.array(rot_matrix)
                )
            else:
                mesh = trimesh.Trimesh(
                    vertices=mesh_model.vertices.cpu().numpy(),
                    faces=mesh_model.faces.cpu().numpy(),
                )
                mesh.vertices = mesh.vertices @ np.array(mesh_add_rot)
                mesh.vertices = mesh.vertices @ np.array(rot_matrix)

            mesh_obj_path = os.path.join(output_root, f"{filename}.obj")
            mesh.export(mesh_obj_path)

            # Release large output tensors before texture baking.
            del color_images, normal_images
            if gs_model is not None:
                del gs_model
            del mesh_model
            free_vram()
            log_vram("after releasing 3D outputs")

            # ── Stage 3: Texture baking ───────────────────────────────────
            if aligned_gs_path is not None:
                # SAM3D / TRELLIS path: project GS colours onto mesh.
                mesh = backproject_api(
                    gs_path=aligned_gs_path,
                    mesh_path=mesh_obj_path,
                    output_path=mesh_obj_path,
                    skip_fix_mesh=False,
                    texture_size=args.texture_size,
                    delight=False,
                )
            # Hunyuan3D path: mesh already textured — skip backproject_api.

            mesh_glb_path = os.path.join(output_root, f"{filename}.glb")
            mesh.export(mesh_glb_path)

            # ── Stage 4: URDF ─────────────────────────────────────────────
            urdf_convertor = URDFGenerator(
                GPT_CLIENT,
                render_view_num=4,
                decompose_convex=not args.disable_decompose_convex,
            )
            asset_attrs = {"version": VERSION}
            if aligned_gs_path is not None:
                asset_attrs["gs_model"] = (
                    f"{urdf_convertor.output_mesh_dir}/{filename}_gs.ply"
                )
            if args.height_range:
                min_h, max_h = map(float, args.height_range.split("-"))
                asset_attrs["min_height"] = min_h
                asset_attrs["max_height"] = max_h
            if args.mass_range:
                min_m, max_m = map(float, args.mass_range.split("-"))
                asset_attrs["min_mass"] = min_m
                asset_attrs["max_mass"] = max_m
            if isinstance(args.asset_type, list) and args.asset_type[idx]:
                asset_attrs["category"] = args.asset_type[idx]
            if args.version:
                asset_attrs["version"] = args.version

            urdf_root = f"{output_root}/URDF_{filename}"
            urdf_path = urdf_convertor(
                mesh_path=mesh_obj_path,
                output_root=urdf_root,
                **asset_attrs,
            )

            # Rescale GS if available.
            real_height = urdf_convertor.get_attr_from_urdf(
                urdf_path, attr_name="real_height"
            )
            if aligned_gs_path is not None:
                out_gs = (
                    f"{urdf_root}/{urdf_convertor.output_mesh_dir}"
                    f"/{filename}_gs.ply"
                )
                GaussianOperator.resave_ply(
                    in_ply=aligned_gs_path,
                    out_ply=out_gs,
                    real_height=real_height,
                    device="cpu",
                )

            # Quality check and update .urdf.
            mesh_out_final = (
                f"{urdf_root}/{urdf_convertor.output_mesh_dir}"
                f"/{filename}.obj"
            )
            trimesh.load(mesh_out_final).export(
                mesh_out_final.replace(".obj", ".glb")
            )

            image_dir = (
                f"{urdf_root}/{urdf_convertor.output_render_dir}"
                f"/image_color"
            )
            image_paths = glob(f"{image_dir}/*.png")
            images_list = []
            for checker in CHECKERS:
                images = combine_images_to_grid(image_paths)
                if isinstance(checker, ImageSegChecker):
                    images = [
                        f"{output_root}/{filename}_raw.png",
                        f"{output_root}/{filename}_cond.png",
                    ]
                images_list.append(images)

            qa_results = BaseChecker.validate(CHECKERS, images_list)
            urdf_convertor.add_quality_tag(urdf_path, qa_results)

            # Organize final results.
            result_dir = f"{output_root}/result"
            if os.path.exists(result_dir):
                rmtree(result_dir, ignore_errors=True)
            os.makedirs(result_dir, exist_ok=True)
            copy(urdf_path, f"{result_dir}/{os.path.basename(urdf_path)}")
            copytree(
                f"{urdf_root}/{urdf_convertor.output_mesh_dir}",
                f"{result_dir}/{urdf_convertor.output_mesh_dir}",
            )
            if os.path.exists(video_path):
                copy(video_path, f"{result_dir}/video.mp4")

            if not args.keep_intermediate:
                delete_dir(output_root, keep_subs=["result"])

            logger.info(f"Saved results for {image_path} in {result_dir}")

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}, skip.")
            continue

    logger.info(f"Processing complete. Outputs saved to {args.output_root}")


if __name__ == "__main__":
    entrypoint()
