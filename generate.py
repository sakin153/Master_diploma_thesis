"""
Main generation script: text prompt → 3D model → MuJoCo-ready MJCF.

Usage:
    python generate.py "a red ceramic mug"
    python generate.py "small wooden chair" --output outputs/chair --name chair
    python generate.py "iron dumbbell" --seed_img 42 --seed_3d 0
    python generate.py "a wooden chair" --texture
"""

import argparse
import glob
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 3D model from a text prompt "
            "and export to MuJoCo MJCF."
        )
    )
    parser.add_argument(
        "prompt", type=str, help="Text description of the object."
    )
    parser.add_argument(
        "--output", type=str, default="outputs/generated",
        help="Output directory (default: outputs/generated).",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Asset name. Defaults to first word of prompt.",
    )
    parser.add_argument(
        "--seed_img", type=int, default=None,
        help="Random seed for image generation.",
    )
    parser.add_argument(
        "--seed_3d", type=int, default=0,
        help="Random seed for 3D generation (default: 0).",
    )
    parser.add_argument(
        "--n_image_retry", type=int, default=2,
        help="Max retries for image generation (default: 2).",
    )
    parser.add_argument(
        "--n_asset_retry", type=int, default=2,
        help="Max retries for 3D generation (default: 2).",
    )
    parser.add_argument(
        "--model", type=str, default="sdxl-turbo",
        choices=["sdxl-turbo", "kolors", "sd35", "flux", "chroma", "cosmos"],
        help="Text-to-image model (default: sdxl-turbo for 8 GB GPU).",
    )
    parser.add_argument(
        "--texture", action="store_true",
        help=(
            "Apply Hunyuan3D-Paint-Turbo texture after mesh generation "
            "(~6 GB VRAM, ~2-3 min extra)."
        ),
    )
    parser.add_argument(
        "--skip_mjcf", action="store_true",
        help="Skip URDF → MJCF conversion (output URDF only).",
    )
    return parser.parse_args()


def find_urdf(result_dir: str, name: str) -> str | None:
    pattern = os.path.join(result_dir, "**", "*.urdf")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    for path in matches:
        if os.path.basename(path) == f"{name}.urdf":
            return path
    return matches[0]


def main():
    args = parse_args()

    if args.name is None:
        args.name = (
            args.prompt.split()[0].lower()
            .replace(",", "").replace(".", "")
        )

    # Set model before textto3d is imported (reads TEXT_MODEL at module level)
    os.environ["TEXT_MODEL"] = args.model

    print("\n=== EmbodiedGen: text → 3D → MuJoCo ===")
    print(f"  Prompt  : {args.prompt}")
    print(f"  Name    : {args.name}")
    print(f"  Model   : {args.model}")
    print(f"  Texture : {args.texture}")
    print(f"  Output  : {args.output}")
    print("=" * 40)

    # Step 1: Text → Image → 3D mesh → URDF
    steps = "1/3" if not args.skip_mjcf else "1/2"
    print(f"\n[{steps}] Generating 3D asset from prompt...")

    from asset_gen.scripts.textto3d import GenerateItem, text_to_3d

    results = text_to_3d(
        items=[GenerateItem(
            name=args.name,
            prompt=args.prompt,
            seed_img=args.seed_img,
            seed_3d=args.seed_3d,
        )],
        output_root=args.output,
        n_image_retry=args.n_image_retry,
        n_asset_retry=args.n_asset_retry,
        n_pipe_retry=1,
        enable_texture=args.texture,
    )

    asset_rel = results.get("assets", {}).get(args.name)
    if not asset_rel:
        print("\n[ERROR] Pipeline did not produce an asset. Check logs above.")
        sys.exit(1)

    asset_dir = os.path.join(args.output, asset_rel)
    urdf_path = find_urdf(asset_dir, args.name)
    if not urdf_path:
        print(f"\n[ERROR] URDF not found in {asset_dir}")
        sys.exit(1)

    print(f"\n  URDF  → {urdf_path}")

    if args.skip_mjcf:
        print("\n=== Done (URDF only) ===")
        return

    # Step 2: URDF → MJCF for MuJoCo
    print("\n[2/2] Converting URDF → MJCF for MuJoCo...")
    from asset_gen.data.asset_converter import cvt_asset_gen_asset_to_anysim
    from asset_gen.utils.enum import AssetType

    mjcf_dir = os.path.join(os.path.dirname(urdf_path), "mjcf")
    asset_paths = cvt_asset_gen_asset_to_anysim(
        urdf_files=[urdf_path],
        target_dirs=[mjcf_dir],
        target_type=AssetType.MJCF,
        source_type=AssetType.MESH,
    )

    mjcf_path = asset_paths.get(urdf_path)
    mesh_path = os.path.join(
        os.path.dirname(urdf_path), "mesh", f"{args.name}.obj"
    )
    print(f"  MJCF  → {mjcf_path}")

    print("\n=== Done ===")
    print(f"  3D mesh : {mesh_path}")
    print(f"  URDF    : {urdf_path}")
    print(f"  MJCF    : {mjcf_path}")
    print()


if __name__ == "__main__":
    main()
