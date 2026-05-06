"""
Test script for multi-image sparse occupancy fusion.

Usage examples
--------------
# Single image (baseline)
python scripts/test_multi_image_fusion.py front.png --output-dir ./out

# Multi-image union fusion
python scripts/test_multi_image_fusion.py front.png side.png rear.png \\
    --fusion union --output-dir ./out

# Multi-image majority vote
python scripts/test_multi_image_fusion.py front.png side.png rear.png \\
    --fusion vote --vote-threshold 0.5 --output-dir ./out

# Logit-mean fusion (fuse raw decoder scores, single threshold at 0.0)
python scripts/test_multi_image_fusion.py front.png side.png rear.png \\
    --fusion logit-mean --output-dir ./out

# Logit-max fusion with spatial smoothing
python scripts/test_multi_image_fusion.py front.png side.png rear.png \\
    --fusion logit-max --logit-smooth-sigma 0.8 --output-dir ./out

# Logit-mean + ensemble ×3 (most evidence, 3× sparse-stage compute)
python scripts/test_multi_image_fusion.py front.png side.png rear.png \\
    --fusion logit-mean --samples-per-view 3 --output-dir ./out

# Side-by-side comparison: runs single-image baseline AND multi-image fusion,
# saves both GLBs, prints voxel count statistics.
python scripts/test_multi_image_fusion.py front.png side.png rear.png \\
    --compare --fusion logit-mean --output-dir ./out

# Use a lower-resolution pipeline to speed up testing
python scripts/test_multi_image_fusion.py front.png side.png \\
    --pipeline 512 --compare --output-dir ./out
"""

import argparse
import os
import sys
import time

# Ensure the repo root is on the path when called from anywhere
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datetime import datetime

import torch
from PIL import Image

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"


def save_glb(pipeline, mesh, res: int, output_dir: str, label: str, texture_size: int, decimation_target: int) -> str:
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{label}_{timestamp()}.glb")
    glb.export(path, extension_webp=True)
    return path


def print_section(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def voxel_count_from_coords(coords: torch.Tensor) -> int:
    return int(coords.shape[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test multi-image sparse occupancy fusion for TRELLIS.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "images",
        nargs="+",
        metavar="IMAGE",
        help="One or more input image paths (first is the primary/front view).",
    )
    parser.add_argument(
        "--output-dir",
        default="./out",
        help="Directory to write output GLB files (default: ./out).",
    )
    parser.add_argument(
        "--fusion",
        choices=["union", "vote", "logit-mean", "logit-max", "logit-sum"],
        default="union",
        help=(
            "Occupancy fusion mode (default: union).  "
            "'union'/'vote' threshold each view first then merge (binary).  "
            "'logit-mean'/'logit-max'/'logit-sum' preserve raw decoder scores "
            "across all views and apply a single threshold after fusion."
        ),
    )
    parser.add_argument(
        "--vote-threshold",
        type=float,
        default=0.5,
        help="Fraction of views that must agree for 'vote' mode (default: 0.5).",
    )
    parser.add_argument(
        "--logit-threshold",
        type=float,
        default=0.0,
        metavar="T",
        help=(
            "Threshold applied to the fused logit volume in logit-* modes "
            "(default: 0.0, the decoder's natural decision boundary).  "
            "Lower values keep more voxels; raise to tighten."
        ),
    )
    parser.add_argument(
        "--logit-smooth-sigma",
        type=float,
        default=0.0,
        metavar="S",
        help=(
            "Gaussian spatial smoothing sigma (in voxels) applied to the "
            "fused logit volume before thresholding in logit-* modes.  "
            "Reinforces weak-evidence voxels surrounded by stronger neighbours.  "
            "0.0 = disabled (default).  0.5–1.5 is a good starting range."
        ),
    )
    parser.add_argument(
        "--samples-per-view",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Run diffusion N times per view and average raw logits before "
            "thresholding.  Reduces per-view hallucinations at N× the compute "
            "cost.  3–5 is a good range (default: 1 = no ensemble)."
        ),
    )
    parser.add_argument(
        "--filter-min-neighbors",
        type=int,
        default=0,
        metavar="K",
        help=(
            "After fusion, remove voxels with fewer than K occupied neighbours "
            "in their 3×3×3 block.  Cleans floating hallucination islands "
            "without eroding real geometry.  0 = disabled (default).  "
            "Recommended starting point: 2."
        ),
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run a single-image baseline for comparison.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
        default="1024_cascade",
        help="Pipeline type (default: 1024_cascade).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip background removal / crop preprocessing.",
    )
    parser.add_argument(
        "--model",
        default="microsoft/TRELLIS.2-4B",
        help="HuggingFace model ID or local path (default: microsoft/TRELLIS.2-4B).",
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=1024,
        help="Texture atlas resolution in pixels (default: 1024).",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=200000,
        help="Target face count for mesh decimation (default: 200000).",
    )
    parser.add_argument(
        "--ss-steps",
        type=int,
        default=12,
        help="Sparse structure diffusion steps (default: 12).",
    )
    parser.add_argument(
        "--shape-steps",
        type=int,
        default=12,
        help="Shape SLAT diffusion steps (default: 12).",
    )
    parser.add_argument(
        "--tex-steps",
        type=int,
        default=12,
        help="Texture SLAT diffusion steps (default: 12).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load images ---
    print_section("Loading images")
    images = []
    for path in args.images:
        if not os.path.isfile(path):
            print(f"ERROR: image not found: {path}", file=sys.stderr)
            sys.exit(1)
        img = Image.open(path).convert("RGBA")
        images.append(img)
        print(f"  loaded: {path}  ({img.width}×{img.height})")

    # --- Load model ---
    print_section(f"Loading pipeline: {args.model}")
    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    print(f"  pipeline loaded in {time.time() - t0:.1f}s")

    preprocess = not args.no_preprocess
    pipeline_type_map = {
        "512": "512",
        "1024": "1024",
        "1024_cascade": "1024_cascade",
        "1536_cascade": "1536_cascade",
    }
    pipeline_type = pipeline_type_map[args.pipeline]

    ss_params = {"steps": args.ss_steps}
    shape_params = {"steps": args.shape_steps}
    tex_params = {"steps": args.tex_steps}

    results = {}

    # -----------------------------------------------------------------------
    # Optional single-image baseline
    # -----------------------------------------------------------------------
    if args.compare or len(images) == 1:
        print_section("Single-image baseline (first image only)")
        t0 = time.time()
        torch.manual_seed(args.seed)

        # Peek at per-view occupancy count for the primary image
        if preprocess:
            primary = pipeline.preprocess_image(images[0])
        else:
            primary = images[0]
        cond_primary = pipeline.get_cond([primary], 512)
        ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[pipeline_type]
        single_vol = pipeline._sample_occupancy_volume(cond_primary, 1, ss_params)
        single_voxels = int(single_vol.sum().item())
        print(f"  single-image occupancy: {single_voxels} voxels")

        out_single, (shape_slat_s, tex_slat_s, res_s) = pipeline.run(
            images[0],
            seed=args.seed,
            preprocess_image=preprocess,
            sparse_structure_sampler_params=ss_params,
            shape_slat_sampler_params=shape_params,
            tex_slat_sampler_params=tex_params,
            pipeline_type=pipeline_type,
            return_latent=True,
        )
        elapsed_single = time.time() - t0
        print(f"  completed in {elapsed_single:.1f}s")

        if len(images) > 1 or args.compare:
            glb_path = save_glb(
                pipeline, out_single[0], res_s, args.output_dir,
                "single", args.texture_size, args.decimation
            )
            print(f"  saved: {glb_path}")
        results["single"] = {"voxels": single_voxels, "elapsed": elapsed_single}
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Multi-image fusion (only when more than one image is supplied)
    # -----------------------------------------------------------------------
    if len(images) > 1:
        extras = []
        if args.fusion == "vote":
            extras.append(f"threshold={args.vote_threshold}")
        if args.fusion in ("logit-mean", "logit-max", "logit-sum"):
            if args.logit_threshold != 0.0:
                extras.append(f"thr={args.logit_threshold}")
            if args.logit_smooth_sigma > 0.0:
                extras.append(f"smooth={args.logit_smooth_sigma}")
        if args.samples_per_view > 1:
            extras.append(f"×{args.samples_per_view}/view")
        if args.filter_min_neighbors > 0:
            extras.append(f"filter≥{args.filter_min_neighbors}nb")
        label_extras = (", " + ", ".join(extras)) if extras else ""
        print_section(
            f"Multi-image fusion  [{args.fusion}{label_extras}]  ({len(images)} images)"
        )
        t0 = time.time()

        out_multi, (shape_slat_m, tex_slat_m, res_m) = pipeline.run_multi_image(
            images,
            seed=args.seed,
            fusion_mode=args.fusion,
            vote_threshold=args.vote_threshold,
            logit_threshold=args.logit_threshold,
            logit_smooth_sigma=args.logit_smooth_sigma,
            samples_per_view=args.samples_per_view,
            filter_min_neighbors=args.filter_min_neighbors,
            preprocess_image=preprocess,
            sparse_structure_sampler_params=ss_params,
            shape_slat_sampler_params=shape_params,
            tex_slat_sampler_params=tex_params,
            pipeline_type=pipeline_type,
            return_latent=True,
        )
        elapsed_multi = time.time() - t0
        print(f"  completed in {elapsed_multi:.1f}s")

        file_label_parts = [f"multi_{args.fusion.replace('-', '_')}"]
        if args.fusion in ("logit-mean", "logit-max", "logit-sum"):
            if args.logit_threshold != 0.0:
                file_label_parts.append(f"thr{args.logit_threshold}")
            if args.logit_smooth_sigma > 0.0:
                file_label_parts.append(f"sm{args.logit_smooth_sigma}")
        if args.samples_per_view > 1:
            file_label_parts.append(f"spv{args.samples_per_view}")
        if args.filter_min_neighbors > 0:
            file_label_parts.append(f"fn{args.filter_min_neighbors}")
        glb_path = save_glb(
            pipeline, out_multi[0], res_m, args.output_dir,
            "_".join(file_label_parts), args.texture_size, args.decimation
        )
        print(f"  saved: {glb_path}")

        # Fused voxel count comes from the shape_slat coords
        fused_voxels = int(shape_slat_m.coords.shape[0])
        results["multi"] = {"voxels": fused_voxels, "elapsed": elapsed_multi}
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print_section("Summary")
    if "single" in results:
        sv = results["single"]["voxels"]
        print(f"  single-image  voxels : {sv:>8,}   time: {results['single']['elapsed']:.1f}s")
    if "multi" in results:
        mv = results["multi"]["voxels"]
        print(f"  multi-image   voxels : {mv:>8,}   time: {results['multi']['elapsed']:.1f}s")
        if "single" in results and sv > 0:
            pct = (mv - sv) / sv * 100
            sign = "+" if pct >= 0 else ""
            print(f"  voxel delta          : {sign}{pct:.1f}% vs single")
    print()


if __name__ == "__main__":
    main()
