"""
Multi-View TRELLIS.2 Benchmark — Full Test Matrix

Runs the five conditions from the research plan on a fixed set of marine
(or other) vessel images and records voxel counts, wall-clock times, and
exports one GLB per condition for visual comparison.

Test matrix
-----------
baseline
    Single-image pipeline (first image only).  The reference point.

P1-mean
    Condition fusion — joint mean of DINOv3 patch tokens from all views.
    One stage-1 sample from the joint condition.

P1-concat
    Condition fusion — DINOv3 tokens concatenated across views (V×N tokens).
    One stage-1 sample from the joint condition.

P2-scaffold
    External visual-hull bypass.  Space-carve silhouettes at assumed-orbit
    azimuths.  Feed coords directly to stage-2.  Primary conditioning only.

P2-scaffold+P1
    External visual-hull bypass + joint mean conditioning for stage-2.
    Expected to combine geometric stability with richer semantic context.

Usage examples
--------------
# Minimal run — all 5 conditions on 4 views at default azimuths
python scripts/benchmark_multiview.py \\
    front.png right.png rear.png left.png \\
    --output-dir ./benchmark_out

# Skip scaffold conditions (if cameras are not calibrated)
python scripts/benchmark_multiview.py \\
    front.png side.png \\
    --skip-scaffold --output-dir ./out

# Sparse-only (fast voxel count sweep, ~40 s per cond)
python scripts/benchmark_multiview.py \\
    front.png right.png rear.png left.png \\
    --sparse-only --output-dir ./out

# Custom azimuths (e.g. 3 views at 0°, 45°, 200°)
python scripts/benchmark_multiview.py \\
    img1.png img2.png img3.png \\
    --azimuths 0 45 200 --elevation 20 \\
    --output-dir ./out

# Use a faster 512-resolution pipeline for quick iteration
python scripts/benchmark_multiview.py \\
    front.png rear.png \\
    --pipeline 512 --output-dir ./out
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline

# Import scaffold utilities from the sibling script
from scaffold_bypass import (
    extract_silhouette,
    make_look_at_rotation,
    space_carve,
    occupancy_to_coords,
    run_scaffold_bypass,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"


def print_section(title: str) -> None:
    width = 64
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def save_glb(pipeline, mesh, res: int, output_dir: str, label: str,
             texture_size: int, decimation_target: int) -> str:
    import o_voxel
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
        verbose=False,
    )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{label}.glb")
    glb.export(path, extension_webp=True)
    return path


def voxel_count_from_coords(coords: torch.Tensor) -> int:
    return int(coords.shape[0])


def coords_iou(a: torch.Tensor, b: torch.Tensor, grid_size: int = 64) -> float:
    """
    Approximate IoU between two sparse coordinate sets by converting to dense
    binary grids and computing set overlap.

    Only meaningful when both coord sets use the same grid resolution.
    """
    def to_dense(coords: torch.Tensor, G: int) -> torch.Tensor:
        vol = torch.zeros(G, G, G, dtype=torch.bool, device="cpu")
        c = coords.cpu()[:, 1:]  # drop batch idx
        c = c.clamp(0, G - 1)
        vol[c[:, 0], c[:, 1], c[:, 2]] = True
        return vol

    va = to_dense(a, grid_size)
    vb = to_dense(b, grid_size)
    intersection = (va & vb).sum().item()
    union = (va | vb).sum().item()
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Benchmark conditions
# ---------------------------------------------------------------------------

def run_condition(
    name: str,
    pipeline: "Trellis2ImageTo3DPipeline",
    images: List[Image.Image],
    azimuths: List[float],
    elevation: float,
    args: argparse.Namespace,
    scaffold_masks: Optional[List[np.ndarray]] = None,
    scaffold_rotations: Optional[list] = None,
) -> Dict:
    """
    Run one benchmark condition and return a result dict.

    Returns:
        dict with keys: name, coords, voxels, elapsed_s, glb_path (or None),
        error (None if success).
    """
    result = {"name": name, "coords": None, "voxels": 0,
              "elapsed_s": 0.0, "glb_path": None, "error": None}
    t0 = time.time()

    try:
        pipeline_type = args.pipeline
        ss_res = {"512": 32, "1024": 64,
                  "1024_cascade": 32, "1536_cascade": 32}[pipeline_type]

        torch.manual_seed(args.seed)

        # --- Baseline: single image ---
        if name == "baseline":
            if args.sparse_only:
                cond = pipeline.get_cond([images[0]], 512)
                coords = pipeline.sample_sparse_structure(cond, ss_res)
            else:
                meshes = pipeline.run(
                    images[0],
                    preprocess_image=False,
                    pipeline_type=pipeline_type,
                    seed=args.seed,
                )
                # Re-run just for coords (run() doesn't return them)
                cond = pipeline.get_cond([images[0]], 512)
                torch.manual_seed(args.seed)
                coords = pipeline.sample_sparse_structure(cond, ss_res)
                if meshes and not args.sparse_only:
                    result["glb_path"] = save_glb(
                        pipeline, meshes[0], ss_res if pipeline_type == "512" else
                        (1024 if "1024" in pipeline_type else 512),
                        args.output_dir, f"{name}_{timestamp()}",
                        args.texture_size, args.decimation_target,
                    )

        # --- P1-mean / P1-concat: condition fusion ---
        elif name in ("P1-mean", "P1-concat"):
            fusion_mode = "mean" if name == "P1-mean" else "concat"
            if args.sparse_only:
                joint_cond = pipeline.get_cond_multi(images, 512, fusion_mode)
                torch.manual_seed(args.seed)
                coords = pipeline.sample_sparse_structure(joint_cond, ss_res)
            else:
                meshes = pipeline.run_multi_image_cond(
                    images,
                    cond_fusion_mode=fusion_mode,
                    preprocess_image=False,
                    pipeline_type=pipeline_type,
                    seed=args.seed,
                )
                joint_cond = pipeline.get_cond_multi(images, 512, fusion_mode)
                torch.manual_seed(args.seed)
                coords = pipeline.sample_sparse_structure(joint_cond, ss_res)
                if meshes and not args.sparse_only:
                    result["glb_path"] = save_glb(
                        pipeline, meshes[0],
                        ss_res if pipeline_type == "512" else 1024,
                        args.output_dir, f"{name}_{timestamp()}",
                        args.texture_size, args.decimation_target,
                    )

        # --- P2-scaffold: external visual hull bypass ---
        elif name == "P2-scaffold":
            if scaffold_masks is None:
                raise RuntimeError("Scaffold masks not available — pass --skip-scaffold to skip P2 conditions.")
            occupancy = space_carve(
                scaffold_masks, scaffold_rotations,
                grid_size=ss_res,
                min_views=len(scaffold_masks),
            )
            coords = occupancy_to_coords(occupancy, device=pipeline.device)
            if not args.sparse_only:
                bypass_result = run_scaffold_bypass(
                    pipeline=pipeline,
                    occupancy=occupancy,
                    conditioning_images=images,
                    cond_fusion_mode="primary",
                    pipeline_type=pipeline_type,
                    seed=args.seed,
                )
                if bypass_result["meshes"] and not args.sparse_only:
                    result["glb_path"] = save_glb(
                        pipeline, bypass_result["meshes"][0],
                        ss_res if pipeline_type == "512" else 1024,
                        args.output_dir, f"{name}_{timestamp()}",
                        args.texture_size, args.decimation_target,
                    )

        # --- P2-scaffold+P1: visual hull bypass + joint mean cond ---
        elif name == "P2-scaffold+P1":
            if scaffold_masks is None:
                raise RuntimeError("Scaffold masks not available — pass --skip-scaffold to skip P2 conditions.")
            occupancy = space_carve(
                scaffold_masks, scaffold_rotations,
                grid_size=ss_res,
                min_views=len(scaffold_masks),
            )
            coords = occupancy_to_coords(occupancy, device=pipeline.device)
            if not args.sparse_only:
                bypass_result = run_scaffold_bypass(
                    pipeline=pipeline,
                    occupancy=occupancy,
                    conditioning_images=images,
                    cond_fusion_mode="mean",
                    pipeline_type=pipeline_type,
                    seed=args.seed,
                )
                if bypass_result["meshes"] and not args.sparse_only:
                    result["glb_path"] = save_glb(
                        pipeline, bypass_result["meshes"][0],
                        ss_res if pipeline_type == "512" else 1024,
                        args.output_dir, f"{name}_{timestamp()}",
                        args.texture_size, args.decimation_target,
                    )

        else:
            raise ValueError(f"Unknown condition: {name!r}")

        result["coords"] = coords
        result["voxels"] = voxel_count_from_coords(coords)

    except Exception as exc:
        result["error"] = str(exc)
        print(f"  [ERROR] {name}: {exc}")

    result["elapsed_s"] = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results: List[Dict], baseline_coords: Optional[torch.Tensor]) -> None:
    print_section("Benchmark Summary")
    header = f"{'Condition':<20} {'Voxels':>8} {'Elapsed':>9}  {'IoU vs baseline':>16}  Status"
    print(header)
    print("-" * len(header))
    for r in results:
        iou_str = "—"
        if (r["coords"] is not None and baseline_coords is not None
                and r["name"] != "baseline"):
            iou = coords_iou(r["coords"], baseline_coords)
            iou_str = f"{iou:.3f}"
        status = "ERROR" if r["error"] else ("GLB" if r["glb_path"] else "sparse-only")
        print(f"{r['name']:<20} {r['voxels']:>8} {r['elapsed_s']:>8.1f}s  "
              f"{iou_str:>16}  {status}")
        if r["glb_path"]:
            print(f"  → {r['glb_path']}")
    print()


def save_json_summary(results: List[Dict], output_dir: str, ts: str) -> str:
    summary = []
    for r in results:
        summary.append({
            "name": r["name"],
            "voxels": r["voxels"],
            "elapsed_s": round(r["elapsed_s"], 2),
            "glb_path": r["glb_path"],
            "error": r["error"],
        })
    path = os.path.join(output_dir, f"benchmark_summary_{ts}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full multi-view benchmark test matrix for TRELLIS.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("images", nargs="+",
                   help="Input image paths, one per viewpoint (first = front).")
    p.add_argument("--output-dir", default="./out",
                   help="Output directory for GLBs and summary JSON.")
    p.add_argument("--azimuths", nargs="+", type=float, default=None,
                   help="Camera azimuth per image in degrees. "
                        "Default: evenly spaced over 360°.")
    p.add_argument("--elevation", type=float, default=15.0,
                   help="Camera elevation in degrees (default: 15).")
    p.add_argument("--skip-scaffold", action="store_true",
                   help="Skip P2-scaffold and P2-scaffold+P1 conditions.")
    p.add_argument("--sparse-only", action="store_true",
                   help="Only compute voxel counts; skip shape/tex SLAT and GLB export.")
    p.add_argument("--pipeline", default="1024_cascade",
                   choices=["512", "1024", "1024_cascade", "1536_cascade"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    p.add_argument("--texture-size", type=int, default=1024)
    p.add_argument("--decimation-target", type=int, default=500_000)
    p.add_argument("--conditions", nargs="+",
                   choices=["baseline", "P1-mean", "P1-concat",
                            "P2-scaffold", "P2-scaffold+P1"],
                   default=None,
                   help="Subset of conditions to run (default: all).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_conditions = ["baseline", "P1-mean", "P1-concat",
                      "P2-scaffold", "P2-scaffold+P1"]
    if args.conditions:
        conditions = args.conditions
    elif args.skip_scaffold:
        conditions = ["baseline", "P1-mean", "P1-concat"]
    else:
        conditions = all_conditions

    print_section(f"TRELLIS.2 Multi-View Benchmark")
    print(f"  Images    : {args.images}")
    print(f"  Conditions: {conditions}")
    print(f"  Pipeline  : {args.pipeline}")
    print(f"  Seed      : {args.seed}")
    print(f"  Output    : {args.output_dir}")

    # --- Load pipeline ---
    print_section("Loading pipeline")
    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    print(f"Pipeline loaded in {time.time() - t0:.1f}s")

    # --- Load and preprocess images ---
    print_section("Loading images")
    raw_images = [Image.open(p).convert("RGB") for p in args.images]
    images = [pipeline.preprocess_image(img) for img in raw_images]
    print(f"Loaded and preprocessed {len(images)} image(s)")

    # --- Determine camera azimuths ---
    n = len(images)
    if args.azimuths is not None:
        if len(args.azimuths) != n:
            raise ValueError(
                f"--azimuths has {len(args.azimuths)} values but {n} images provided."
            )
        azimuths = args.azimuths
    else:
        azimuths = [360.0 * i / n for i in range(n)]
        print(f"Auto azimuths: {[f'{a:.1f}°' for a in azimuths]}")

    # --- Extract silhouettes for scaffold conditions ---
    scaffold_masks = None
    scaffold_rotations = None
    if not args.skip_scaffold and any(c in conditions for c in ("P2-scaffold", "P2-scaffold+P1")):
        print_section("Extracting silhouettes for scaffold")
        scaffold_masks = [extract_silhouette(img, pipeline) for img in images]
        scaffold_rotations = [
            make_look_at_rotation(az, args.elevation) for az in azimuths
        ]
        print(f"Silhouettes extracted: {[m.sum() for m in scaffold_masks]} foreground pixels")

    # --- Run conditions ---
    run_ts = timestamp()
    results = []
    baseline_coords = None

    for cond_name in conditions:
        print_section(f"Condition: {cond_name}")
        r = run_condition(
            name=cond_name,
            pipeline=pipeline,
            images=images,
            azimuths=azimuths,
            elevation=args.elevation,
            args=args,
            scaffold_masks=scaffold_masks,
            scaffold_rotations=scaffold_rotations,
        )
        results.append(r)
        print(f"  Voxels : {r['voxels']:,}")
        print(f"  Time   : {r['elapsed_s']:.1f}s")
        if r["error"]:
            print(f"  Error  : {r['error']}")
        if cond_name == "baseline" and r["coords"] is not None:
            baseline_coords = r["coords"]

    # --- Summary ---
    print_summary(results, baseline_coords)

    json_path = save_json_summary(results, args.output_dir, run_ts)
    print(f"Summary JSON: {json_path}")

    print_section("Done")


if __name__ == "__main__":
    main()
