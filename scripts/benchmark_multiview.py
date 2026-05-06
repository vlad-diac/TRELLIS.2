"""
Multi-View TRELLIS.2 Benchmark — Full Test Matrix

Runs the five conditions from the research plan on a fixed set of marine
(or other) vessel images and records voxel counts, wall-clock times, and
exports one GLB per condition for visual comparison.

Memory model
------------
Images are preprocessed once with a temporary pipeline that is then
unloaded.  Each benchmark condition loads a fresh pipeline, runs to
completion, saves the GLB, then unloads the pipeline and flushes the
CUDA cache before the next condition starts.  This prevents GPU memory
fragmentation from building up across conditions and ensures every
condition starts from a clean 24 GB state.

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

# Skip scaffold conditions
python scripts/benchmark_multiview.py \\
    front.png side.png \\
    --skip-scaffold --output-dir ./out

# Sparse-only (fast voxel count sweep, no GLB export)
python scripts/benchmark_multiview.py \\
    front.png right.png rear.png left.png \\
    --sparse-only --output-dir ./out

# Custom azimuths (3 views at 0°, 45°, 200°)
python scripts/benchmark_multiview.py \\
    img1.png img2.png img3.png \\
    --azimuths 0 45 200 --elevation 20 \\
    --output-dir ./out

# Faster 512-resolution pipeline for quick iteration
python scripts/benchmark_multiview.py \\
    front.png rear.png \\
    --pipeline 512 --output-dir ./out
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
# Add scripts dir so scaffold_bypass can be imported
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from scaffold_bypass import (
    make_look_at_rotation,
    space_carve,
    occupancy_to_coords,
    run_scaffold_bypass,
)


# ---------------------------------------------------------------------------
# Pipeline lifecycle helpers
# ---------------------------------------------------------------------------

def load_pipeline(model: str) -> "Trellis2ImageTo3DPipeline":
    pl = Trellis2ImageTo3DPipeline.from_pretrained(model)
    pl.cuda()
    return pl


def unload_pipeline(pipeline) -> None:
    """Delete the pipeline and flush CUDA memory completely."""
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


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


def coords_iou(a: torch.Tensor, b: torch.Tensor, grid_size: int = 64) -> float:
    """IoU between two sparse coord sets via dense grid comparison."""
    def to_dense(coords: torch.Tensor) -> torch.Tensor:
        vol = torch.zeros(grid_size, grid_size, grid_size, dtype=torch.bool)
        c = coords.cpu()[:, 1:].clamp(0, grid_size - 1)
        vol[c[:, 0], c[:, 1], c[:, 2]] = True
        return vol
    va, vb = to_dense(a), to_dense(b)
    intersection = (va & vb).sum().item()
    union = (va | vb).sum().item()
    return intersection / union if union > 0 else 0.0


def silhouette_from_rgba(image: Image.Image) -> np.ndarray:
    """
    Extract a binary foreground mask from an RGBA PIL image by reading the
    alpha channel directly.  Assumes the image has already been preprocessed
    by the pipeline (background = alpha 0, object = alpha > 0).
    """
    return np.array(image.convert("RGBA"))[:, :, 3] > 0


# ---------------------------------------------------------------------------
# Per-condition runners
# ---------------------------------------------------------------------------

def run_baseline(pipeline, images, args) -> Dict:
    ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[args.pipeline]
    torch.manual_seed(args.seed)
    if args.sparse_only:
        cond = pipeline.get_cond([images[0]], 512)
        coords = pipeline.sample_sparse_structure(cond, ss_res)
        return {"coords": coords, "glb_path": None}
    meshes = pipeline.run(
        images[0], preprocess_image=False,
        pipeline_type=args.pipeline, seed=args.seed,
    )
    # Stage-1 re-run for coord stats (adds ~10s but no GPU residual)
    torch.manual_seed(args.seed)
    cond = pipeline.get_cond([images[0]], 512)
    coords = pipeline.sample_sparse_structure(cond, ss_res)
    glb_path = None
    if meshes:
        glb_path = save_glb(
            pipeline, meshes[0],
            ss_res if args.pipeline == "512" else 1024,
            args.output_dir, f"baseline_{timestamp()}",
            args.texture_size, args.decimation_target,
        )
    return {"coords": coords, "glb_path": glb_path}


def run_p1(pipeline, images, fusion_mode, args) -> Dict:
    ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[args.pipeline]
    torch.manual_seed(args.seed)
    if args.sparse_only:
        cond = pipeline.get_cond_multi(images, 512, fusion_mode)
        coords = pipeline.sample_sparse_structure(cond, ss_res)
        return {"coords": coords, "glb_path": None}
    meshes = pipeline.run_multi_image_cond(
        images, cond_fusion_mode=fusion_mode,
        preprocess_image=False, pipeline_type=args.pipeline, seed=args.seed,
    )
    torch.manual_seed(args.seed)
    cond = pipeline.get_cond_multi(images, 512, fusion_mode)
    coords = pipeline.sample_sparse_structure(cond, ss_res)
    label = f"P1-{fusion_mode}_{timestamp()}"
    glb_path = None
    if meshes:
        glb_path = save_glb(
            pipeline, meshes[0],
            ss_res if args.pipeline == "512" else 1024,
            args.output_dir, label,
            args.texture_size, args.decimation_target,
        )
    return {"coords": coords, "glb_path": glb_path}


def run_p2(pipeline, images, occupancy, cond_fusion_mode, label, args) -> Dict:
    ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[args.pipeline]
    coords = occupancy_to_coords(occupancy, device=pipeline.device)
    if args.sparse_only:
        return {"coords": coords, "glb_path": None}
    result = run_scaffold_bypass(
        pipeline=pipeline,
        occupancy=occupancy,
        conditioning_images=images,
        cond_fusion_mode=cond_fusion_mode,
        pipeline_type=args.pipeline,
        seed=args.seed,
    )
    glb_path = None
    if result["meshes"]:
        glb_path = save_glb(
            pipeline, result["meshes"][0],
            ss_res if args.pipeline == "512" else 1024,
            args.output_dir, f"{label}_{timestamp()}",
            args.texture_size, args.decimation_target,
        )
    return {"coords": result["coords"], "glb_path": glb_path}


# ---------------------------------------------------------------------------
# Condition dispatcher
# ---------------------------------------------------------------------------

def run_condition(
    name: str,
    images: List[Image.Image],
    args: argparse.Namespace,
    occupancy: Optional[np.ndarray],
) -> Dict:
    """
    Load a fresh pipeline, run one condition, save the GLB, unload the
    pipeline, and return a result dict.

    The pipeline is fully unloaded (weights deleted, CUDA cache flushed)
    before this function returns so the next condition starts from a clean
    GPU state.
    """
    result = {
        "name": name, "coords": None, "voxels": 0,
        "elapsed_s": 0.0, "glb_path": None, "error": None,
    }
    t0 = time.time()
    pipeline = None
    try:
        print(f"  Loading pipeline for {name} ...")
        pipeline = load_pipeline(args.model)

        if name == "baseline":
            out = run_baseline(pipeline, images, args)
        elif name == "P1-mean":
            out = run_p1(pipeline, images, "mean", args)
        elif name == "P1-concat":
            out = run_p1(pipeline, images, "concat", args)
        elif name == "P2-scaffold":
            if occupancy is None:
                raise RuntimeError("No scaffold — pass --skip-scaffold to omit P2 conditions.")
            out = run_p2(pipeline, images, occupancy, "primary", "P2-scaffold", args)
        elif name == "P2-scaffold+P1":
            if occupancy is None:
                raise RuntimeError("No scaffold — pass --skip-scaffold to omit P2 conditions.")
            out = run_p2(pipeline, images, occupancy, "mean", "P2-scaffold+P1", args)
        else:
            raise ValueError(f"Unknown condition: {name!r}")

        result["coords"] = out["coords"]
        result["voxels"] = int(out["coords"].shape[0]) if out["coords"] is not None else 0
        result["glb_path"] = out["glb_path"]

    except Exception as exc:
        result["error"] = str(exc)
        print(f"  [ERROR] {exc}")

    finally:
        if pipeline is not None:
            print(f"  Unloading pipeline ...")
            unload_pipeline(pipeline)

    result["elapsed_s"] = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results: List[Dict], baseline_coords: Optional[torch.Tensor]) -> None:
    print_section("Benchmark Summary")
    hdr = f"{'Condition':<22} {'Voxels':>8} {'Elapsed':>9}  {'IoU vs base':>12}  Status"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        iou_str = "—"
        if r["coords"] is not None and baseline_coords is not None and r["name"] != "baseline":
            iou = coords_iou(r["coords"], baseline_coords)
            iou_str = f"{iou:.3f}"
        status = "ERROR" if r["error"] else ("GLB" if r["glb_path"] else "sparse-only")
        print(f"{r['name']:<22} {r['voxels']:>8} {r['elapsed_s']:>8.1f}s  "
              f"{iou_str:>12}  {status}")
        if r["glb_path"]:
            print(f"    → {r['glb_path']}")
    print()


def save_json_summary(results: List[Dict], output_dir: str, ts: str) -> str:
    rows = [{"name": r["name"], "voxels": r["voxels"],
             "elapsed_s": round(r["elapsed_s"], 2),
             "glb_path": r["glb_path"], "error": r["error"]}
            for r in results]
    path = os.path.join(output_dir, f"benchmark_summary_{ts}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
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
    p.add_argument("--output-dir", default="./out")
    p.add_argument("--azimuths", nargs="+", type=float, default=None,
                   help="Camera azimuth per image in degrees (default: evenly spaced 360°).")
    p.add_argument("--elevation", type=float, default=15.0)
    p.add_argument("--skip-scaffold", action="store_true",
                   help="Skip P2-scaffold and P2-scaffold+P1.")
    p.add_argument("--sparse-only", action="store_true",
                   help="Skip shape/tex SLAT; only count voxels.")
    p.add_argument("--pipeline", default="1024_cascade",
                   choices=["512", "1024", "1024_cascade", "1536_cascade"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    p.add_argument("--texture-size", type=int, default=1024)
    p.add_argument("--decimation-target", type=int, default=500_000)
    p.add_argument("--min-views-scaffold", type=int, default=None,
                   help="Min views a voxel must be inside the silhouette to "
                        "survive space carving (default: all views = true visual hull). "
                        "Try 2 or 3 if the hull is too sparse.")
    p.add_argument("--conditions", nargs="+",
                   choices=["baseline", "P1-mean", "P1-concat",
                            "P2-scaffold", "P2-scaffold+P1"],
                   default=None)
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

    print_section("TRELLIS.2 Multi-View Benchmark")
    print(f"  Images    : {args.images}")
    print(f"  Conditions: {conditions}")
    print(f"  Pipeline  : {args.pipeline}")
    print(f"  Seed      : {args.seed}")
    print(f"  Output    : {args.output_dir}")

    # -----------------------------------------------------------------------
    # Step 1: preprocess images ONCE with a temporary pipeline, then unload.
    # Silhouettes are extracted directly from the RGBA alpha channel of the
    # preprocessed output — no second rembg pass.
    # -----------------------------------------------------------------------
    print_section("Preprocessing images (temp pipeline)")
    t0 = time.time()
    tmp_pipeline = load_pipeline(args.model)

    n = len(args.images)
    raw_images = [Image.open(p).convert("RGB") for p in args.images]
    images = [tmp_pipeline.preprocess_image(img) for img in raw_images]
    print(f"  Preprocessed {n} image(s) in {time.time() - t0:.1f}s")

    # Determine azimuths
    if args.azimuths is not None:
        if len(args.azimuths) != n:
            raise ValueError(
                f"--azimuths has {len(args.azimuths)} values but {n} images provided."
            )
        azimuths = list(args.azimuths)
    else:
        azimuths = [360.0 * i / n for i in range(n)]
        print(f"  Auto azimuths: {[f'{a:.1f}°' for a in azimuths]}")

    # Build scaffold occupancy (CPU-only) while pipeline is still loaded for
    # any optional future use, then unload.
    occupancy = None
    if not args.skip_scaffold and any(
        c in conditions for c in ("P2-scaffold", "P2-scaffold+P1")
    ):
        print_section("Building visual-hull scaffold")
        # Extract silhouettes from the already-preprocessed RGBA alpha channels.
        # Do NOT call preprocess_image again — that would double-run rembg and
        # produce incorrect (all-opaque) masks.
        scaffold_masks = [silhouette_from_rgba(img) for img in images]
        scaffold_rotations = [
            make_look_at_rotation(az, args.elevation) for az in azimuths
        ]
        for i, m in enumerate(scaffold_masks):
            print(f"  View {i}: {m.sum():,} foreground pixels "
                  f"({100*m.mean():.1f}% of frame)")

        ss_res = {"512": 32, "1024": 64,
                  "1024_cascade": 32, "1536_cascade": 32}[args.pipeline]
        min_views = (args.min_views_scaffold
                     if args.min_views_scaffold is not None
                     else len(scaffold_masks))
        occupancy = space_carve(
            scaffold_masks, scaffold_rotations,
            grid_size=ss_res, min_views=min_views,
        )
        n_occ = int(occupancy.sum())
        density = 100 * n_occ / occupancy.size
        print(f"  Scaffold: {n_occ:,} / {occupancy.size:,} voxels "
              f"({density:.1f}% density, min_views={min_views})")
        if density > 80:
            print("  WARNING: scaffold density >80% — the visual hull is nearly "
                  "full. Check that azimuths match the actual camera positions. "
                  "Try --min-views-scaffold with a lower value if using >4 views.")

    print_section("Unloading temp pipeline")
    unload_pipeline(tmp_pipeline)
    print("  GPU memory cleared.")

    # -----------------------------------------------------------------------
    # Step 2: run each condition with its own fresh pipeline.
    # -----------------------------------------------------------------------
    run_ts = timestamp()
    results = []
    baseline_coords = None

    for cond_name in conditions:
        print_section(f"Condition: {cond_name}")
        r = run_condition(
            name=cond_name,
            images=images,
            args=args,
            occupancy=occupancy,
        )
        results.append(r)
        print(f"  Voxels : {r['voxels']:,}")
        print(f"  Time   : {r['elapsed_s']:.1f}s")
        if r["error"]:
            print(f"  Error  : {r['error']}")
        if cond_name == "baseline" and r["coords"] is not None:
            baseline_coords = r["coords"].cpu()

    # -----------------------------------------------------------------------
    # Step 3: summary.
    # -----------------------------------------------------------------------
    print_summary(results, baseline_coords)
    json_path = save_json_summary(results, args.output_dir, run_ts)
    print(f"Summary JSON: {json_path}")
    print_section("Done")


if __name__ == "__main__":
    main()
