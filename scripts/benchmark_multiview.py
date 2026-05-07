"""
Multi-View TRELLIS.2 Benchmark — Full Test Matrix

Runs the five conditions from the research plan on a fixed set of marine
(or other) vessel images and records voxel counts, wall-clock times, and
exports one model + preview per condition for visual comparison.

Memory model
------------
Images are preprocessed once with a temporary pipeline that is then
unloaded.  Each benchmark condition loads a fresh pipeline, runs to
completion, saves outputs, then unloads the pipeline and flushes the
CUDA cache before the next condition starts.  This prevents GPU memory
fragmentation from building up across conditions and ensures every
condition starts from a clean 24 GB state.

Output layout
-------------
Each invocation creates a timestamped run folder inside --output-dir::

    {output_dir}/
    └── run_YYYYMMDD_HHMMSS/
        ├── baseline/
        │   ├── model.glb       (model.obj with --skip-tex)
        │   └── preview.png
        ├── P1-mean/
        │   ├── model.glb
        │   └── preview.png
        ├── P1-concat/ ...
        ├── P2-scaffold/ ...
        ├── P2-scaffold+P1/ ...
        └── summary.json

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
    --output-dir ./out

# Skip scaffold conditions
python scripts/benchmark_multiview.py \\
    front.png side.png \\
    --skip-scaffold --output-dir ./out

# Geometry-only (no texture SLAT — much faster, lower VRAM)
python scripts/benchmark_multiview.py \\
    front.png right.png rear.png left.png \\
    --skip-tex --output-dir ./out

# Sparse-only (fast voxel count sweep, no model export)
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

def run_timestamp() -> str:
    """Human-readable timestamp for run folder names (no colons)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_section(title: str) -> None:
    width = 64
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def cond_folder_name(condition_name: str) -> str:
    """Convert a condition name to a safe directory name."""
    return condition_name.replace("+", "_plus_")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_glb(pipeline, mesh, res: int, cond_dir: str,
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
    path = os.path.join(cond_dir, "model.glb")
    glb.export(path, extension_webp=True)
    return path


def save_obj(mesh, cond_dir: str) -> str:
    """Export a plain geometry-only Mesh as .obj via trimesh."""
    import trimesh
    tm = trimesh.Trimesh(
        vertices=mesh.vertices.cpu().float().numpy(),
        faces=mesh.faces.cpu().int().numpy(),
    )
    path = os.path.join(cond_dir, "model.obj")
    tm.export(path)
    return path


def save_preview(mesh, cond_dir: str, resolution: int = 512) -> Optional[str]:
    """
    Render a 4-view snapshot of a mesh and save as preview.png.

    Works with both MeshWithVoxel (PbrMeshRenderer) and plain Mesh
    (MeshRenderer) — render_utils.get_renderer() dispatches automatically.
    Must be called while the pipeline is still loaded on GPU.
    """
    from trellis2.utils import render_utils
    try:
        snapshot = render_utils.render_snapshot(
            mesh, resolution=resolution, r=2, fov=36, nviews=4,
        )
        frames = snapshot.get("shaded", next(iter(snapshot.values())))
        strip = np.concatenate(frames[:4], axis=1)  # (H, W*4, C)
        path = os.path.join(cond_dir, "preview.png")
        Image.fromarray(strip).save(path)
        return path
    except Exception as exc:
        print(f"  [WARN] Preview rendering failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Silhouette extraction (must use rembg directly on raw RGB images)
# ---------------------------------------------------------------------------

def extract_silhouette_rembg(pipeline, raw_rgb: Image.Image) -> np.ndarray:
    """
    Run rembg on a raw RGB image and return a binary foreground mask.

    preprocess_image() must NOT be used here: it premultiplies alpha into
    the RGB channels and returns a 3-channel image, so converting back to
    RGBA gives alpha=255 everywhere (all foreground → full-cube scaffold).
    """
    rgb = raw_rgb.convert("RGB")
    if getattr(pipeline, "low_vram", False):
        pipeline.rembg_model.to(pipeline.device)
    rgba = pipeline.rembg_model(rgb)
    if getattr(pipeline, "low_vram", False):
        pipeline.rembg_model.cpu()
    return np.array(rgba)[:, :, 3] > 0


# ---------------------------------------------------------------------------
# Coords IoU
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Shape-only (skip-tex) helper
# ---------------------------------------------------------------------------

def _run_shape_slat(pipeline, cond_512, cond_1024, coords, pipeline_type):
    """
    Run shape SLAT sampling and decode without any texture stages.
    Returns List[Mesh] (plain geometry, no PBR attributes).
    """
    if pipeline_type == "512":
        slat = pipeline.sample_shape_slat(
            cond_512, pipeline.models["shape_slat_flow_model_512"], coords)
        res = 512
    elif pipeline_type == "1024":
        slat = pipeline.sample_shape_slat(
            cond_1024, pipeline.models["shape_slat_flow_model_1024"], coords)
        res = 1024
    else:
        hr_res = 1024 if pipeline_type == "1024_cascade" else 1536
        slat, res = pipeline.sample_shape_slat_cascade(
            cond_512, cond_1024,
            pipeline.models["shape_slat_flow_model_512"],
            pipeline.models["shape_slat_flow_model_1024"],
            512, hr_res, coords,
        )
    torch.cuda.empty_cache()
    meshes, _ = pipeline.decode_shape_slat(slat, res)
    meshes[0].fill_holes()
    return meshes, res


# ---------------------------------------------------------------------------
# Per-condition runners
# ---------------------------------------------------------------------------

_SS_RES = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}


def run_baseline(pipeline, images, cond_dir, args) -> Dict:
    ss_res = _SS_RES[args.pipeline]
    torch.manual_seed(args.seed)

    if args.sparse_only:
        cond = pipeline.get_cond([images[0]], 512)
        coords = pipeline.sample_sparse_structure(cond, ss_res)
        return {"coords": coords, "model_path": None, "preview_path": None}

    cond_512 = pipeline.get_cond([images[0]], 512)
    cond_1024 = pipeline.get_cond([images[0]], 1024) if args.pipeline != "512" else None
    coords = pipeline.sample_sparse_structure(cond_512, ss_res)

    model_path = None
    preview_path = None

    if args.skip_tex:
        meshes, _ = _run_shape_slat(pipeline, cond_512, cond_1024, coords, args.pipeline)
        if meshes:
            model_path = save_obj(meshes[0], cond_dir)
            if not args.no_preview:
                preview_path = save_preview(meshes[0], cond_dir)
    else:
        meshes = pipeline.run(
            images[0], preprocess_image=False,
            pipeline_type=args.pipeline, seed=args.seed,
        )
        if meshes:
            res = ss_res if args.pipeline == "512" else 1024
            model_path = save_glb(pipeline, meshes[0], res, cond_dir,
                                   args.texture_size, args.decimation_target)
            if not args.no_preview:
                preview_path = save_preview(meshes[0], cond_dir)

    if model_path:
        print(f"  Model   : {model_path}")
    if preview_path:
        print(f"  Preview : {preview_path}")

    return {"coords": coords, "model_path": model_path, "preview_path": preview_path}


def run_p1(pipeline, images, fusion_mode, cond_dir, args) -> Dict:
    ss_res = _SS_RES[args.pipeline]
    torch.manual_seed(args.seed)

    if args.sparse_only:
        cond = pipeline.get_cond_multi(images, 512, fusion_mode)
        coords = pipeline.sample_sparse_structure(cond, ss_res)
        return {"coords": coords, "model_path": None, "preview_path": None}

    cond_512 = pipeline.get_cond_multi(images, 512, fusion_mode)
    cond_1024 = (pipeline.get_cond_multi(images, 1024, fusion_mode)
                 if args.pipeline != "512" else None)
    coords = pipeline.sample_sparse_structure(cond_512, ss_res)

    model_path = None
    preview_path = None

    if args.skip_tex:
        meshes, _ = _run_shape_slat(pipeline, cond_512, cond_1024, coords, args.pipeline)
        if meshes:
            model_path = save_obj(meshes[0], cond_dir)
            if not args.no_preview:
                preview_path = save_preview(meshes[0], cond_dir)
    else:
        meshes = pipeline.run_multi_image_cond(
            images, cond_fusion_mode=fusion_mode,
            preprocess_image=False, pipeline_type=args.pipeline, seed=args.seed,
        )
        if meshes:
            res = ss_res if args.pipeline == "512" else 1024
            model_path = save_glb(pipeline, meshes[0], res, cond_dir,
                                   args.texture_size, args.decimation_target)
            if not args.no_preview:
                preview_path = save_preview(meshes[0], cond_dir)

    if model_path:
        print(f"  Model   : {model_path}")
    if preview_path:
        print(f"  Preview : {preview_path}")

    return {"coords": coords, "model_path": model_path, "preview_path": preview_path}


def run_p2(pipeline, images, occupancy, cond_fusion_mode, cond_dir, args) -> Dict:
    ss_res = _SS_RES[args.pipeline]
    coords = occupancy_to_coords(occupancy, device=pipeline.device)

    if args.sparse_only:
        return {"coords": coords, "model_path": None, "preview_path": None}

    model_path = None
    preview_path = None

    if args.skip_tex:
        # Bypass run_scaffold_bypass and call stages directly to skip texture.
        primary_img = images[0]
        cond_512 = (pipeline.get_cond_multi(images, 512, "mean")
                    if cond_fusion_mode == "mean"
                    else pipeline.get_cond([primary_img], 512))
        cond_1024 = (pipeline.get_cond_multi(images, 1024, "mean")
                     if cond_fusion_mode == "mean" and args.pipeline != "512"
                     else (pipeline.get_cond([primary_img], 1024)
                           if args.pipeline != "512" else None))
        meshes, _ = _run_shape_slat(pipeline, cond_512, cond_1024, coords, args.pipeline)
        if meshes:
            model_path = save_obj(meshes[0], cond_dir)
            if not args.no_preview:
                preview_path = save_preview(meshes[0], cond_dir)
    else:
        result = run_scaffold_bypass(
            pipeline=pipeline,
            occupancy=occupancy,
            conditioning_images=images,
            cond_fusion_mode=cond_fusion_mode,
            pipeline_type=args.pipeline,
            seed=args.seed,
        )
        coords = result["coords"]
        if result["meshes"]:
            res = ss_res if args.pipeline == "512" else 1024
            model_path = save_glb(pipeline, result["meshes"][0], res, cond_dir,
                                   args.texture_size, args.decimation_target)
            if not args.no_preview:
                preview_path = save_preview(result["meshes"][0], cond_dir)

    if model_path:
        print(f"  Model   : {model_path}")
    if preview_path:
        print(f"  Preview : {preview_path}")

    return {"coords": coords, "model_path": model_path, "preview_path": preview_path}


# ---------------------------------------------------------------------------
# Condition dispatcher
# ---------------------------------------------------------------------------

def run_condition(
    name: str,
    images: List[Image.Image],
    args: argparse.Namespace,
    occupancy: Optional[np.ndarray],
    run_dir: str,
) -> Dict:
    """
    Load a fresh pipeline, run one condition, save outputs, unload the
    pipeline, and return a result dict.

    The pipeline is fully unloaded (weights deleted, CUDA cache flushed)
    before this function returns so the next condition starts from a clean
    GPU state.
    """
    result = {
        "name": name,
        "coords": None,
        "voxels": 0,
        "elapsed_s": 0.0,
        "model_path": None,
        "preview_path": None,
        "error": None,
    }
    t0 = time.time()
    pipeline = None

    cond_dir = os.path.join(run_dir, cond_folder_name(name))
    os.makedirs(cond_dir, exist_ok=True)

    try:
        print(f"  Loading pipeline for {name} ...")
        pipeline = load_pipeline(args.model)

        if name == "baseline":
            out = run_baseline(pipeline, images, cond_dir, args)
        elif name == "P1-mean":
            out = run_p1(pipeline, images, "mean", cond_dir, args)
        elif name == "P1-concat":
            out = run_p1(pipeline, images, "concat", cond_dir, args)
        elif name == "P2-scaffold":
            if occupancy is None:
                raise RuntimeError("No scaffold — pass --skip-scaffold to omit P2 conditions.")
            out = run_p2(pipeline, images, occupancy, "primary", cond_dir, args)
        elif name == "P2-scaffold+P1":
            if occupancy is None:
                raise RuntimeError("No scaffold — pass --skip-scaffold to omit P2 conditions.")
            out = run_p2(pipeline, images, occupancy, "mean", cond_dir, args)
        else:
            raise ValueError(f"Unknown condition: {name!r}")

        result["coords"] = out["coords"]
        result["voxels"] = int(out["coords"].shape[0]) if out["coords"] is not None else 0
        result["model_path"] = out["model_path"]
        result["preview_path"] = out["preview_path"]

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
        status = "ERROR" if r["error"] else ("model" if r["model_path"] else "sparse-only")
        print(f"{r['name']:<22} {r['voxels']:>8} {r['elapsed_s']:>8.1f}s  "
              f"{iou_str:>12}  {status}")
        if r["model_path"]:
            print(f"    model   → {r['model_path']}")
        if r["preview_path"]:
            print(f"    preview → {r['preview_path']}")
        if r["error"]:
            print(f"    error   : {r['error']}")
    print()


def save_json_summary(
    results: List[Dict],
    run_dir: str,
    run_meta: dict,
    baseline_coords: Optional[torch.Tensor],
) -> str:
    """
    Write a self-contained summary.json with full run parameters and
    per-condition metrics.  Structure::

        {
          "run": { ...parameters, scaffold stats, timing... },
          "conditions": [
            { "name", "status", "voxels", "iou_vs_baseline",
              "elapsed_s", "model_path", "preview_path", "error" },
            ...
          ]
        }
    """
    condition_rows = []
    for r in results:
        iou = None
        if (r["coords"] is not None and baseline_coords is not None
                and r["name"] != "baseline"):
            iou = round(coords_iou(r["coords"], baseline_coords), 4)
        status = "error" if r["error"] else ("ok" if r["model_path"] else "sparse-only")
        condition_rows.append({
            "name": r["name"],
            "status": status,
            "voxels": r["voxels"],
            "iou_vs_baseline": iou,
            "elapsed_s": round(r["elapsed_s"], 2),
            "model_path": r["model_path"],
            "preview_path": r["preview_path"],
            "error": r["error"],
        })

    doc = {
        "run": run_meta,
        "conditions": condition_rows,
    }
    path = os.path.join(run_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
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
                   help="Skip all model export; only count voxels.")
    p.add_argument("--skip-tex", action="store_true",
                   help="Skip texture SLAT stages; output geometry-only .obj. "
                        "Saves significant GPU time and VRAM.")
    p.add_argument("--no-preview", action="store_true",
                   help="Skip rendering preview PNGs.")
    p.add_argument("--pipeline", default="1024_cascade",
                   choices=["512", "1024", "1024_cascade", "1536_cascade"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    p.add_argument("--texture-size", type=int, default=1024)
    p.add_argument("--decimation-target", type=int, default=500_000)
    p.add_argument("--min-views-scaffold", type=int, default=None,
                   help="Min views a voxel must be inside the silhouette to "
                        "survive space carving (default: all views). "
                        "Try 2 or 3 to relax the hull when using >4 views.")
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
    print(f"  Skip-tex  : {args.skip_tex}")
    print(f"  Seed      : {args.seed}")
    print(f"  Output    : {args.output_dir}")

    # -----------------------------------------------------------------------
    # Step 1: preprocess images ONCE with a temporary pipeline, then unload.
    #
    # Silhouettes for space carving are extracted by calling rembg DIRECTLY
    # on the raw RGB images (extract_silhouette_rembg).  preprocess_image()
    # must NOT be used for silhouettes: it premultiplies alpha into RGB and
    # drops the alpha channel, so any subsequent RGBA conversion gives
    # alpha=255 everywhere → 100% scaffold density.
    # -----------------------------------------------------------------------
    print_section("Preprocessing images (temp pipeline)")
    t0 = time.time()
    tmp_pipeline = load_pipeline(args.model)

    n = len(args.images)
    raw_images = [Image.open(p).convert("RGB") for p in args.images]
    images = [tmp_pipeline.preprocess_image(img) for img in raw_images]
    print(f"  Preprocessed {n} image(s) in {time.time() - t0:.1f}s")

    if args.azimuths is not None:
        if len(args.azimuths) != n:
            raise ValueError(
                f"--azimuths has {len(args.azimuths)} values but {n} images provided."
            )
        azimuths = list(args.azimuths)
    else:
        azimuths = [360.0 * i / n for i in range(n)]
        print(f"  Auto azimuths: {[f'{a:.1f}°' for a in azimuths]}")

    occupancy = None
    scaffold_info: dict = {}
    if not args.skip_scaffold and any(
        c in conditions for c in ("P2-scaffold", "P2-scaffold+P1")
    ):
        print_section("Building visual-hull scaffold")
        scaffold_masks = [
            extract_silhouette_rembg(tmp_pipeline, raw_img) for raw_img in raw_images
        ]
        scaffold_rotations = [
            make_look_at_rotation(az, args.elevation) for az in azimuths
        ]
        for i, m in enumerate(scaffold_masks):
            print(f"  View {i}: {m.sum():,} foreground pixels "
                  f"({100 * m.mean():.1f}% of frame)")

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
                  "full. Check that azimuths match actual camera positions. "
                  "Try --min-views-scaffold with a lower value if using >4 views.")
        scaffold_info = {
            "grid_size": ss_res,
            "min_views": min_views,
            "occupied_voxels": n_occ,
            "total_voxels": int(occupancy.size),
            "density_pct": round(density, 2),
        }

    print_section("Unloading temp pipeline")
    unload_pipeline(tmp_pipeline)
    print("  GPU memory cleared.")

    # -----------------------------------------------------------------------
    # Step 2: create the run folder, then run each condition with its own
    # fresh pipeline.
    # -----------------------------------------------------------------------
    run_ts = run_timestamp()
    run_dir = os.path.join(args.output_dir, f"run_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n  Run dir: {run_dir}")

    wall_t0 = time.time()
    results = []
    baseline_coords = None

    for cond_name in conditions:
        print_section(f"Condition: {cond_name}")
        r = run_condition(
            name=cond_name,
            images=images,
            args=args,
            occupancy=occupancy,
            run_dir=run_dir,
        )
        results.append(r)
        print(f"  Voxels : {r['voxels']:,}")
        print(f"  Time   : {r['elapsed_s']:.1f}s")
        if r["error"]:
            print(f"  Error  : {r['error']}")
        if cond_name == "baseline" and r["coords"] is not None:
            baseline_coords = r["coords"].cpu()

    total_elapsed = round(time.time() - wall_t0, 2)

    # -----------------------------------------------------------------------
    # Step 3: build run metadata and write summary.
    # -----------------------------------------------------------------------
    run_meta = {
        "run_id": run_ts,
        "run_dir": os.path.abspath(run_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "total_elapsed_s": total_elapsed,
        # --- inputs ---
        "images": [os.path.abspath(p) for p in args.images],
        "azimuths_deg": azimuths,
        "elevation_deg": args.elevation,
        # --- pipeline config ---
        "model": args.model,
        "pipeline_type": args.pipeline,
        "seed": args.seed,
        # --- flags ---
        "skip_tex": args.skip_tex,
        "skip_scaffold": args.skip_scaffold,
        "sparse_only": args.sparse_only,
        "no_preview": args.no_preview,
        # --- export config ---
        "texture_size": args.texture_size,
        "decimation_target": args.decimation_target,
        # --- conditions ---
        "conditions_run": conditions,
        # --- scaffold (only present when P2 conditions were run) ---
        "scaffold": scaffold_info if scaffold_info else None,
    }

    print_summary(results, baseline_coords)
    json_path = save_json_summary(results, run_dir, run_meta, baseline_coords)
    print(f"Summary JSON : {json_path}")
    print(f"Run folder   : {run_dir}")
    print_section("Done")


if __name__ == "__main__":
    main()
