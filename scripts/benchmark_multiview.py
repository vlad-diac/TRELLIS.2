"""
Multi-View TRELLIS.2 Benchmark — Full Test Matrix

Runs the five conditions from the research plan on a fixed set of marine
(or other) vessel images and records voxel counts, wall-clock times, and
exports one model + preview per condition for visual comparison.

Memory model
------------
Image preprocessing uses only the BiRefNet background-removal model (~885 MB
GPU) which is loaded, used, and unloaded before any condition starts.  Each
benchmark condition then loads the full 4B TRELLIS pipeline (~22.6 GB GPU),
runs to completion, and fully unloads (weights deleted + CUDA cache flushed)
before the next condition starts.  This prevents GPU memory fragmentation and
ensures every condition starts from a clean state.

Stage flow (per condition)
--------------------------
[load]   TRELLIS.2-4B            ~18s   ~22.6 GiB VRAM
[1/5]    image conditioning       ~3s   cond_512, cond_1024
[2/5]    sparse structure        ~45s   N voxels  [skipped for P2 — hull used]
[3/5]    shape SLAT              ~92s
[4/5]    texture SLAT            ~88s   [skipped with --skip-tex]
[5/5]    decode + export         ~12s   model.glb / model.obj
         preview render           ~4s   preview.png
[unload] TRELLIS.2-4B                   VRAM freed

Output layout
-------------
Each invocation creates a timestamped run folder inside --output-dir::

    {output_dir}/
    └── run_YYYYMMDD_HHMMSS/
        ├── baseline/
        │   ├── model.glb       (model.obj with --skip-tex)
        │   └── preview.png
        ├── P1-mean/ ...
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

P1-concat
    Condition fusion — DINOv3 tokens concatenated across views (V×N tokens).

P2-scaffold
    External visual-hull bypass.  Space-carve silhouettes at assumed-orbit
    azimuths.  Feed coords directly to stage 3 (shape SLAT).

P2-scaffold+P1
    External visual-hull bypass + joint mean conditioning for stages 3–5.

Usage examples
--------------
# All 5 conditions on 4 views at default azimuths
python scripts/benchmark_multiview.py \\
    front.png right.png rear.png left.png \\
    --output-dir ./out

# Skip scaffold conditions (no visual hull needed)
python scripts/benchmark_multiview.py \\
    front.png side.png \\
    --skip-scaffold --output-dir ./out

# Geometry-only — skip texture SLAT (faster, lower VRAM)
python scripts/benchmark_multiview.py \\
    front.png right.png rear.png left.png \\
    --skip-tex --output-dir ./out

# Sparse-only — count voxels without exporting any model
python scripts/benchmark_multiview.py \\
    front.png right.png rear.png left.png \\
    --sparse-only --output-dir ./out

# Custom azimuths (3 views at 0°, 45°, 200°)
python scripts/benchmark_multiview.py \\
    img1.png img2.png img3.png \\
    --azimuths 0 45 200 --elevation 20 \\
    --output-dir ./out
"""

import argparse
import gc
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# #region agent log
import json as _json
_DBG_LOG = os.path.join(_repo_root, ".cursor", "debug-a41768.log")
def _dbg(loc, msg, data, hyp, run_id="pre-fix"):
    entry = {"sessionId": "a41768", "runId": run_id, "hypothesisId": hyp,
             "timestamp": int(time.time() * 1000), "location": loc,
             "message": msg, "data": data}
    with open(_DBG_LOG, "a") as _f:
        _f.write(_json.dumps(entry) + "\n")
# #endregion

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
)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

@contextmanager
def log_step(label: str, step: int = None, total: int = None):
    """
    Context manager for timed, structured stage logging.

    Prints ``[step/total] label ...`` on enter and ``done in Xs`` on exit.
    Yields a single-element list whose value is set to elapsed seconds on exit
    so callers can record it::

        with log_step("shape SLAT", 3, 5) as t:
            ...
        stage_times["shape_slat_s"] = round(t[0], 2)
    """
    prefix = f"[{step}/{total}] " if step is not None else ""
    print(f"\n  {prefix}{label} ...", flush=True)
    t0 = time.time()
    elapsed = [0.0]
    try:
        yield elapsed
    finally:
        elapsed[0] = time.time() - t0
        print(f"       └─ done in {elapsed[0]:.1f}s")


def gpu_mem_str() -> str:
    """Return a compact string with current GPU memory allocation."""
    if not torch.cuda.is_available():
        return "no CUDA"
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return f"{alloc:.1f} GiB allocated, {reserved:.1f} GiB reserved"


def print_banner(title: str) -> None:
    """Print a top-level section banner."""
    width = 64
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


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


def load_rembg_model(model_name: str = "briaai/RMBG-2.0"):
    """
    Load only the BiRefNet background-removal model (~885 MB GPU).

    Used for image preprocessing and silhouette extraction without loading the
    full 4B TRELLIS pipeline, avoiding the back-to-back double-load that
    triggers the OS OOM killer.
    """
    from trellis2.pipelines.rembg import BiRefNet
    model = BiRefNet(model_name=model_name)
    model.cuda()
    return model


def unload_rembg_model(model) -> None:
    """Delete the rembg model and flush GPU memory."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Preprocessing (rembg-only, no full pipeline)
# ---------------------------------------------------------------------------

def preprocess_image_standalone(raw_img: Image.Image, rembg_model) -> Image.Image:
    """
    Replicate pipeline.preprocess_image() using only the rembg model.

    Returns the same premultiplied-alpha RGB image the pipeline would produce,
    so conditioning tokens computed later are identical to what the pipeline
    would generate if it ran preprocess_image itself.
    """
    img = raw_img
    has_alpha = False
    if img.mode == "RGBA":
        alpha_ch = np.array(img)[:, :, 3]
        if not np.all(alpha_ch == 255):
            has_alpha = True
    max_size = max(img.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.Resampling.LANCZOS)
    output = img if has_alpha else rembg_model(img.convert("RGB"))
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox_pts = np.argwhere(alpha > 0.8 * 255)
    if len(bbox_pts) == 0:
        return img.convert("RGB")
    bbox = (int(np.min(bbox_pts[:, 1])), int(np.min(bbox_pts[:, 0])),
            int(np.max(bbox_pts[:, 1])), int(np.max(bbox_pts[:, 0])))
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    bbox = (center[0] - size // 2, center[1] - size // 2,
            center[0] + size // 2, center[1] + size // 2)
    output = output.crop(bbox)
    out_np = np.array(output).astype(np.float32) / 255
    out_np = out_np[:, :, :3] * out_np[:, :, 3:4]
    return Image.fromarray((out_np * 255).astype(np.uint8))


def extract_silhouette_rembg(rembg_model, raw_rgb: Image.Image) -> np.ndarray:
    """
    Run rembg on a raw RGB image and return a binary foreground mask.

    Must use the rembg model directly on raw images — NOT on preprocessed
    output.  preprocess_image_standalone() premultiplies alpha into RGB and
    drops the alpha channel; converting back to RGBA then gives alpha=255
    everywhere, producing a 100%-dense visual hull.
    """
    rgba = rembg_model(raw_rgb.convert("RGB"))
    return np.array(rgba)[:, :, 3] > 0


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def cond_folder_name(condition_name: str) -> str:
    return condition_name.replace("+", "_plus_")


def run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
    """Export a geometry-only Mesh as .obj via trimesh."""
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
    Render a 4-view snapshot and save as preview.png.

    For textured meshes (MeshWithVoxel / MeshWithPbrMaterial) a synthetic
    uniform-grey EnvMap is created on-the-fly so PbrMeshRenderer gets the
    mandatory envmap argument it requires.  For geometry-only Mesh objects
    MeshRenderer is used and no envmap is needed.

    Key preference: "shaded" (PBR) → "normal" (geometry) → first available.
    """
    from trellis2.utils import render_utils
    from trellis2.representations import MeshWithVoxel, MeshWithPbrMaterial
    try:
        render_kwargs = {}
        if isinstance(mesh, (MeshWithVoxel, MeshWithPbrMaterial)):
            from trellis2.renderers.pbr_mesh_renderer import EnvMap
            env_img = torch.ones(16, 32, 3, dtype=torch.float32,
                                 device=mesh.vertices.device) * 0.7
            render_kwargs["envmap"] = EnvMap(env_img)

        snapshot = render_utils.render_snapshot(
            mesh, resolution=resolution, r=2, fov=36, nviews=4,
            **render_kwargs,
        )
        frames = snapshot.get("shaded",
                 snapshot.get("normal",
                 next(iter(snapshot.values()))))
        strip = np.concatenate(frames[:4], axis=1)
        path = os.path.join(cond_dir, "preview.png")
        Image.fromarray(strip).save(path)
        return path
    except Exception as exc:
        print(f"       [WARN] preview rendering failed: {exc}")
        return None


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
# Shared stage runner (stages 3–5)
# ---------------------------------------------------------------------------

_SS_RES = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}


def _run_stages(
    pipeline,
    cond_512: dict,
    cond_1024: Optional[dict],
    coords: torch.Tensor,
    args: argparse.Namespace,
    cond_dir: str,
) -> Dict:
    """
    Run stages 3–5 for any condition type.

    Stage 3: shape SLAT (always)
    Stage 4: texture SLAT (skipped with --skip-tex)
    Stage 5: decode + export + optional preview render

    Returns {"model_path", "preview_path", "stage_times", "meshes"}.
    Callers merge their own stage_times (conditioning, sparse structure) with
    the returned stage_times dict.
    """
    stage_times: Dict = {}

    # ------------------------------------------------------------------
    # Stage 3: Shape SLAT
    # ------------------------------------------------------------------
    with log_step("shape SLAT", 3, 5) as t:
        if args.pipeline == "512":
            shape_slat = pipeline.sample_shape_slat(
                cond_512, pipeline.models["shape_slat_flow_model_512"], coords)
            res = 512
        elif args.pipeline == "1024":
            shape_slat = pipeline.sample_shape_slat(
                cond_1024, pipeline.models["shape_slat_flow_model_1024"], coords)
            res = 1024
        else:
            hr_res = 1536 if args.pipeline == "1536_cascade" else 1024
            shape_slat, res = pipeline.sample_shape_slat_cascade(
                cond_512, cond_1024,
                pipeline.models["shape_slat_flow_model_512"],
                pipeline.models["shape_slat_flow_model_1024"],
                512, hr_res, coords,
            )
    stage_times["shape_slat_s"] = round(t[0], 2)
    torch.cuda.empty_cache()

    model_path = preview_path = None
    meshes = []

    # ------------------------------------------------------------------
    # Stage 4 + 5
    # ------------------------------------------------------------------
    if args.skip_tex:
        print(f"\n  [4/5] texture SLAT ... skipped (--skip-tex)")
        stage_times["tex_slat_s"] = None

        with log_step("decode + export (geometry-only)", 5, 5) as t:
            meshes, _ = pipeline.decode_shape_slat(shape_slat, res)
            meshes[0].fill_holes()
            model_path = save_obj(meshes[0], cond_dir)
        stage_times["decode_s"] = round(t[0], 2)
        del shape_slat

    else:
        cond_tex = cond_1024 if cond_1024 is not None else cond_512
        flow_tex = (pipeline.models["tex_slat_flow_model_512"]
                    if args.pipeline == "512"
                    else pipeline.models["tex_slat_flow_model_1024"])

        with log_step("texture SLAT", 4, 5) as t:
            tex_slat = pipeline.sample_tex_slat(cond_tex, flow_tex, shape_slat)
        stage_times["tex_slat_s"] = round(t[0], 2)
        torch.cuda.empty_cache()

        # #region agent log
        _dbg("_run_stages:before_decode", "GPU mem before decode_latent",
             {"alloc_gib": round(torch.cuda.memory_allocated()/1e9, 3),
              "reserved_gib": round(torch.cuda.memory_reserved()/1e9, 3),
              "free_gib": round((torch.cuda.get_device_properties(0).total_memory
                                 - torch.cuda.memory_reserved())/1e9, 3),
              "cond_512_alive": cond_512 is not None,
              "cond_1024_alive": cond_1024 is not None}, "H-A,H-C")
        # #endregion

        with log_step("decode + export (textured)", 5, 5) as t:
            meshes = pipeline.decode_latent(shape_slat, tex_slat, res)

            # #region agent log
            _dbg("_run_stages:after_decode_before_glb", "GPU mem after decode_latent, before save_glb",
                 {"alloc_gib": round(torch.cuda.memory_allocated()/1e9, 3),
                  "reserved_gib": round(torch.cuda.memory_reserved()/1e9, 3),
                  "free_gib": round((torch.cuda.get_device_properties(0).total_memory
                                     - torch.cuda.memory_reserved())/1e9, 3),
                  "shape_slat_alive": shape_slat is not None,
                  "tex_slat_alive": tex_slat is not None,
                  "cond_512_alive": cond_512 is not None,
                  "cond_1024_alive": cond_1024 is not None}, "H-B,H-E")
            # #endregion

            model_path = save_glb(
                pipeline, meshes[0], res, cond_dir,
                args.texture_size, args.decimation_target,
            )
        stage_times["decode_s"] = round(t[0], 2)
        del shape_slat, tex_slat

        # #region agent log
        _dbg("_run_stages:after_del_slats", "GPU mem after del shape_slat+tex_slat",
             {"alloc_gib": round(torch.cuda.memory_allocated()/1e9, 3),
              "reserved_gib": round(torch.cuda.memory_reserved()/1e9, 3),
              "free_gib": round((torch.cuda.get_device_properties(0).total_memory
                                 - torch.cuda.memory_reserved())/1e9, 3)}, "H-B")
        # #endregion

    # Free conditioning + SLAT tensors before the preview renderer allocates
    # its buffers.  The pipeline weights (~22.6 GiB) stay loaded; we only
    # need to reclaim the working tensors (~300–600 MiB) so nvdiffrast can
    # render.
    del cond_512, cond_1024
    torch.cuda.empty_cache()

    # #region agent log
    _dbg("_run_stages:after_del_conds", "GPU mem after del cond_512+cond_1024 + empty_cache",
         {"alloc_gib": round(torch.cuda.memory_allocated()/1e9, 3),
          "reserved_gib": round(torch.cuda.memory_reserved()/1e9, 3),
          "free_gib": round((torch.cuda.get_device_properties(0).total_memory
                             - torch.cuda.memory_reserved())/1e9, 3)}, "H-A,H-D")
    # #endregion

    if model_path:
        print(f"       → {model_path}")

    # ------------------------------------------------------------------
    # Preview render (optional)
    # ------------------------------------------------------------------
    stage_times["preview_s"] = None
    if not args.no_preview and meshes:
        with log_step("preview render") as t:
            preview_path = save_preview(meshes[0], cond_dir)
        stage_times["preview_s"] = round(t[0], 2)
        if preview_path:
            print(f"       → {preview_path}")

    return {
        "model_path": model_path,
        "preview_path": preview_path,
        "stage_times": stage_times,
        "meshes": meshes,
    }


# ---------------------------------------------------------------------------
# Per-condition runners (stages 1–2 only; delegate 3–5 to _run_stages)
# ---------------------------------------------------------------------------

def run_baseline(pipeline, images: List[Image.Image], cond_dir: str,
                 args: argparse.Namespace) -> Dict:
    ss_res = _SS_RES[args.pipeline]
    stage_times: Dict = {}
    torch.manual_seed(args.seed)

    # Stage 1: image conditioning
    with log_step("image conditioning (single image)", 1, 5) as t:
        cond_512 = pipeline.get_cond([images[0]], 512)
        cond_1024 = (pipeline.get_cond([images[0]], 1024)
                     if args.pipeline != "512" else None)
    stage_times["conditioning_s"] = round(t[0], 2)

    # Stage 2: sparse structure
    with log_step("sparse structure", 2, 5) as t:
        coords = pipeline.sample_sparse_structure(cond_512, ss_res)
    stage_times["sparse_structure_s"] = round(t[0], 2)
    print(f"       {coords.shape[0]:,} voxels")

    if args.sparse_only:
        stage_times.update({"shape_slat_s": None, "tex_slat_s": None,
                            "decode_s": None, "preview_s": None})
        return {"coords": coords, "model_path": None, "preview_path": None,
                "stage_times": stage_times}

    # Stages 3–5
    out = _run_stages(pipeline, cond_512, cond_1024, coords, args, cond_dir)
    out["stage_times"] = {**stage_times, **out["stage_times"]}
    out["coords"] = coords
    return out


def run_p1(pipeline, images: List[Image.Image], fusion_mode: str,
           cond_dir: str, args: argparse.Namespace) -> Dict:
    ss_res = _SS_RES[args.pipeline]
    stage_times: Dict = {}
    torch.manual_seed(args.seed)

    # Stage 1: multi-view condition fusion
    with log_step(f"image conditioning (multi/{fusion_mode})", 1, 5) as t:
        cond_512 = pipeline.get_cond_multi(images, 512, fusion_mode)
        cond_1024 = (pipeline.get_cond_multi(images, 1024, fusion_mode)
                     if args.pipeline != "512" else None)
    stage_times["conditioning_s"] = round(t[0], 2)

    # Stage 2: sparse structure
    with log_step("sparse structure", 2, 5) as t:
        coords = pipeline.sample_sparse_structure(cond_512, ss_res)
    stage_times["sparse_structure_s"] = round(t[0], 2)
    print(f"       {coords.shape[0]:,} voxels")

    if args.sparse_only:
        stage_times.update({"shape_slat_s": None, "tex_slat_s": None,
                            "decode_s": None, "preview_s": None})
        return {"coords": coords, "model_path": None, "preview_path": None,
                "stage_times": stage_times}

    # Stages 3–5
    out = _run_stages(pipeline, cond_512, cond_1024, coords, args, cond_dir)
    out["stage_times"] = {**stage_times, **out["stage_times"]}
    out["coords"] = coords
    return out


def run_p2(pipeline, images: List[Image.Image], occupancy: np.ndarray,
           cond_fusion_mode: str, cond_dir: str,
           args: argparse.Namespace) -> Dict:
    stage_times: Dict = {}
    torch.manual_seed(args.seed)

    # Stage 2 is skipped — coords come from the visual hull
    coords = occupancy_to_coords(occupancy, device=pipeline.device)
    print(f"\n  [2/5] sparse structure ... skipped (hull: {coords.shape[0]:,} voxels)")
    stage_times["sparse_structure_s"] = None

    if args.sparse_only:
        stage_times.update({"conditioning_s": None, "shape_slat_s": None,
                            "tex_slat_s": None, "decode_s": None, "preview_s": None})
        return {"coords": coords, "model_path": None, "preview_path": None,
                "stage_times": stage_times}

    # Stage 1: conditioning (needed for stages 3–5 even in P2)
    primary = images[0]
    with log_step(f"image conditioning ({cond_fusion_mode})", 1, 5) as t:
        if cond_fusion_mode == "mean":
            cond_512 = pipeline.get_cond_multi(images, 512, "mean")
            cond_1024 = (pipeline.get_cond_multi(images, 1024, "mean")
                         if args.pipeline != "512" else None)
        else:
            cond_512 = pipeline.get_cond([primary], 512)
            cond_1024 = (pipeline.get_cond([primary], 1024)
                         if args.pipeline != "512" else None)
    stage_times["conditioning_s"] = round(t[0], 2)

    # Stages 3–5
    out = _run_stages(pipeline, cond_512, cond_1024, coords, args, cond_dir)
    out["stage_times"] = {**stage_times, **out["stage_times"]}
    out["coords"] = coords
    return out


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
    Load the 4B pipeline, run one condition through all explicit stages,
    save outputs, unload the pipeline, and return a result dict.

    The pipeline is fully unloaded before this function returns so the next
    condition starts from a clean GPU state.
    """
    result: Dict = {
        "name": name,
        "coords": None,
        "voxels": 0,
        "elapsed_s": 0.0,
        "model_path": None,
        "preview_path": None,
        "stage_times": {},
        "error": None,
    }
    t0 = time.time()
    pipeline = None

    cond_dir = os.path.join(run_dir, cond_folder_name(name))
    os.makedirs(cond_dir, exist_ok=True)

    try:
        with log_step(f"[LOAD] {args.model}") as t:
            pipeline = load_pipeline(args.model)
        print(f"       {gpu_mem_str()}")
        result["stage_times"]["load_s"] = round(t[0], 2)

        if name == "baseline":
            out = run_baseline(pipeline, images, cond_dir, args)
        elif name == "P1-mean":
            out = run_p1(pipeline, images, "mean", cond_dir, args)
        elif name == "P1-concat":
            out = run_p1(pipeline, images, "concat", cond_dir, args)
        elif name == "P2-scaffold":
            if occupancy is None:
                raise RuntimeError(
                    "No scaffold — pass --skip-scaffold to omit P2 conditions.")
            out = run_p2(pipeline, images, occupancy, "primary", cond_dir, args)
        elif name == "P2-scaffold+P1":
            if occupancy is None:
                raise RuntimeError(
                    "No scaffold — pass --skip-scaffold to omit P2 conditions.")
            out = run_p2(pipeline, images, occupancy, "mean", cond_dir, args)
        else:
            raise ValueError(f"Unknown condition: {name!r}")

        result["coords"] = out["coords"]
        result["voxels"] = (int(out["coords"].shape[0])
                            if out["coords"] is not None else 0)
        result["model_path"] = out["model_path"]
        result["preview_path"] = out["preview_path"]
        result["stage_times"].update(out.get("stage_times", {}))

    except Exception as exc:
        result["error"] = str(exc)
        print(f"\n  [ERROR] {exc}")

    finally:
        if pipeline is not None:
            with log_step(f"[UNLOAD] {args.model}") as t:
                unload_pipeline(pipeline)
            print(f"       GPU memory cleared")
            result["stage_times"]["unload_s"] = round(t[0], 2)

    result["elapsed_s"] = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_stage_times(st: Dict) -> str:
    """Format stage_times dict as a compact one-liner for print_summary."""
    keys = [
        ("load_s",             "load"),
        ("conditioning_s",     "cond"),
        ("sparse_structure_s", "sparse"),
        ("shape_slat_s",       "shape"),
        ("tex_slat_s",         "tex"),
        ("decode_s",           "decode"),
        ("preview_s",          "preview"),
    ]
    parts = []
    for k, label in keys:
        v = st.get(k)
        if v is not None:
            parts.append(f"{label}={v:.1f}s")
        elif k in st:
            parts.append(f"{label}=skip")
    return "  " + "  ".join(parts) if parts else ""


def print_summary(results: List[Dict],
                  baseline_coords: Optional[torch.Tensor]) -> None:
    print_banner("Benchmark Summary")
    hdr = f"{'Condition':<22} {'Voxels':>8} {'Elapsed':>9}  {'IoU vs base':>12}  Status"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        iou_str = "—"
        if (r["coords"] is not None and baseline_coords is not None
                and r["name"] != "baseline"):
            iou = coords_iou(r["coords"], baseline_coords)
            iou_str = f"{iou:.3f}"
        status = "ERROR" if r["error"] else ("model" if r["model_path"] else "sparse-only")
        print(f"{r['name']:<22} {r['voxels']:>8} {r['elapsed_s']:>8.1f}s  "
              f"{iou_str:>12}  {status}")
        if r["stage_times"]:
            print(_fmt_stage_times(r["stage_times"]))
        if r["model_path"]:
            print(f"  model   → {r['model_path']}")
        if r["preview_path"]:
            print(f"  preview → {r['preview_path']}")
        if r["error"]:
            print(f"  error   : {r['error']}")
    print()


def save_json_summary(
    results: List[Dict],
    run_dir: str,
    run_meta: dict,
    baseline_coords: Optional[torch.Tensor],
) -> str:
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
            "stage_times": r.get("stage_times", {}),
            "model_path": r["model_path"],
            "preview_path": r["preview_path"],
            "error": r["error"],
        })

    doc = {"run": run_meta, "conditions": condition_rows}
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
                   help="Camera azimuth per image in degrees "
                        "(default: evenly spaced 360°).")
    p.add_argument("--elevation", type=float, default=15.0)
    p.add_argument("--skip-scaffold", action="store_true",
                   help="Skip P2-scaffold and P2-scaffold+P1.")
    p.add_argument("--sparse-only", action="store_true",
                   help="Stop after stage 2; count voxels only, no model export.")
    p.add_argument("--skip-tex", action="store_true",
                   help="Skip stage 4 (texture SLAT); output geometry-only .obj.")
    p.add_argument("--no-preview", action="store_true",
                   help="Skip preview PNG rendering.")
    p.add_argument("--pipeline", default="1024_cascade",
                   choices=["512", "1024", "1024_cascade", "1536_cascade"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    p.add_argument("--rembg-model", default="briaai/RMBG-2.0",
                   help="HuggingFace model ID for background removal "
                        "(default: briaai/RMBG-2.0).")
    p.add_argument("--texture-size", type=int, default=1024)
    p.add_argument("--decimation-target", type=int, default=500_000)
    p.add_argument("--min-views-scaffold", type=int, default=None,
                   help="Min views a voxel must be inside the silhouette to "
                        "survive space carving (default: all views). "
                        "Try 2 or 3 to relax the hull when using >4 views.")
    p.add_argument("--conditions", nargs="+",
                   choices=["baseline", "P1-mean", "P1-concat",
                            "P2-scaffold", "P2-scaffold+P1"],
                   default=None,
                   help="Run only specific conditions (default: all).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    print_banner("TRELLIS.2 Multi-View Benchmark")
    print(f"  Model      : {args.model}")
    print(f"  Pipeline   : {args.pipeline}")
    print(f"  Images     : {args.images}")
    print(f"  Conditions : {conditions}")
    print(f"  Seed       : {args.seed}")
    print(f"  Skip-tex   : {args.skip_tex}")
    print(f"  Output     : {args.output_dir}")

    # -----------------------------------------------------------------------
    # Phase 1: Image preprocessing (BiRefNet only, ~885 MB GPU)
    #
    # The full 4B model is NOT loaded here.  BiRefNet is loaded, used, and
    # fully unloaded before any condition pipeline is loaded, so GPU memory
    # is clean when the first condition starts.
    # -----------------------------------------------------------------------
    print_banner("Phase 1 — Preprocessing")

    with log_step(f"[LOAD] BiRefNet ({args.rembg_model})") as t:
        rembg_model = load_rembg_model(args.rembg_model)
    print(f"       {gpu_mem_str()}")

    n = len(args.images)
    raw_images = [Image.open(p).convert("RGB") for p in args.images]

    with log_step(f"preprocess {n} image(s)") as t:
        images = [preprocess_image_standalone(img, rembg_model)
                  for img in raw_images]
    preprocess_elapsed = round(t[0], 2)
    print(f"       {n} image(s) ready")

    if args.azimuths is not None:
        if len(args.azimuths) != n:
            raise ValueError(
                f"--azimuths has {len(args.azimuths)} values but {n} images provided.")
        azimuths = list(args.azimuths)
    else:
        azimuths = [360.0 * i / n for i in range(n)]
        print(f"       auto azimuths: {[f'{a:.1f}°' for a in azimuths]}")

    # -----------------------------------------------------------------------
    # Phase 1b: Visual hull scaffold (still with BiRefNet loaded)
    # -----------------------------------------------------------------------
    occupancy = None
    scaffold_info: dict = {}

    if not args.skip_scaffold and any(
        c in conditions for c in ("P2-scaffold", "P2-scaffold+P1")
    ):
        print_banner("Phase 1b — Visual Hull Scaffold")
        with log_step("extract silhouettes") as t:
            scaffold_masks = [
                extract_silhouette_rembg(rembg_model, raw_img)
                for raw_img in raw_images
            ]
        for i, m in enumerate(scaffold_masks):
            print(f"       view {i}: {m.sum():,} foreground px "
                  f"({100 * m.mean():.1f}% of frame)")

        scaffold_rotations = [
            make_look_at_rotation(az, args.elevation) for az in azimuths
        ]
        ss_res = {"512": 32, "1024": 64,
                  "1024_cascade": 32, "1536_cascade": 32}[args.pipeline]
        min_views = (args.min_views_scaffold
                     if args.min_views_scaffold is not None
                     else len(scaffold_masks))

        with log_step(f"space carve (grid={ss_res}³, min_views={min_views})") as t:
            occupancy = space_carve(
                scaffold_masks, scaffold_rotations,
                grid_size=ss_res, min_views=min_views,
            )
        n_occ = int(occupancy.sum())
        density = 100 * n_occ / occupancy.size
        print(f"       {n_occ:,} / {occupancy.size:,} voxels  ({density:.1f}% density)")
        if density > 80:
            print("       [WARN] density >80% — hull may be nearly full. "
                  "Check azimuths or try --min-views-scaffold with a lower value.")
        scaffold_info = {
            "grid_size": ss_res,
            "min_views": min_views,
            "occupied_voxels": n_occ,
            "total_voxels": int(occupancy.size),
            "density_pct": round(density, 2),
        }

    with log_step(f"[UNLOAD] BiRefNet") as t:
        unload_rembg_model(rembg_model)
    print(f"       GPU memory cleared")

    # -----------------------------------------------------------------------
    # Phase 2: Run each condition
    # -----------------------------------------------------------------------
    print_banner("Phase 2 — Benchmark Conditions")

    run_ts = run_timestamp()
    run_dir = os.path.join(args.output_dir, f"run_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"  Run dir: {run_dir}\n")

    wall_t0 = time.time()
    results = []
    baseline_coords = None

    for cond_name in conditions:
        print_banner(f"Condition: {cond_name}")
        r = run_condition(
            name=cond_name,
            images=images,
            args=args,
            occupancy=occupancy,
            run_dir=run_dir,
        )
        results.append(r)
        print(f"\n  Voxels  : {r['voxels']:,}")
        print(f"  Total   : {r['elapsed_s']:.1f}s")
        if r["error"]:
            print(f"  Error   : {r['error']}")
        if cond_name == "baseline" and r["coords"] is not None:
            baseline_coords = r["coords"].cpu()

    total_elapsed = round(time.time() - wall_t0, 2)

    # -----------------------------------------------------------------------
    # Phase 3: Summary
    # -----------------------------------------------------------------------
    print_summary(results, baseline_coords)

    run_meta = {
        "run_id": run_ts,
        "run_dir": os.path.abspath(run_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "total_elapsed_s": total_elapsed,
        "preprocess_elapsed_s": preprocess_elapsed,
        "images": [os.path.abspath(p) for p in args.images],
        "azimuths_deg": azimuths,
        "elevation_deg": args.elevation,
        "model": args.model,
        "rembg_model": args.rembg_model,
        "pipeline_type": args.pipeline,
        "seed": args.seed,
        "skip_tex": args.skip_tex,
        "skip_scaffold": args.skip_scaffold,
        "sparse_only": args.sparse_only,
        "no_preview": args.no_preview,
        "texture_size": args.texture_size,
        "decimation_target": args.decimation_target,
        "conditions_run": conditions,
        "scaffold": scaffold_info if scaffold_info else None,
    }

    json_path = save_json_summary(results, run_dir, run_meta, baseline_coords)
    print(f"Summary JSON : {json_path}")
    print(f"Run folder   : {run_dir}")
    print_banner("Done")


if __name__ == "__main__":
    main()
