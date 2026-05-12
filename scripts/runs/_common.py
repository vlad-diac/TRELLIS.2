"""
Shared helpers for TRELLIS.2 multi-view run scripts (scripts/runs/*.py).

Sets repo + scripts on sys.path when imported so scaffold_bypass can be found.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Bootstrap paths (runs/ → repo root, scripts/)
# ---------------------------------------------------------------------------
_RUNS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_RUNS_DIR))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
for p in (_REPO_ROOT, _RUNS_DIR, _SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

@contextmanager
def log_step(label: str, step: Optional[int] = None, total: Optional[int] = None):
    prefix = f"[{step}/{total}] " if step is not None else ""
    print(f"\n  {prefix}{label} ...", flush=True)
    t0 = datetime.now().timestamp()
    elapsed = [0.0]
    try:
        yield elapsed
    finally:
        elapsed[0] = datetime.now().timestamp() - t0
        print(f"       └─ done in {elapsed[0]:.1f}s")


def gpu_mem_str() -> str:
    if not torch.cuda.is_available():
        return "no CUDA"
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return f"{alloc:.1f} GiB allocated, {reserved:.1f} GiB reserved"


def print_banner(title: str) -> None:
    width = 64
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------

def load_pipeline(model: str):
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pl = Trellis2ImageTo3DPipeline.from_pretrained(model)
    pl.cuda()
    return pl


def unload_pipeline(pipeline) -> None:
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def load_rembg_model(model_name: str = "briaai/RMBG-2.0"):
    from trellis2.pipelines.rembg import BiRefNet
    model = BiRefNet(model_name=model_name)
    model.cuda()
    return model


def unload_rembg_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Preprocessing (rembg-only)
# ---------------------------------------------------------------------------

def preprocess_image_standalone(raw_img: Image.Image, rembg_model) -> Image.Image:
    img = raw_img
    has_alpha = False
    if img.mode == "RGBA":
        alpha_ch = np.array(img)[:, :, 3]
        if not np.all(alpha_ch == 255):
            has_alpha = True
    max_size = max(img.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.Resampling.LANCZOS,
        )
    output = img if has_alpha else rembg_model(img.convert("RGB"))
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox_pts = np.argwhere(alpha > 0.8 * 255)
    if len(bbox_pts) == 0:
        return img.convert("RGB")
    bbox = (
        int(np.min(bbox_pts[:, 1])),
        int(np.min(bbox_pts[:, 0])),
        int(np.max(bbox_pts[:, 1])),
        int(np.max(bbox_pts[:, 0])),
    )
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    bbox = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    output = output.crop(bbox)
    out_np = np.array(output).astype(np.float32) / 255
    out_np = out_np[:, :, :3] * out_np[:, :, 3:4]
    return Image.fromarray((out_np * 255).astype(np.uint8))


def extract_silhouette_rembg(rembg_model, raw_rgb: Image.Image) -> np.ndarray:
    rgba = rembg_model(raw_rgb.convert("RGB"))
    return np.array(rgba)[:, :, 3] > 0


# ---------------------------------------------------------------------------
# Run directory + summary
# ---------------------------------------------------------------------------

def run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(output_dir: str, strategy: str) -> str:
    path = os.path.join(output_dir, f"{strategy}_{run_timestamp()}")
    os.makedirs(path, exist_ok=True)
    return path


def resolve_run_dir(output_dir: str, strategy: str, dest_dir: Optional[str]) -> str:
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
        return os.path.abspath(dest_dir)
    return os.path.abspath(make_run_dir(output_dir, strategy))


def write_summary(run_dir: str, payload: dict) -> str:
    path = os.path.join(run_dir, "summary.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return path


def copy_inputs_to_run_dir(run_dir: str, image_paths: List[str]) -> None:
    inp = os.path.join(run_dir, "inputs")
    os.makedirs(inp, exist_ok=True)
    for p in image_paths:
        if os.path.isfile(p):
            base = os.path.basename(p)
            dst = os.path.join(inp, base)
            if os.path.abspath(p) != os.path.abspath(dst):
                shutil.copy2(p, dst)


def discover_inputs(input_root: str) -> Dict[str, Any]:
    """For Gradio: list loose image files and per-folder image lists under input_root."""
    files: List[str] = []
    folders: Dict[str, List[str]] = {}
    if not os.path.isdir(input_root):
        return {"files": files, "folders": folders}
    for name in sorted(os.listdir(input_root)):
        path = os.path.join(input_root, name)
        if os.path.isdir(path):
            imgs = sorted(
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.splitext(f)[1] in IMAGE_EXTENSIONS
                and os.path.isfile(os.path.join(path, f))
            )
            if imgs:
                folders[name] = imgs
        elif os.path.isfile(path) and os.path.splitext(name)[1] in IMAGE_EXTENSIONS:
            files.append(path)
    return {"files": files, "folders": folders}


def cond_folder_name(condition_name: str) -> str:
    return condition_name.replace("+", "_plus_")


# ---------------------------------------------------------------------------
# Export mesh / preview
# ---------------------------------------------------------------------------

def save_glb(
    pipeline,
    mesh,
    res: int,
    out_dir: str,
    texture_size: int,
    decimation_target: int,
) -> str:
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
    path = os.path.join(out_dir, "model.glb")
    glb.export(path, extension_webp=True)
    return path


def save_obj(mesh, out_dir: str) -> str:
    import trimesh
    tm = trimesh.Trimesh(
        vertices=mesh.vertices.cpu().float().numpy(),
        faces=mesh.faces.cpu().int().numpy(),
    )
    path = os.path.join(out_dir, "model.obj")
    tm.export(path)
    return path


def save_geometry_glb(mesh, out_dir: str) -> str:
    import trimesh
    tm = trimesh.Trimesh(
        vertices=mesh.vertices.cpu().float().numpy(),
        faces=mesh.faces.cpu().int().numpy(),
    )
    path = os.path.join(out_dir, "model.glb")
    tm.export(path)
    return path


def save_preview(mesh, out_dir: str, resolution: int = 512) -> Optional[str]:
    from trellis2.utils import render_utils
    from trellis2.representations import MeshWithVoxel, MeshWithPbrMaterial
    try:
        render_kwargs = {}
        if isinstance(mesh, (MeshWithVoxel, MeshWithPbrMaterial)):
            from trellis2.renderers.pbr_mesh_renderer import EnvMap
            env_img = torch.ones(
                16, 32, 3, dtype=torch.float32, device=mesh.vertices.device
            ) * 0.7
            render_kwargs["envmap"] = EnvMap(env_img)
        snapshot = render_utils.render_snapshot(
            mesh, resolution=resolution, r=2, fov=36, nviews=4, **render_kwargs
        )
        frames = snapshot.get(
            "shaded", snapshot.get("normal", next(iter(snapshot.values())))
        )
        strip = np.concatenate(frames[:4], axis=1)
        path = os.path.join(out_dir, "preview.png")
        Image.fromarray(strip).save(path)
        return path
    except Exception as exc:
        print(f"       [WARN] preview rendering failed: {exc}")
        return None


def coords_iou(a: torch.Tensor, b: torch.Tensor, grid_size: int = 64) -> float:
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
# Shared stage runner (benchmark-style P1/P2/baseline) stages 3–5
# ---------------------------------------------------------------------------

_SS_RES = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}


def sparse_grid_resolution(pipeline_mode: str) -> int:
    return _SS_RES[pipeline_mode]


def run_stages_3_to_5(
    pipeline,
    cond_512: dict,
    cond_1024: Optional[dict],
    coords: torch.Tensor,
    args: argparse.Namespace,
    run_dir: str,
) -> Dict[str, Any]:
    stage_times: Dict[str, Any] = {}

    with log_step("shape SLAT", 3, 5) as t:
        if args.pipeline == "512":
            shape_slat = pipeline.sample_shape_slat(
                cond_512, pipeline.models["shape_slat_flow_model_512"], coords
            )
            res = 512
        elif args.pipeline == "1024":
            shape_slat = pipeline.sample_shape_slat(
                cond_1024, pipeline.models["shape_slat_flow_model_1024"], coords
            )
            res = 1024
        else:
            hr_res = 1536 if args.pipeline == "1536_cascade" else 1024
            shape_slat, res = pipeline.sample_shape_slat_cascade(
                cond_512,
                cond_1024,
                pipeline.models["shape_slat_flow_model_512"],
                pipeline.models["shape_slat_flow_model_1024"],
                512,
                hr_res,
                coords,
            )
    stage_times["shape_slat_s"] = round(t[0], 2)
    torch.cuda.empty_cache()

    model_path = preview_path = None
    meshes = []

    if args.skip_tex or args.obj:
        flag = "--obj" if args.obj else "--skip-tex"
        print(f"\n  [4/5] texture SLAT ... skipped ({flag})")
        stage_times["tex_slat_s"] = None
        del cond_512, cond_1024
        torch.cuda.empty_cache()
        with log_step("decode + export (geometry-only)", 5, 5) as t:
            meshes, _ = pipeline.decode_shape_slat(shape_slat, res)
            del shape_slat
            torch.cuda.empty_cache()
            meshes[0].fill_holes()
            if args.obj:
                model_path = save_obj(meshes[0], run_dir)
            else:
                model_path = save_geometry_glb(meshes[0], run_dir)
        stage_times["decode_s"] = round(t[0], 2)
    else:
        cond_tex = cond_1024 if cond_1024 is not None else cond_512
        flow_tex = (
            pipeline.models["tex_slat_flow_model_512"]
            if args.pipeline == "512"
            else pipeline.models["tex_slat_flow_model_1024"]
        )
        with log_step("texture SLAT", 4, 5) as t:
            tex_slat = pipeline.sample_tex_slat(cond_tex, flow_tex, shape_slat)
        stage_times["tex_slat_s"] = round(t[0], 2)
        del cond_512, cond_1024
        torch.cuda.empty_cache()
        with log_step("decode + export (textured)", 5, 5) as t:
            meshes = pipeline.decode_latent(shape_slat, tex_slat, res)
            del shape_slat, tex_slat
            torch.cuda.empty_cache()
            model_path = save_glb(
                pipeline,
                meshes[0],
                res,
                run_dir,
                args.texture_size,
                args.decimation_target,
            )
        stage_times["decode_s"] = round(t[0], 2)

    if model_path:
        print(f"       → {model_path}")

    stage_times["preview_s"] = None
    if not args.no_preview and meshes:
        with log_step("preview render") as t:
            preview_path = save_preview(meshes[0], run_dir)
        stage_times["preview_s"] = round(t[0], 2)
        if preview_path:
            print(f"       → {preview_path}")

    return {
        "model_path": model_path,
        "preview_path": preview_path,
        "stage_times": stage_times,
        "meshes": meshes,
    }


def run_baseline_core(
    pipeline, images: List[Image.Image], run_dir: str, args: argparse.Namespace
) -> Dict[str, Any]:
    ss_res = _SS_RES[args.pipeline]
    stage_times: Dict[str, Any] = {}
    torch.manual_seed(args.seed)
    with log_step("image conditioning (single image)", 1, 5) as t:
        cond_512 = pipeline.get_cond([images[0]], 512)
        cond_1024 = (
            pipeline.get_cond([images[0]], 1024) if args.pipeline != "512" else None
        )
    stage_times["conditioning_s"] = round(t[0], 2)
    with log_step("sparse structure", 2, 5) as t:
        coords = pipeline.sample_sparse_structure(cond_512, ss_res)
    stage_times["sparse_structure_s"] = round(t[0], 2)
    print(f"       {coords.shape[0]:,} voxels")
    if args.sparse_only:
        stage_times.update(
            {
                "shape_slat_s": None,
                "tex_slat_s": None,
                "decode_s": None,
                "preview_s": None,
            }
        )
        return {
            "coords": coords,
            "model_path": None,
            "preview_path": None,
            "stage_times": stage_times,
        }
    out = run_stages_3_to_5(pipeline, cond_512, cond_1024, coords, args, run_dir)
    out["stage_times"] = {**stage_times, **out["stage_times"]}
    out["coords"] = coords
    return out


def run_p1_core(
    pipeline,
    images: List[Image.Image],
    fusion_mode: str,
    run_dir: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    ss_res = _SS_RES[args.pipeline]
    stage_times: Dict[str, Any] = {}
    torch.manual_seed(args.seed)
    with log_step(f"image conditioning (multi/{fusion_mode})", 1, 5) as t:
        cond_512 = pipeline.get_cond_multi(images, 512, fusion_mode)
        cond_1024 = (
            pipeline.get_cond_multi(images, 1024, fusion_mode)
            if args.pipeline != "512"
            else None
        )
    stage_times["conditioning_s"] = round(t[0], 2)
    with log_step("sparse structure", 2, 5) as t:
        coords = pipeline.sample_sparse_structure(cond_512, ss_res)
    stage_times["sparse_structure_s"] = round(t[0], 2)
    print(f"       {coords.shape[0]:,} voxels")
    if args.sparse_only:
        stage_times.update(
            {
                "shape_slat_s": None,
                "tex_slat_s": None,
                "decode_s": None,
                "preview_s": None,
            }
        )
        return {
            "coords": coords,
            "model_path": None,
            "preview_path": None,
            "stage_times": stage_times,
        }
    out = run_stages_3_to_5(pipeline, cond_512, cond_1024, coords, args, run_dir)
    out["stage_times"] = {**stage_times, **out["stage_times"]}
    out["coords"] = coords
    return out


def run_p2_core(
    pipeline,
    images: List[Image.Image],
    occupancy: np.ndarray,
    cond_fusion_mode: str,
    run_dir: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    from scaffold_bypass import occupancy_to_coords

    stage_times: Dict[str, Any] = {}
    torch.manual_seed(args.seed)
    coords = occupancy_to_coords(occupancy, device=pipeline.device)
    print(
        f"\n  [2/5] sparse structure ... skipped (hull: {coords.shape[0]:,} voxels)"
    )
    stage_times["sparse_structure_s"] = None
    if args.sparse_only:
        stage_times.update(
            {
                "conditioning_s": None,
                "shape_slat_s": None,
                "tex_slat_s": None,
                "decode_s": None,
                "preview_s": None,
            }
        )
        return {
            "coords": coords,
            "model_path": None,
            "preview_path": None,
            "stage_times": stage_times,
        }
    primary = images[0]
    with log_step(f"image conditioning ({cond_fusion_mode})", 1, 5) as t:
        if cond_fusion_mode in ("mean", "concat"):
            cond_512 = pipeline.get_cond_multi(images, 512, cond_fusion_mode)
            cond_1024 = (
                pipeline.get_cond_multi(images, 1024, cond_fusion_mode)
                if args.pipeline != "512"
                else None
            )
        else:
            cond_512 = pipeline.get_cond([primary], 512)
            cond_1024 = (
                pipeline.get_cond([primary], 1024)
                if args.pipeline != "512"
                else None
            )
    stage_times["conditioning_s"] = round(t[0], 2)
    out = run_stages_3_to_5(pipeline, cond_512, cond_1024, coords, args, run_dir)
    out["stage_times"] = {**stage_times, **out["stage_times"]}
    out["coords"] = coords
    return out


def args_namespace_to_dict(args: argparse.Namespace) -> dict:
    return {k: v for k, v in vars(args).items() if not k.startswith("_")}


def add_shared_pipeline_args(p: argparse.ArgumentParser) -> None:
    """Flags common to all strategy CLIs."""
    p.add_argument("--output-dir", default="./out")
    p.add_argument(
        "--dest-dir",
        default=None,
        help="Exact output directory (skip strategy_<timestamp> subfolder). "
        "Used by legacy wrappers.",
    )
    p.add_argument(
        "--pipeline",
        default="1024_cascade",
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    p.add_argument("--sparse-only", action="store_true")
    p.add_argument("--skip-tex", action="store_true")
    p.add_argument("--obj", action="store_true")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--texture-size", type=int, default=1024)
    p.add_argument("--decimation-target", type=int, default=500_000)
    p.add_argument(
        "--rembg-model",
        default="briaai/RMBG-2.0",
        help="BiRefNet HF id for preprocess / hull silhouettes",
    )
