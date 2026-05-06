"""
Sweep multiple fusion configurations without reloading the pipeline.

The pipeline and images are loaded once.  Per-view sparse diffusion runs are
cached per ``samples_per_view`` value, so every config that shares the same
``samples_per_view`` reuses the already-computed logit tensors.

Modes
-----
Default (full pipeline):
    Runs shape SLAT + texture SLAT + GLB export for every config.
    Expensive but produces inspectable meshes.

--sparse-only:
    Skips shape/texture SLAT and GLB export entirely.
    Only computes and fuses the sparse occupancy logits.
    The entire sweep (8+ configs) finishes in ~20s after the shared
    sparse diffusion run.  Use this to quickly compare voxel counts
    across fusion strategies before committing to a full run.

Config format (--configs path/to/configs.json)
----------------------------------------------
A JSON array of objects.  All keys are optional; unset keys fall back to
their CLI defaults.

    [
      {"label": "union",          "fusion": "union"},
      {"label": "lm",             "fusion": "logit-mean"},
      {"label": "lm+sm0.8",       "fusion": "logit-mean", "logit_smooth_sigma": 0.8},
      {"label": "lm+fn2",         "fusion": "logit-mean", "filter_min_neighbors": 2},
      {"label": "lmax",           "fusion": "logit-max"},
      {"label": "lsum",           "fusion": "logit-sum"},
      {"label": "vote33",         "fusion": "vote",       "vote_threshold": 0.33},
      {"label": "lm+spv3",        "fusion": "logit-mean", "samples_per_view": 3}
    ]

Examples
--------
# Fast voxel-count sweep with default configs (no GLB output):
python scripts/sweep_fusion.py front-left-top.png back-right.png front-left.png \\
    --sparse-only --output-dir ./out

# Full sweep (GLB per config):
python scripts/sweep_fusion.py front-left-top.png back-right.png front-left.png \\
    --output-dir ./out

# Custom config list:
python scripts/sweep_fusion.py front-left-top.png back-right.png front-left.png \\
    --configs my_configs.json --sparse-only --output-dir ./out
"""

import argparse
import json
import os
import sys
import time

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline

# ---------------------------------------------------------------------------
# Default sweep configs
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS = [
    {"label": "union",              "fusion": "union"},
    {"label": "union+fn2",          "fusion": "union",       "filter_min_neighbors": 2},
    {"label": "vote-33",            "fusion": "vote",        "vote_threshold": 0.33},
    {"label": "logit-mean",         "fusion": "logit-mean"},
    {"label": "logit-max",          "fusion": "logit-max"},
    {"label": "logit-sum",          "fusion": "logit-sum"},
    {"label": "logit-mean+sm0.5",   "fusion": "logit-mean",  "logit_smooth_sigma": 0.5},
    {"label": "logit-mean+sm0.8",   "fusion": "logit-mean",  "logit_smooth_sigma": 0.8},
    {"label": "logit-mean+sm1.2",   "fusion": "logit-mean",  "logit_smooth_sigma": 1.2},
    {"label": "logit-mean+fn2",     "fusion": "logit-mean",  "filter_min_neighbors": 2},
    {"label": "logit-max+sm0.8",    "fusion": "logit-max",   "logit_smooth_sigma": 0.8},
    {"label": "logit-mean+thr-0.5", "fusion": "logit-mean",  "logit_threshold": -0.5},
]

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


def make_label(cfg: dict) -> str:
    """Return cfg['label'] if set, else build one from the config values."""
    if "label" in cfg:
        return cfg["label"]
    parts = [cfg.get("fusion", "union")]
    if cfg.get("vote_threshold", 0.5) != 0.5 and cfg.get("fusion") == "vote":
        parts.append(f"thr{cfg['vote_threshold']}")
    if cfg.get("logit_threshold", 0.0) != 0.0:
        parts.append(f"thr{cfg['logit_threshold']}")
    if cfg.get("logit_smooth_sigma", 0.0) > 0.0:
        parts.append(f"sm{cfg['logit_smooth_sigma']}")
    if cfg.get("filter_min_neighbors", 0) > 0:
        parts.append(f"fn{cfg['filter_min_neighbors']}")
    if cfg.get("samples_per_view", 1) > 1:
        parts.append(f"spv{cfg['samples_per_view']}")
    return "+".join(parts)


def apply_fusion(
    per_view_logits: list,           # list of (B,1,R,R,R) float tensors
    fusion: str,
    vote_threshold: float,
    logit_threshold: float,
    logit_smooth_sigma: float,
    filter_min_neighbors: int,
    resolution: int,
    pipeline,
) -> tuple:
    """
    Apply fusion to a cached list of per-view logit tensors.

    Returns (coords, fused_bool_volume).
    """
    _LOGIT_MODES = {"logit-mean", "logit-max", "logit-sum"}
    stacked_logits = torch.stack(per_view_logits, dim=0)   # (V, B, 1, R, R, R)
    stacked_bool   = stacked_logits > 0                    # pre-threshold view vols

    if fusion == "union":
        fused = stacked_bool.any(dim=0)
    elif fusion == "vote":
        fused = stacked_bool.float().mean(dim=0) >= vote_threshold
    elif fusion in _LOGIT_MODES:
        if fusion == "logit-mean":
            fused_logits = stacked_logits.mean(dim=0)
        elif fusion == "logit-max":
            fused_logits = stacked_logits.max(dim=0).values
        else:  # logit-sum
            fused_logits = stacked_logits.sum(dim=0)

        if logit_smooth_sigma > 0.0:
            fused_logits = pipeline._smooth_logit_volume(fused_logits, logit_smooth_sigma)

        fused = fused_logits > logit_threshold
    else:
        raise ValueError(f"Unknown fusion mode: {fusion!r}")

    if filter_min_neighbors > 0:
        fused = pipeline._filter_isolated_voxels(fused, filter_min_neighbors)

    # Downsample if needed
    if resolution != fused.shape[2]:
        ratio = fused.shape[2] // resolution
        fused = F.max_pool3d(fused.float(), ratio, ratio, 0) > 0.5

    coords = torch.argwhere(fused)[:, [0, 2, 3, 4]].int()
    return coords, fused


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
        use_tqdm=True,
    )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{label}_{timestamp()}.glb")
    glb.export(path, extension_webp=True)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep fusion configurations without reloading the pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "images", nargs="+", metavar="IMAGE",
        help="Input image paths (first is the primary/front view).",
    )
    parser.add_argument(
        "--output-dir", default="./out",
        help="Directory for output GLB files (default: ./out).",
    )
    parser.add_argument(
        "--configs", metavar="JSON",
        help="Path to a JSON file containing a list of config dicts.  "
             "Defaults to the built-in DEFAULT_CONFIGS list.",
    )
    parser.add_argument(
        "--sparse-only", action="store_true",
        help="Skip shape/texture SLAT and GLB export.  Only measure voxel "
             "counts from each fusion config.  Very fast — use for rapid "
             "exploration.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
        default="1024_cascade",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--decimation", type=int, default=200000)
    parser.add_argument("--ss-steps",    type=int, default=12)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--tex-steps",   type=int, default=12)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Load configs ---
    if args.configs:
        with open(args.configs, encoding="utf-8") as f:
            configs = json.load(f)
        print(f"Loaded {len(configs)} configs from {args.configs}")
    else:
        configs = DEFAULT_CONFIGS
        print(f"Using {len(configs)} built-in default configs.")

    # Fill in defaults for any missing keys
    for cfg in configs:
        cfg.setdefault("fusion", "union")
        cfg.setdefault("vote_threshold", 0.5)
        cfg.setdefault("logit_threshold", 0.0)
        cfg.setdefault("logit_smooth_sigma", 0.0)
        cfg.setdefault("samples_per_view", 1)
        cfg.setdefault("filter_min_neighbors", 0)
        if "label" not in cfg:
            cfg["label"] = make_label(cfg)

    # --- Load images ---
    print_section("Loading images")
    raw_images = []
    for path in args.images:
        if not os.path.isfile(path):
            print(f"ERROR: image not found: {path}", file=sys.stderr)
            sys.exit(1)
        img = Image.open(path).convert("RGBA")
        raw_images.append(img)
        print(f"  loaded: {path}  ({img.width}×{img.height})")

    # --- Load pipeline ---
    print_section(f"Loading pipeline: {args.model}")
    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    print(f"  pipeline loaded in {time.time() - t0:.1f}s")

    preprocess = not args.no_preprocess
    pipeline_type = args.pipeline
    ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[pipeline_type]

    ss_params    = {"steps": args.ss_steps}
    shape_params = {"steps": args.shape_steps}
    tex_params   = {"steps": args.tex_steps}

    # --- Preprocess + conditioning (done once) ---
    print_section("Preprocessing images and extracting conditioning")
    t0 = time.time()
    if preprocess:
        images = [pipeline.preprocess_image(img) for img in raw_images]
    else:
        images = raw_images

    conds_512 = [pipeline.get_cond([img], 512) for img in images]
    cond_512_primary  = conds_512[0]
    cond_1024_primary = pipeline.get_cond([images[0]], 1024) if pipeline_type != "512" else None
    print(f"  conditioning ready in {time.time() - t0:.1f}s")

    # --- Cache per-view logits, keyed by samples_per_view ---
    # Collect all unique spv values needed by the configs
    spv_values = sorted({cfg["samples_per_view"] for cfg in configs})

    logit_cache: dict[int, list] = {}  # spv → list of (B,1,R,R,R) float tensors

    for spv in spv_values:
        print_section(
            f"Sampling per-view logits  "
            f"(samples_per_view={spv}, {len(images)} views)"
        )
        t0 = time.time()
        torch.manual_seed(args.seed)
        per_view_logits = []
        for view_idx, cond in enumerate(conds_512):
            print(f"  view {view_idx + 1}/{len(conds_512)}"
                  + (f" (×{spv} ensemble)" if spv > 1 else "") + " ...")
            _vol, logits = pipeline._sample_occupancy_volume(
                cond, 1, ss_params, samples_per_view=spv, return_logits=True
            )
            voxels = int((_vol).sum().item())
            print(f"    occupied (pre-fusion): {voxels:,} voxels")
            per_view_logits.append(logits)
        logit_cache[spv] = per_view_logits
        print(f"  sparse sampling done in {time.time() - t0:.1f}s")

    # --- Sweep configs ---
    print_section(f"Sweeping {len(configs)} configs")
    results = []

    for cfg_idx, cfg in enumerate(configs):
        label   = cfg["label"]
        fusion  = cfg["fusion"]
        spv     = cfg["samples_per_view"]
        per_view_logits = logit_cache[spv]

        print(f"\n[{cfg_idx + 1}/{len(configs)}] {label}")
        t_fuse = time.time()

        coords, fused_vol = apply_fusion(
            per_view_logits,
            fusion=fusion,
            vote_threshold=cfg["vote_threshold"],
            logit_threshold=cfg["logit_threshold"],
            logit_smooth_sigma=cfg["logit_smooth_sigma"],
            filter_min_neighbors=cfg["filter_min_neighbors"],
            resolution=ss_res,
            pipeline=pipeline,
        )
        n_voxels_fused = int(coords.shape[0])
        elapsed_fuse = time.time() - t_fuse
        print(f"  fused voxels: {n_voxels_fused:,}  ({elapsed_fuse:.2f}s)")

        result = {
            "label":        label,
            "fusion":       fusion,
            "spv":          spv,
            "voxels_fused": n_voxels_fused,
            "elapsed_fuse": elapsed_fuse,
            "glb_path":     None,
            "voxels_slat":  None,
            "elapsed_full": None,
        }

        if not args.sparse_only:
            t_full = time.time()
            torch.manual_seed(args.seed)

            # Shape SLAT
            if pipeline_type == "512":
                shape_slat = pipeline.sample_shape_slat(
                    cond_512_primary,
                    pipeline.models["shape_slat_flow_model_512"],
                    coords, shape_params,
                )
                tex_slat = pipeline.sample_tex_slat(
                    cond_512_primary,
                    pipeline.models["tex_slat_flow_model_512"],
                    shape_slat, tex_params,
                )
                res = 512
            elif pipeline_type == "1024":
                shape_slat = pipeline.sample_shape_slat(
                    cond_1024_primary,
                    pipeline.models["shape_slat_flow_model_1024"],
                    coords, shape_params,
                )
                tex_slat = pipeline.sample_tex_slat(
                    cond_1024_primary,
                    pipeline.models["tex_slat_flow_model_1024"],
                    shape_slat, tex_params,
                )
                res = 1024
            elif pipeline_type in ("1024_cascade", "1536_cascade"):
                shape_slat, res = pipeline.sample_shape_slat_cascade(
                    cond_512_primary, cond_1024_primary,
                    pipeline.models["shape_slat_flow_model_512"],
                    pipeline.models["shape_slat_flow_model_1024"],
                    512, 1024,
                    coords, shape_params,
                )
                tex_slat = pipeline.sample_tex_slat(
                    cond_1024_primary,
                    pipeline.models["tex_slat_flow_model_1024"],
                    shape_slat, tex_params,
                )

            mesh_list = pipeline.decode_latent(shape_slat, tex_slat, res)
            glb_path = save_glb(
                pipeline, mesh_list[0], res, args.output_dir,
                label.replace("/", "_"), args.texture_size, args.decimation,
            )
            elapsed_full = time.time() - t_full
            print(f"  SLAT+GLB done in {elapsed_full:.1f}s  →  {glb_path}")

            result["glb_path"]    = glb_path
            result["voxels_slat"] = int(shape_slat.coords.shape[0])
            result["elapsed_full"] = elapsed_full

            torch.cuda.empty_cache()

        results.append(result)

    # --- Summary table ---
    print_section("Sweep summary")
    col_w = max(len(r["label"]) for r in results) + 2

    if args.sparse_only:
        header = f"{'Config':<{col_w}} {'Vox(fused)':>12}  {'t_fuse':>8}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r['label']:<{col_w}} {r['voxels_fused']:>12,}  "
                f"{r['elapsed_fuse']:>7.2f}s"
            )
    else:
        header = (
            f"{'Config':<{col_w}} {'Vox(fused)':>12} {'Vox(SLAT)':>12} "
            f"{'t_fuse':>8} {'t_full':>8}  GLB"
        )
        print(header)
        print("-" * len(header))
        for r in results:
            slat_str = f"{r['voxels_slat']:>12,}" if r["voxels_slat"] is not None else f"{'—':>12}"
            full_str = f"{r['elapsed_full']:>7.1f}s" if r["elapsed_full"] is not None else f"{'—':>8}"
            glb_name = os.path.basename(r["glb_path"]) if r["glb_path"] else "—"
            print(
                f"{r['label']:<{col_w}} {r['voxels_fused']:>12,} {slat_str} "
                f"{r['elapsed_fuse']:>7.2f}s {full_str}  {glb_name}"
            )
    print()


if __name__ == "__main__":
    main()
