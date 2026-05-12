#!/usr/bin/env python3
"""P2 — visual-hull scaffold bypass + SLAT stages (primary | mean | concat cond)."""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

_RUNS = os.path.dirname(os.path.abspath(__file__))
if _RUNS not in sys.path:
    sys.path.insert(0, _RUNS)

from _common import (
    add_shared_pipeline_args,
    args_namespace_to_dict,
    copy_inputs_to_run_dir,
    extract_silhouette_rembg,
    gpu_mem_str,
    load_pipeline,
    load_rembg_model,
    log_step,
    preprocess_image_standalone,
    print_banner,
    resolve_run_dir,
    sparse_grid_resolution,
    unload_pipeline,
    unload_rembg_model,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P2 scaffold — space-carve hull + stages 3–5")
    p.add_argument("images", nargs="+", help="Input images (raw RGB paths)")
    p.add_argument(
        "--cond-fusion",
        choices=("primary", "mean", "concat"),
        default="primary",
        help="SLAT conditioning: first view only, or fused tokens",
    )
    p.add_argument("--azimuths", nargs="+", type=float, default=None)
    p.add_argument("--elevation", type=float, default=15.0)
    p.add_argument(
        "--min-views-scaffold",
        type=int,
        default=None,
        help="Min views inside silhouette (default: all views)",
    )
    add_shared_pipeline_args(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.images) < 2:
        print("ERROR: p2_scaffold needs at least 2 images for a useful hull.", file=sys.stderr)
        return 2

    strategy = "p2_scaffold"
    run_dir = resolve_run_dir(args.output_dir, strategy, args.dest_dir)
    print_banner(f"{strategy} (cond={args.cond_fusion}) → {run_dir}")
    image_paths = [os.path.abspath(x) for x in args.images]
    copy_inputs_to_run_dir(run_dir, image_paths)

    from PIL import Image

    from scaffold_bypass import make_look_at_rotation, space_carve

    n = len(args.images)
    raw_images = [Image.open(p).convert("RGB") for p in args.images]

    print_banner("Phase 1 — BiRefNet preprocess + hull")
    with log_step(f"[LOAD] BiRefNet ({args.rembg_model})") as t:
        rembg_model = load_rembg_model(args.rembg_model)
    print(f"       {gpu_mem_str()}")

    with log_step(f"preprocess {n} image(s)") as t:
        images = [preprocess_image_standalone(img, rembg_model) for img in raw_images]

    if args.azimuths is not None:
        if len(args.azimuths) != n:
            raise ValueError(
                f"--azimuths has {len(args.azimuths)} values but {n} images given."
            )
        azimuths = list(args.azimuths)
    else:
        azimuths = [360.0 * i / n for i in range(n)]
        print(f"       auto azimuths: {[f'{a:.1f}°' for a in azimuths]}")

    with log_step("extract silhouettes (raw)") as t:
        scaffold_masks = [extract_silhouette_rembg(rembg_model, raw) for raw in raw_images]

    rotations = [make_look_at_rotation(az, args.elevation) for az in azimuths]
    ss_res = sparse_grid_resolution(args.pipeline)
    min_views = args.min_views_scaffold if args.min_views_scaffold is not None else len(scaffold_masks)

    with log_step(f"space carve (grid={ss_res}³, min_views={min_views})") as t:
        occupancy = space_carve(
            scaffold_masks, rotations, grid_size=ss_res, min_views=min_views
        )
    n_occ = int(occupancy.sum())
    density = 100 * n_occ / occupancy.size
    print(f"       {n_occ:,} / {occupancy.size:,} voxels ({density:.1f}% density)")

    with log_step("[UNLOAD] BiRefNet") as t:
        unload_rembg_model(rembg_model)
    print("       GPU memory cleared")

    scaffold_info = {
        "grid_size": ss_res,
        "min_views": min_views,
        "occupied_voxels": n_occ,
        "total_voxels": int(occupancy.size),
        "density_pct": round(density, 2),
    }

    t_wall0 = time.time()
    stage_times: dict = {}
    pipeline = None
    error = None
    voxels = 0
    model_path = None
    preview_path = None
    coords = None

    try:
        with log_step(f"[LOAD] {args.model}") as t:
            pipeline = load_pipeline(args.model)
        print(f"       {gpu_mem_str()}")
        stage_times["load_s"] = round(t[0], 2)

        from _common import run_p2_core

        out = run_p2_core(
            pipeline, images, occupancy, args.cond_fusion, run_dir, args
        )
        stage_times.update(out["stage_times"])
        coords = out["coords"]
        voxels = int(coords.shape[0]) if coords is not None else 0
        model_path = out["model_path"]
        preview_path = out["preview_path"]
    except Exception as exc:
        error = str(exc)
        print(f"\n  [ERROR] {exc}")
    finally:
        if pipeline is not None:
            with log_step(f"[UNLOAD] {args.model}") as t:
                unload_pipeline(pipeline)
            print("       GPU memory cleared")
            stage_times["unload_s"] = round(t[0], 2)

    elapsed = round(time.time() - t_wall0, 2)
    summary = {
        "strategy": strategy,
        "mode": args.cond_fusion,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": elapsed,
        "voxels": voxels,
        "iou_vs_baseline": None,
        "stage_times": stage_times,
        "args": args_namespace_to_dict(args),
        "images": image_paths,
        "run_dir": run_dir,
        "model_path": model_path,
        "preview_path": preview_path,
        "error": error,
        "scaffold": scaffold_info,
    }
    write_summary(run_dir, summary)
    print(f"Summary JSON : {os.path.join(run_dir, 'summary.json')}")
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
