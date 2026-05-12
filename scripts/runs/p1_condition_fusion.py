#!/usr/bin/env python3
"""P1 — DINOv3 patch-token fusion (mean or concat) before sparse structure."""
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
    gpu_mem_str,
    load_pipeline,
    load_rembg_model,
    log_step,
    preprocess_image_standalone,
    print_banner,
    resolve_run_dir,
    unload_pipeline,
    unload_rembg_model,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P1 condition fusion — get_cond_multi(mean|concat)"
    )
    p.add_argument("images", nargs="+", help="Input image paths (multi-view)")
    p.add_argument(
        "--mode",
        choices=("mean", "concat"),
        default="mean",
        help="Token fusion mode (default: mean)",
    )
    add_shared_pipeline_args(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.images) < 2:
        print("ERROR: p1_condition_fusion requires at least 2 images.", file=sys.stderr)
        return 2

    strategy = "p1_condition_fusion"
    run_dir = resolve_run_dir(args.output_dir, strategy, args.dest_dir)
    print_banner(f"{strategy} ({args.mode}) → {run_dir}")
    image_paths = [os.path.abspath(x) for x in args.images]
    copy_inputs_to_run_dir(run_dir, image_paths)

    from PIL import Image

    raw_images = [Image.open(p).convert("RGB") for p in args.images]

    print_banner("Phase 1 — BiRefNet preprocess")
    with log_step(f"[LOAD] BiRefNet ({args.rembg_model})") as t:
        rembg_model = load_rembg_model(args.rembg_model)
    print(f"       {gpu_mem_str()}")
    with log_step(f"preprocess {len(raw_images)} image(s)") as t:
        images = [preprocess_image_standalone(img, rembg_model) for img in raw_images]
    with log_step("[UNLOAD] BiRefNet") as t:
        unload_rembg_model(rembg_model)
    print("       GPU memory cleared")

    t_wall0 = time.time()
    stage_times = {}
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

        from _common import run_p1_core

        out = run_p1_core(pipeline, images, args.mode, run_dir, args)
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
        "mode": args.mode,
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
    }
    write_summary(run_dir, summary)
    print(f"Summary JSON : {os.path.join(run_dir, 'summary.json')}")
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
