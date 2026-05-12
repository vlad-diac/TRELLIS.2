#!/usr/bin/env python3
"""Single-image TRELLIS.2 baseline (first image only if multiple paths given)."""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

# Ensure runs dir on path
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
    print_banner,
    resolve_run_dir,
    unload_pipeline,
    unload_rembg_model,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRELLIS.2 baseline — single-image pipeline")
    p.add_argument(
        "images",
        nargs="+",
        help="Input image paths (only the first is used)",
    )
    add_shared_pipeline_args(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    strategy = "baseline"
    run_dir = resolve_run_dir(args.output_dir, strategy, args.dest_dir)
    print_banner(f"{strategy} → {run_dir}")
    image_paths = [os.path.abspath(x) for x in args.images]
    if len(args.images) > 1:
        print(f"  [info] using first image only: {image_paths[0]}")
    copy_inputs_to_run_dir(run_dir, image_paths)

    from PIL import Image

    print_banner("Phase 1 — BiRefNet preprocess")
    with log_step(f"[LOAD] BiRefNet ({args.rembg_model})") as t:
        rembg_model = load_rembg_model(args.rembg_model)
    print(f"       {gpu_mem_str()}")
    raw = Image.open(args.images[0]).convert("RGB")
    with log_step("preprocess image") as t:
        from _common import preprocess_image_standalone

        proc = preprocess_image_standalone(raw, rembg_model)
    with log_step("[UNLOAD] BiRefNet") as t:
        unload_rembg_model(rembg_model)
    print("       GPU memory cleared")

    t_wall0 = time.time()
    stage_times = {}
    pipeline = None
    error = None
    coords = None
    voxels = 0
    model_path = None
    preview_path = None

    try:
        with log_step(f"[LOAD] {args.model}") as t:
            pipeline = load_pipeline(args.model)
        print(f"       {gpu_mem_str()}")
        stage_times["load_s"] = round(t[0], 2)

        from _common import run_baseline_core

        out = run_baseline_core(pipeline, [proc], run_dir, args)
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
        "mode": "single",
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
