#!/usr/bin/env python3
"""Sparse occupancy fusion via pipeline.run_multi_image (SLAT fusion = primary)."""
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
    args_namespace_to_dict,
    copy_inputs_to_run_dir,
    gpu_mem_str,
    load_pipeline,
    log_step,
    print_banner,
    resolve_run_dir,
    save_glb,
    save_preview,
    unload_pipeline,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-image sparse fusion (union / vote / logit-*) + primary SLAT"
    )
    p.add_argument("images", nargs="+", metavar="IMAGE")
    p.add_argument("--output-dir", default="./out")
    p.add_argument("--dest-dir", default=None)
    p.add_argument(
        "--fusion",
        choices=["union", "vote", "logit-mean", "logit-max", "logit-sum"],
        default="union",
    )
    p.add_argument("--vote-threshold", type=float, default=0.5)
    p.add_argument("--logit-threshold", type=float, default=0.0)
    p.add_argument("--logit-smooth-sigma", type=float, default=0.0)
    p.add_argument("--samples-per-view", type=int, default=1)
    p.add_argument("--filter-min-neighbors", type=int, default=0)
    p.add_argument(
        "--pipeline",
        default="1024_cascade",
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-preprocess", action="store_true")
    p.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    p.add_argument("--texture-size", type=int, default=1024)
    p.add_argument("--decimation-target", type=int, default=200_000)
    p.add_argument("--ss-steps", type=int, default=12)
    p.add_argument("--shape-steps", type=int, default=12)
    p.add_argument("--tex-steps", type=int, default=12)
    p.add_argument("--no-preview", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.images) < 2:
        print("ERROR: sparse_fusion requires at least 2 images.", file=sys.stderr)
        return 2

    strategy = "sparse_fusion"
    run_dir = resolve_run_dir(args.output_dir, strategy, args.dest_dir)
    print_banner(f"{strategy} ({args.fusion}) → {run_dir}")
    image_paths = [os.path.abspath(x) for x in args.images]
    copy_inputs_to_run_dir(run_dir, image_paths)

    from PIL import Image

    raw = [Image.open(p).convert("RGBA") for p in args.images]

    ss_params = {"steps": args.ss_steps}
    shape_params = {"steps": args.shape_steps}
    tex_params = {"steps": args.tex_steps}

    t_wall0 = time.time()
    stage_times: dict = {}
    pipeline = None
    error = None
    model_path = None
    preview_path = None
    voxels_fused = 0

    try:
        with log_step(f"[LOAD] {args.model}") as t:
            pipeline = load_pipeline(args.model)
        print(f"       {gpu_mem_str()}")
        stage_times["load_s"] = round(t[0], 2)

        with log_step("run_multi_image (sparse fusion, primary SLAT)") as t:
            import torch

            torch.manual_seed(args.seed)
            out_multi, (shape_slat, _tex_slat, res_m) = pipeline.run_multi_image(
                raw,
                seed=args.seed,
                fusion_mode=args.fusion,
                vote_threshold=args.vote_threshold,
                logit_threshold=args.logit_threshold,
                logit_smooth_sigma=args.logit_smooth_sigma,
                samples_per_view=args.samples_per_view,
                filter_min_neighbors=args.filter_min_neighbors,
                slat_fusion_mode="primary",
                preprocess_image=not args.no_preprocess,
                sparse_structure_sampler_params=ss_params,
                shape_slat_sampler_params=shape_params,
                tex_slat_sampler_params=tex_params,
                pipeline_type=args.pipeline,
                return_latent=True,
            )
        stage_times["pipeline_s"] = round(t[0], 2)
        voxels_fused = int(shape_slat.coords.shape[0])

        model_path = save_glb(
            pipeline,
            out_multi[0],
            res_m,
            run_dir,
            args.texture_size,
            args.decimation_target,
        )
        print(f"       → {model_path}")

        if not args.no_preview:
            with log_step("preview render") as t:
                preview_path = save_preview(out_multi[0], run_dir)
            stage_times["preview_s"] = round(t[0], 2)
            if preview_path:
                print(f"       → {preview_path}")
        else:
            stage_times["preview_s"] = None

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
        "mode": args.fusion,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": elapsed,
        "voxels": voxels_fused,
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
