"""
Multi-View TRELLIS.2 Benchmark — Full Test Matrix

Dispatches each condition to a dedicated script under ``scripts/runs/`` so each
run loads and fully unloads the 4B pipeline in a separate process (clean VRAM).

See ``docs/benchmark-multiview-script.md`` for the full research matrix.

Legacy CLI and output layout ``run_YYYYMMDD_HHMMSS/<condition>/`` are preserved.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def cond_folder_name(condition_name: str) -> str:
    return condition_name.replace("+", "_plus_")


def run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full multi-view benchmark test matrix for TRELLIS.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "images",
        nargs="+",
        help="Input image paths, one per viewpoint (first = front).",
    )
    p.add_argument("--output-dir", default="./out")
    p.add_argument(
        "--azimuths",
        nargs="+",
        type=float,
        default=None,
        help="Camera azimuth per image in degrees (default: evenly spaced 360°).",
    )
    p.add_argument("--elevation", type=float, default=15.0)
    p.add_argument(
        "--skip-scaffold",
        action="store_true",
        help="Skip P2-scaffold and P2-scaffold+P1.",
    )
    p.add_argument(
        "--sparse-only",
        action="store_true",
        help="Stop after stage 2; count voxels only, no model export.",
    )
    p.add_argument(
        "--skip-tex",
        action="store_true",
        help="Skip stage 4 (texture SLAT); output geometry-only.",
    )
    p.add_argument(
        "--obj",
        action="store_true",
        help="Output all models as geometry-only .obj (implies --skip-tex).",
    )
    p.add_argument("--no-preview", action="store_true")
    p.add_argument(
        "--pipeline",
        default="1024_cascade",
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    p.add_argument("--rembg-model", default="briaai/RMBG-2.0")
    p.add_argument("--texture-size", type=int, default=1024)
    p.add_argument("--decimation-target", type=int, default=500_000)
    p.add_argument(
        "--min-views-scaffold",
        type=int,
        default=None,
        help="Min views for hull carving (default: all views).",
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        choices=[
            "baseline",
            "P1-mean",
            "P1-concat",
            "P2-scaffold",
            "P2-scaffold+P1",
        ],
        default=None,
    )
    return p.parse_args()


def _shared_flags(args: argparse.Namespace) -> List[str]:
    out: List[str] = [
        "--pipeline",
        args.pipeline,
        "--seed",
        str(args.seed),
        "--model",
        args.model,
        "--rembg-model",
        args.rembg_model,
        "--texture-size",
        str(args.texture_size),
        "--decimation-target",
        str(args.decimation_target),
    ]
    if args.sparse_only:
        out.append("--sparse-only")
    if args.skip_tex:
        out.append("--skip-tex")
    if args.obj:
        out.append("--obj")
    if args.no_preview:
        out.append("--no-preview")
    return out


def _dispatch_condition(
    cond_name: str,
    args: argparse.Namespace,
    dest_dir: str,
) -> int:
    py = sys.executable
    base = _shared_flags(args)
    images = list(args.images)

    if cond_name == "baseline":
        cmd = [py, os.path.join(_runs_dir, "baseline.py"), *images, *base]
    elif cond_name == "P1-mean":
        cmd = [
            py,
            os.path.join(_runs_dir, "p1_condition_fusion.py"),
            *images,
            "--mode",
            "mean",
            *base,
        ]
    elif cond_name == "P1-concat":
        cmd = [
            py,
            os.path.join(_runs_dir, "p1_condition_fusion.py"),
            *images,
            "--mode",
            "concat",
            *base,
        ]
    elif cond_name == "P2-scaffold":
        cmd = [
            py,
            os.path.join(_runs_dir, "p2_scaffold.py"),
            *images,
            "--cond-fusion",
            "primary",
            "--elevation",
            str(args.elevation),
            *base,
        ]
        if args.azimuths:
            cmd += ["--azimuths", *[str(x) for x in args.azimuths]]
        if args.min_views_scaffold is not None:
            cmd += ["--min-views-scaffold", str(args.min_views_scaffold)]
    elif cond_name == "P2-scaffold+P1":
        cmd = [
            py,
            os.path.join(_runs_dir, "p2_scaffold.py"),
            *images,
            "--cond-fusion",
            "mean",
            "--elevation",
            str(args.elevation),
            *base,
        ]
        if args.azimuths:
            cmd += ["--azimuths", *[str(x) for x in args.azimuths]]
        if args.min_views_scaffold is not None:
            cmd += ["--min-views-scaffold", str(args.min_views_scaffold)]
    else:
        raise ValueError(cond_name)

    cmd += ["--dest-dir", dest_dir, "--output-dir", args.output_dir]
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, cwd=_repo_root).returncode


def _load_sub_summary(cond_dir: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(cond_dir, "summary.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _print_table(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 64)
    print("  Benchmark Summary (merged)")
    print("=" * 64)
    hdr = f"{'Condition':<24} {'Voxels':>8} {'Elapsed':>9}  Status"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['name']:<24} {r['voxels']:>8} {r['elapsed_s']:>8.1f}s  {r['status']}"
        )
        if r.get("model_path"):
            print(f"    model   → {r['model_path']}")
        if r.get("preview_path"):
            print(f"    preview → {r['preview_path']}")
        if r.get("error"):
            print(f"    error   : {r['error']}")


def main() -> None:
    args = parse_args()
    all_conds = [
        "baseline",
        "P1-mean",
        "P1-concat",
        "P2-scaffold",
        "P2-scaffold+P1",
    ]
    if args.conditions:
        conditions = args.conditions
    elif args.skip_scaffold:
        conditions = ["baseline", "P1-mean", "P1-concat"]
    else:
        conditions = all_conds

    print("\n" + "=" * 64)
    print("  TRELLIS.2 Multi-View Benchmark (subprocess dispatcher)")
    print("=" * 64)
    print(f"  Model      : {args.model}")
    print(f"  Pipeline   : {args.pipeline}")
    print(f"  Images     : {args.images}")
    print(f"  Conditions : {conditions}")

    run_ts = run_timestamp()
    run_dir = os.path.join(args.output_dir, f"run_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"  Run dir    : {run_dir}\n")

    wall0 = time.time()
    preprocess_note = (
        "Each subprocess runs BiRefNet preprocess independently (clean VRAM between runs)."
    )
    print(f"  Note: {preprocess_note}\n")

    for cond_name in conditions:
        sub = os.path.join(run_dir, cond_folder_name(cond_name))
        code = _dispatch_condition(cond_name, args, sub)
        if code != 0:
            print(f"\n[WARN] Condition {cond_name!r} exited with code {code}")

    total_elapsed = round(time.time() - wall0, 2)

    n = len(args.images)
    azimuths = (
        list(args.azimuths)
        if args.azimuths is not None
        else [360.0 * i / n for i in range(n)]
    )

    results: List[Dict[str, Any]] = []
    scaffold_merged: Optional[dict] = None
    for cond_name in conditions:
        sub = os.path.join(run_dir, cond_folder_name(cond_name))
        js = _load_sub_summary(sub)
        if js is None:
            results.append(
                {
                    "name": cond_name,
                    "voxels": 0,
                    "elapsed_s": 0.0,
                    "status": "missing-summary",
                    "model_path": None,
                    "preview_path": None,
                    "error": "no summary.json",
                    "stage_times": {},
                }
            )
            continue
        if js.get("scaffold") and scaffold_merged is None:
            scaffold_merged = js["scaffold"]
        err = js.get("error")
        results.append(
            {
                "name": cond_name,
                "voxels": js.get("voxels", 0),
                "elapsed_s": js.get("elapsed_s", 0.0),
                "status": "ERROR" if err else ("ok" if js.get("model_path") else "sparse-only"),
                "model_path": js.get("model_path"),
                "preview_path": js.get("preview_path"),
                "error": err,
                "stage_times": js.get("stage_times", {}),
                "iou_vs_baseline": js.get("iou_vs_baseline"),
            }
        )

    _print_table(results)

    run_meta = {
        "run_id": run_ts,
        "run_dir": os.path.abspath(run_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "total_elapsed_s": total_elapsed,
        "preprocess_elapsed_s": None,
        "dispatcher_note": preprocess_note,
        "images": [os.path.abspath(p) for p in args.images],
        "azimuths_deg": azimuths,
        "elevation_deg": args.elevation,
        "model": args.model,
        "rembg_model": args.rembg_model,
        "pipeline_type": args.pipeline,
        "seed": args.seed,
        "skip_tex": args.skip_tex,
        "obj": args.obj,
        "skip_scaffold": args.skip_scaffold,
        "sparse_only": args.sparse_only,
        "no_preview": args.no_preview,
        "texture_size": args.texture_size,
        "decimation_target": args.decimation_target,
        "conditions_run": conditions,
        "scaffold": scaffold_merged,
    }
    merged = {"run": run_meta, "conditions": results}
    out_path = os.path.join(run_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged summary : {out_path}")
    print("Done.\n")


if __name__ == "__main__":
    main()
