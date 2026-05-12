"""
Test script for multi-image sparse occupancy fusion and SLAT feature fusion.

This file dispatches to ``scripts/runs/sparse_fusion.py`` and
``scripts/runs/slat_fusion.py`` (and optionally ``baseline.py``) so each run
executes in a fresh process with clean VRAM.

Usage is unchanged; see module docstring examples in the repository history or
``docs/multi-image-fusion-script.md``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test multi-image sparse occupancy fusion for TRELLIS.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("images", nargs="+", metavar="IMAGE")
    parser.add_argument("--output-dir", default="./out")
    parser.add_argument(
        "--fusion",
        choices=["union", "vote", "logit-mean", "logit-max", "logit-sum"],
        default="union",
    )
    parser.add_argument("--vote-threshold", type=float, default=0.5)
    parser.add_argument("--logit-threshold", type=float, default=0.0)
    parser.add_argument("--logit-smooth-sigma", type=float, default=0.0)
    parser.add_argument(
        "--slat-fusion",
        choices=["primary", "slat-mean", "slat-norm-weighted", "slat-max"],
        default="primary",
    )
    parser.add_argument("--samples-per-view", type=int, default=1)
    parser.add_argument("--filter-min-neighbors", type=int, default=0)
    parser.add_argument("--compare", action="store_true")
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
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--tex-steps", type=int, default=12)
    return parser.parse_args()


def _base(args: argparse.Namespace) -> list:
    cmd = [
        "--pipeline",
        args.pipeline,
        "--seed",
        str(args.seed),
        "--model",
        args.model,
        "--texture-size",
        str(args.texture_size),
        "--decimation-target",
        str(args.decimation),
        "--ss-steps",
        str(args.ss_steps),
        "--shape-steps",
        str(args.shape_steps),
        "--tex-steps",
        str(args.tex_steps),
    ]
    if args.no_preprocess:
        cmd.append("--no-preprocess")
    return cmd


def main() -> None:
    args = parse_args()
    py = sys.executable

    if args.compare or len(args.images) == 1:
        cmd = [
            py,
            os.path.join(_runs_dir, "baseline.py"),
            args.images[0],
            "--output-dir",
            args.output_dir,
            "--pipeline",
            args.pipeline,
            "--seed",
            str(args.seed),
            "--model",
            args.model,
            "--texture-size",
            str(args.texture_size),
            "--decimation-target",
            str(args.decimation),
        ]
        print(">>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=_repo_root, check=False)

    if len(args.images) < 2:
        return

    base = _base(args)
    common_multi = [*args.images, *base, "--output-dir", args.output_dir]

    if args.slat_fusion != "primary":
        cmd = [
            py,
            os.path.join(_runs_dir, "slat_fusion.py"),
            *common_multi,
            "--sparse-mode",
            args.fusion,
            "--mode",
            args.slat_fusion,
            "--vote-threshold",
            str(args.vote_threshold),
            "--logit-threshold",
            str(args.logit_threshold),
            "--logit-smooth-sigma",
            str(args.logit_smooth_sigma),
            "--samples-per-view",
            str(args.samples_per_view),
            "--filter-min-neighbors",
            str(args.filter_min_neighbors),
        ]
    else:
        cmd = [
            py,
            os.path.join(_runs_dir, "sparse_fusion.py"),
            *common_multi,
            "--fusion",
            args.fusion,
            "--vote-threshold",
            str(args.vote_threshold),
            "--logit-threshold",
            str(args.logit_threshold),
            "--logit-smooth-sigma",
            str(args.logit_smooth_sigma),
            "--samples-per-view",
            str(args.samples_per_view),
            "--filter-min-neighbors",
            str(args.filter_min_neighbors),
        ]

    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=_repo_root, check=False)


if __name__ == "__main__":
    main()
