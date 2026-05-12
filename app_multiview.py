"""
Gradio UI: pick inputs from ./input, choose strategy, run scripts/runs/*.py via subprocess
(streaming logs), inspect GLB + preview + summary.json. Does not import trellis2/torch.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(REPO_ROOT, "scripts", "runs")
INPUT_ROOT = os.path.join(REPO_ROOT, "input")
DEFAULT_OUT = os.path.join(REPO_ROOT, "out")

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")


def discover_inputs(input_root: str) -> Tuple[List[str], Dict[str, List[str]]]:
    files: List[str] = []
    folders: Dict[str, List[str]] = {}
    if not os.path.isdir(input_root):
        return files, folders
    for name in sorted(os.listdir(input_root)):
        path = os.path.join(input_root, name)
        if os.path.isdir(path):
            imgs = sorted(
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in IMAGE_EXT
                and os.path.isfile(os.path.join(path, f))
            )
            if imgs:
                folders[name] = imgs
        elif os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXT:
            files.append(path)
    return files, folders


def latest_strategy_run_dir(output_dir: str, strategy: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    prefix = strategy + "_"
    cand = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith(prefix) and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not cand:
        return None
    return max(cand, key=os.path.getmtime)


def scan_summaries(output_dir: str, limit: int = 200) -> List[dict]:
    rows: List[dict] = []
    if not os.path.isdir(output_dir):
        return rows
    for path in glob.glob(os.path.join(output_dir, "**", "summary.json"), recursive=True):
        try:
            with open(path, encoding="utf-8") as f:
                js = json.load(f)
        except Exception:
            continue
        if not isinstance(js, dict):
            continue
        if "run" in js and "conditions" in js:
            run_id = js["run"].get("run_id", "")
            for c in js["conditions"]:
                rows.append(
                    {
                        "path": path,
                        "dir": os.path.join(os.path.dirname(path), c.get("name", "")),
                        "strategy": "benchmark",
                        "mode": c.get("name", ""),
                        "voxels": c.get("voxels", ""),
                        "elapsed_s": c.get("elapsed_s", ""),
                        "started": js["run"].get("started_at", ""),
                        "error": c.get("error"),
                    }
                )
            continue
        if "strategy" in js:
            rows.append(
                {
                    "path": path,
                    "dir": os.path.dirname(path),
                    "strategy": js.get("strategy", ""),
                    "mode": str(
                        js.get("mode", js.get("sparse_mode", ""))
                    ),
                    "voxels": js.get("voxels", ""),
                    "elapsed_s": js.get("elapsed_s", ""),
                    "started": js.get("started_at", ""),
                    "error": js.get("error"),
                }
            )
    rows.sort(key=lambda r: r.get("path", ""), reverse=True)
    return rows[:limit]


def build_command(
    strategy: str,
    image_paths: List[str],
    output_dir: str,
    *,
    model: str,
    pipeline: str,
    seed: int,
    texture_size: int,
    decimation_target: int,
    skip_tex: bool,
    obj: bool,
    no_preview: bool,
    sparse_only: bool,
    rembg_model: str,
    p1_mode: str,
    p2_cond_fusion: str,
    p2_elevation: float,
    p2_azimuths_text: str,
    p2_min_views: int,
    sparse_fusion: str,
    vote_threshold: float,
    logit_threshold: float,
    logit_smooth: float,
    samples_per_view: int,
    filter_min_neighbors: int,
    ss_steps: int,
    shape_steps: int,
    tex_steps: int,
    no_preprocess_multi: bool,
    slat_sparse_mode: str,
    slat_mode: str,
) -> List[str]:
    script = os.path.join(RUNS_DIR, f"{strategy}.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(script)
    cmd: List[str] = [sys.executable, script]
    cmd += image_paths

    if strategy in ("baseline", "p1_condition_fusion", "p2_scaffold"):
        cmd += [
            "--output-dir",
            output_dir,
            "--pipeline",
            pipeline,
            "--seed",
            str(seed),
            "--model",
            model,
            "--rembg-model",
            rembg_model,
            "--texture-size",
            str(texture_size),
            "--decimation-target",
            str(decimation_target),
        ]
        if skip_tex:
            cmd.append("--skip-tex")
        if obj:
            cmd.append("--obj")
        if no_preview:
            cmd.append("--no-preview")
        if sparse_only:
            cmd.append("--sparse-only")
        if strategy == "p1_condition_fusion":
            cmd += ["--mode", p1_mode]
        if strategy == "p2_scaffold":
            cmd += [
                "--cond-fusion",
                p2_cond_fusion,
                "--elevation",
                str(p2_elevation),
            ]
            parts = [x.strip() for x in p2_azimuths_text.replace(",", " ").split() if x.strip()]
            if parts:
                cmd += ["--azimuths", *parts]
            if p2_min_views > 0:
                cmd += ["--min-views-scaffold", str(p2_min_views)]
    else:
        cmd += [
            "--output-dir",
            output_dir,
            "--pipeline",
            pipeline,
            "--seed",
            str(seed),
            "--model",
            model,
            "--texture-size",
            str(texture_size),
            "--decimation-target",
            str(decimation_target),
            "--ss-steps",
            str(ss_steps),
            "--shape-steps",
            str(shape_steps),
            "--tex-steps",
            str(tex_steps),
        ]
        if no_preprocess_multi:
            cmd.append("--no-preprocess")
        if no_preview:
            cmd.append("--no-preview")
        if strategy == "sparse_fusion":
            cmd += [
                "--fusion",
                sparse_fusion,
                "--vote-threshold",
                str(vote_threshold),
                "--logit-threshold",
                str(logit_threshold),
                "--logit-smooth-sigma",
                str(logit_smooth),
                "--samples-per-view",
                str(samples_per_view),
                "--filter-min-neighbors",
                str(filter_min_neighbors),
            ]
        else:
            cmd += [
                "--sparse-mode",
                slat_sparse_mode,
                "--mode",
                slat_mode,
                "--vote-threshold",
                str(vote_threshold),
                "--logit-threshold",
                str(logit_threshold),
                "--logit-smooth-sigma",
                str(logit_smooth),
                "--samples-per-view",
                str(samples_per_view),
                "--filter-min-neighbors",
                str(filter_min_neighbors),
            ]
    return cmd


def run_subprocess_stream(cmd: List[str]) -> Iterator[str]:
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line
    proc.wait()


def _abspath_if_file(p: Optional[str]) -> Optional[str]:
    if not p or not isinstance(p, str):
        return None
    ap = os.path.abspath(os.path.expanduser(p.strip()))
    return ap if os.path.isfile(ap) else None


def artifacts_from_run_dir(run_dir: str) -> Tuple[Optional[str], Optional[str], Optional[dict]]:
    """Load preview + 3D path from summary.json when present, else conventional names."""
    run_dir = os.path.abspath(run_dir)
    summ_path = os.path.join(run_dir, "summary.json")
    data: Optional[dict] = None
    if os.path.isfile(summ_path):
        try:
            with open(summ_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None
    prev: Optional[str] = None
    model: Optional[str] = None
    if data:
        prev = _abspath_if_file(data.get("preview_path"))
        model = _abspath_if_file(data.get("model_path"))
    if not prev:
        prev = _abspath_if_file(os.path.join(run_dir, "preview.png"))
    if not model:
        for name in ("model.glb", "model.gltf", "model.obj"):
            model = _abspath_if_file(os.path.join(run_dir, name))
            if model:
                break
    return prev, model, data


def load_result_paths(output_dir: str, strategy: str) -> Tuple[Optional[str], Optional[str], Optional[dict]]:
    d = latest_strategy_run_dir(output_dir, strategy)
    if not d:
        return None, None, None
    return artifacts_from_run_dir(d)


def main() -> None:
    import gradio as gr

    files, folders = discover_inputs(INPUT_ROOT)
    folder_names = sorted(folders.keys())

    def refresh() -> Tuple[Any, ...]:
        nf, nd = discover_inputs(INPUT_ROOT)
        nn = sorted(nd.keys())
        first_imgs = nd[nn[0]] if nn else []
        return (
            gr.Dropdown(choices=[os.path.basename(f) for f in nf], value=None),
            gr.Dropdown(choices=nn, value=nn[0] if nn else None),
            gr.Gallery(value=first_imgs, label="Folder images"),
        )

    def run_exp(
        input_mode: str,
        single_pick: Optional[str],
        folder: Optional[str],
        strategy: str,
        out_dir: str,
        model: str,
        pipeline: str,
        seed: float,
        texture_size: float,
        decimation: float,
        skip_tex: bool,
        obj: bool,
        no_preview: bool,
        sparse_only: bool,
        rembg_model: str,
        p1_mode: str,
        p2_cond: str,
        p2_el: float,
        p2_az: str,
        p2_minv: float,
        sparse_fusion: str,
        vote_thr: float,
        logit_thr: float,
        logit_sm: float,
        spv: float,
        fnb: float,
        ss_s: float,
        sh_s: float,
        tx_s: float,
        no_pp: bool,
        slat_sm: str,
        slat_m: str,
        logbox: str,
    ) -> Iterator[Tuple[str, Any, Any, Any, Any]]:
        nf, flds = discover_inputs(INPUT_ROOT)
        name_to_full = {os.path.basename(f): f for f in nf}
        imgs: List[str] = []
        if input_mode == "single":
            if not single_pick or single_pick not in name_to_full:
                yield logbox + "\nSelect a file.\n", None, None, None, None
                return
            imgs = [name_to_full[single_pick]]
        else:
            if not folder or folder not in flds:
                yield logbox + "\nSelect a folder.\n", None, None, None, None
                return
            imgs = flds[folder]

        if strategy == "baseline":
            imgs = [imgs[0]]

        if strategy in ("p1_condition_fusion", "p2_scaffold", "sparse_fusion", "slat_fusion") and len(imgs) < 2:
            yield (
                logbox + "\nThis strategy needs at least 2 images (multi folder).\n",
                None,
                None,
                None,
                None,
            )
            return

        seed_i = int(seed)
        cmd = build_command(
            strategy,
            imgs,
            out_dir,
            model=model,
            pipeline=pipeline,
            seed=seed_i,
            texture_size=int(texture_size),
            decimation_target=int(decimation),
            skip_tex=skip_tex,
            obj=obj,
            no_preview=no_preview,
            sparse_only=sparse_only,
            rembg_model=rembg_model,
            p1_mode=p1_mode,
            p2_cond_fusion=p2_cond,
            p2_elevation=float(p2_el),
            p2_azimuths_text=p2_az,
            p2_min_views=int(p2_minv),
            sparse_fusion=sparse_fusion,
            vote_threshold=float(vote_thr),
            logit_threshold=float(logit_thr),
            logit_smooth=float(logit_sm),
            samples_per_view=int(spv),
            filter_min_neighbors=int(fnb),
            ss_steps=int(ss_s),
            shape_steps=int(sh_s),
            tex_steps=int(tx_s),
            no_preprocess_multi=no_pp,
            slat_sparse_mode=slat_sm,
            slat_mode=slat_m,
        )
        log = logbox + "\n$ " + " ".join(cmd) + "\n\n"
        yield log, None, None, None, None
        try:
            for chunk in run_subprocess_stream(cmd):
                log += chunk
                yield log, gr.update(), gr.update(), gr.update(), gr.update()
        except Exception as e:
            log += f"\n[UI ERROR] {e}\n"
            yield log, None, None, None, None
            return
        prev, glb, summ = load_result_paths(out_dir, strategy)
        tbl = None
        if summ and summ.get("stage_times"):
            st = summ["stage_times"]
            tbl = [[k, st[k]] for k in sorted(st.keys())]
        info_keys = (
            "strategy",
            "mode",
            "elapsed_s",
            "voxels",
            "error",
            "run_dir",
            "model_path",
            "preview_path",
        )
        info = json.dumps(
            {k: summ[k] for k in info_keys if summ and k in summ},
            indent=2,
        ) if summ else ""
        # Clear Model3D first, then set path — avoids stale mesh when the viewer
        # does not refresh on a new file at a new path (Gradio / browser quirk).
        if glb:
            yield log, prev, None, tbl, info
        yield log, prev, glb, tbl, info

    with gr.Blocks(title="TRELLIS.2 Multi-view runs") as demo:
        gr.Markdown("# TRELLIS.2 — Multi-view strategy runner")
        gr.Markdown(
            "Runs `scripts/runs/*.py` in a **subprocess** (full VRAM release between runs). "
            "Place images under `input/` or subfolders."
        )
        with gr.Row():
            input_mode = gr.Radio(
                choices=["single", "multi"],
                value="multi",
                label="Input",
            )
            refresh_btn = gr.Button("Refresh input list")
        with gr.Row():
            single_dd = gr.Dropdown(
                choices=[os.path.basename(f) for f in files],
                label="Single file (root of input/)",
            )
            folder_dd = gr.Dropdown(choices=folder_names, label="Multi: folder", value=folder_names[0] if folder_names else None)
        folder_gal = gr.Gallery(
            value=folders[folder_names[0]] if folder_names else [],
            label="Images in folder",
            columns=4,
            height=200,
        )

        def on_folder(f: Optional[str]) -> Any:
            if f and f in folders:
                return gr.Gallery(value=folders[f])
            return gr.Gallery(value=[])

        folder_dd.change(on_folder, folder_dd, folder_gal)

        with gr.Row():
            strategy = gr.Dropdown(
                choices=[
                    "baseline",
                    "p1_condition_fusion",
                    "p2_scaffold",
                    "sparse_fusion",
                    "slat_fusion",
                ],
                value="sparse_fusion",
                label="Strategy",
            )
            out_dir = gr.Textbox(value=DEFAULT_OUT, label="Output directory")

        with gr.Accordion("Common", open=True):
            with gr.Row():
                model = gr.Textbox(value="microsoft/TRELLIS.2-4B", label="Model")
                rembg_model = gr.Textbox(value="briaai/RMBG-2.0", label="BiRefNet (P1/P2/baseline)")
            with gr.Row():
                pipeline = gr.Dropdown(
                    choices=["512", "1024", "1024_cascade", "1536_cascade"],
                    value="1024_cascade",
                )
                seed = gr.Number(value=42, label="Seed", precision=0)
            with gr.Row():
                texture_size = gr.Number(value=1024, label="Texture size", precision=0)
                decimation = gr.Number(value=200000, label="Decimation target", precision=0)
            with gr.Row():
                skip_tex = gr.Checkbox(label="skip_tex (benchmark-style only)", value=False)
                obj = gr.Checkbox(label="obj (benchmark-style only)", value=False)
                no_preview = gr.Checkbox(label="no preview", value=False)
                sparse_only = gr.Checkbox(label="sparse only (baseline/P1/P2)", value=False)

        with gr.Accordion("P1 — condition fusion", open=False):
            p1_mode = gr.Radio(choices=["mean", "concat"], value="mean", label="--mode")

        with gr.Accordion("P2 — scaffold", open=False):
            with gr.Row():
                p2_cond = gr.Dropdown(
                    choices=["primary", "mean", "concat"],
                    value="primary",
                    label="cond-fusion",
                )
                p2_el = gr.Number(value=15.0, label="elevation")
            p2_az = gr.Textbox(
                label="Azimuths (space-separated, optional)",
                placeholder="e.g. 0 90 180 270",
            )
            p2_minv = gr.Number(value=0, label="min_views_scaffold (0 = all)", precision=0)

        with gr.Accordion("Sparse / SLAT (run_multi_image)", open=False):
            sparse_fusion = gr.Dropdown(
                choices=["union", "vote", "logit-mean", "logit-max", "logit-sum"],
                value="logit-mean",
                label="sparse fusion mode",
            )
            slat_sm = gr.Dropdown(
                choices=["union", "vote", "logit-mean", "logit-max", "logit-sum"],
                value="logit-mean",
                label="slat_fusion: sparse-mode",
            )
            slat_m = gr.Dropdown(
                choices=["slat-mean", "slat-norm-weighted", "slat-max"],
                value="slat-mean",
                label="slat_fusion: --mode",
            )
            with gr.Row():
                vote_thr = gr.Number(value=0.5, label="vote threshold")
                logit_thr = gr.Number(value=0.0, label="logit threshold")
                logit_sm = gr.Number(value=0.0, label="logit smooth sigma")
            with gr.Row():
                spv = gr.Number(value=1, label="samples / view", precision=0)
                fnb = gr.Number(value=0, label="filter min neighbors", precision=0)
            with gr.Row():
                ss_s = gr.Number(value=12, label="ss steps", precision=0)
                sh_s = gr.Number(value=12, label="shape steps", precision=0)
                tx_s = gr.Number(value=12, label="tex steps", precision=0)
            no_pp = gr.Checkbox(label="no_preprocess (sparse/slat)", value=False)

        run_btn = gr.Button("Run", variant="primary")
        log_out = gr.Textbox(label="Log", lines=24, max_lines=40)

        with gr.Row():
            prev_img = gr.Image(label="preview.png")
            glb_out = gr.Model3D(label="model.glb", height=500, display_mode="solid", clear_color=(0.2, 0.2, 0.2, 1.0))
        stage_tbl = gr.Dataframe(label="stage_times", headers=["stage", "seconds"])
        summ_json = gr.Textbox(label="Summary snippet", lines=8)

        refresh_btn.click(refresh, outputs=[single_dd, folder_dd, folder_gal])

        run_btn.click(
            run_exp,
            inputs=[
                input_mode,
                single_dd,
                folder_dd,
                strategy,
                out_dir,
                model,
                pipeline,
                seed,
                texture_size,
                decimation,
                skip_tex,
                obj,
                no_preview,
                sparse_only,
                rembg_model,
                p1_mode,
                p2_cond,
                p2_el,
                p2_az,
                p2_minv,
                sparse_fusion,
                vote_thr,
                logit_thr,
                logit_sm,
                spv,
                fnb,
                ss_s,
                sh_s,
                tx_s,
                no_pp,
                slat_sm,
                slat_m,
                log_out,
            ],
            outputs=[log_out, prev_img, glb_out, stage_tbl, summ_json],
        )

        gr.Markdown(
            "## History (recent `summary.json` under output dir)\n\n"
            "Click a row to highlight it, then click **Load selected run**."
        )
        _HIST_COL_DIR = 5

        def _hist_rows_from_summary(od: str) -> list:
            rows = scan_summaries(od or DEFAULT_OUT)
            return [
                [r["started"], r["strategy"], r["mode"], r["voxels"], r["elapsed_s"], r["dir"]]
                for r in rows
            ]

        hist_btn = gr.Button("Refresh history")
        load_hist_btn = gr.Button("Load selected run", variant="secondary")

        # non-interactive so clicking fires select instead of entering edit mode
        hist_tbl = gr.Dataframe(
            headers=["started", "strategy", "mode", "voxels", "elapsed_s", "dir"],
            value=_hist_rows_from_summary(DEFAULT_OUT),
            interactive=False,
        )
        # state holds the run dir path directly — avoids re-indexing the table later
        hist_selected_dir = gr.State("")

        def on_hist_select(evt: gr.SelectData, tbl: Any) -> str:
            """Store the run dir of the clicked row into state."""
            row_idx = evt.index[0] if hasattr(evt, "index") and evt.index else None
            if row_idx is None:
                return ""
            try:
                row_idx = int(row_idx)
            except (TypeError, ValueError):
                return ""
            # tbl may arrive as list-of-lists or pandas DataFrame
            if isinstance(tbl, list):
                rows = tbl
            else:
                try:
                    rows = tbl.values.tolist()
                except Exception:
                    return ""
            if row_idx < 0 or row_idx >= len(rows):
                return ""
            row_vals = rows[row_idx]
            if len(row_vals) <= _HIST_COL_DIR:
                return ""
            return str(row_vals[_HIST_COL_DIR]).strip()

        def load_hist(od: str) -> Any:
            return gr.Dataframe(
                value=_hist_rows_from_summary(od or DEFAULT_OUT),
                headers=["started", "strategy", "mode", "voxels", "elapsed_s", "dir"],
                interactive=False,
            )

        def load_selected_run(
            run_dir: str,
            logbox: str,
        ) -> Iterator[Tuple[str, Any, Any, Any, Any]]:
            log = logbox or ""
            run_dir = (run_dir or "").strip()
            if not run_dir:
                log += "\n[History] Click a row in the table first, then Load selected run.\n"
                yield log, gr.update(), gr.update(), gr.update(), gr.update()
                return
            if not os.path.isdir(run_dir):
                log += f"\n[History] Run directory not found: {run_dir!r}\n"
                yield log, gr.update(), gr.update(), gr.update(), gr.update()
                return
            prev, model, summ = artifacts_from_run_dir(run_dir)
            tbl_out = None
            if summ and summ.get("stage_times"):
                st = summ["stage_times"]
                tbl_out = [[k, st[k]] for k in sorted(st.keys())]
            keys = ("strategy", "mode", "elapsed_s", "voxels", "error", "run_dir", "model_path", "preview_path")
            info = (
                json.dumps({k: summ[k] for k in keys if summ and k in summ}, indent=2)
                if summ
                else json.dumps({"run_dir": run_dir}, indent=2)
            )
            log += f"\n[History] Loaded {run_dir}\n"
            if model is None:
                log += "  (no mesh found — check summary model_path or model.glb in run dir)\n"
            if model:
                yield log, prev, None, tbl_out, info
            yield log, prev, model, tbl_out, info

        hist_btn.click(load_hist, inputs=[out_dir], outputs=[hist_tbl])
        hist_tbl.select(on_hist_select, inputs=[hist_tbl], outputs=[hist_selected_dir])
        load_hist_btn.click(
            load_selected_run,
            inputs=[hist_selected_dir, log_out],
            outputs=[log_out, prev_img, glb_out, stage_tbl, summ_json],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
