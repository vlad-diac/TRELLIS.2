# Multi-Image Fusion Script — `test_multi_image_fusion.py`

This document describes the pipeline paths implemented in the multi-image fusion test script. The first diagram shows the **unmodified single-image (baseline) pipeline**. Each subsequent diagram shows what changes for each fusion strategy we added.

---

## 1. Default Single-Image Pipeline (Baseline)

The standard TRELLIS.2 flow. One image enters, DINOv3 extracts patch tokens, a sparse structure diffusion model samples a voxel scaffold, two SLAT flow models fill in shape and texture, and a decoder produces the final mesh.

```mermaid
flowchart TD
    img["Input Image (1 view)"]
    pre["Preprocess\n(BiRefNet rembg + crop)"]
    cond512["get_cond @ 512px\nDINOv3 patch tokens"]
    cond1024["get_cond @ 1024px\nDINOv3 patch tokens"]

    subgraph stage1 [Stage 1 — Sparse Structure]
        ss["Sparse Structure\nFlow Model (512³ → 32³)\nDiffusion steps: 12"]
        occ["Occupancy Volume\n(32³ bool)"]
        coords["Sparse Coords\n(N × 4)"]
        ss --> occ --> coords
    end

    subgraph stage2 [Stage 2 — Shape SLAT]
        shape_flow["Shape SLAT\nFlow Model (cascade:\nLR 512 → HR 1024)\nDiffusion steps: 12"]
        shape_slat["Shape SLAT\n(SparseTensor, 32-ch)"]
        shape_flow --> shape_slat
    end

    subgraph stage3 [Stage 3 — Texture SLAT]
        tex_flow["Texture SLAT\nFlow Model @ 1024px\nDiffusion steps: 12"]
        tex_slat["Texture SLAT\n(SparseTensor, PBR attrs)"]
        tex_flow --> tex_slat
    end

    decode["Decode Latent\nshape_slat_decoder\n+ tex_slat_decoder"]
    mesh["MeshWithVoxel\n(vertices, faces, PBR atlas)"]
    glb["model.glb"]

    img --> pre --> cond512 & cond1024
    cond512 --> stage1
    cond512 --> stage2
    cond1024 --> stage2
    coords --> stage2
    cond1024 --> stage3
    shape_slat --> stage3
    shape_slat & tex_slat --> decode --> mesh --> glb
```

---

## 2. Change A — Binary Occupancy Fusion (`union` / `vote`)

The same flow model runs **once per view**. Each view independently thresholds its occupancy volume to a binary mask, then the masks are merged. Shape and texture SLAT stages still use only the **primary (first) view** for conditioning.

**What changed:** Stage 1 — the sparse structure flow model is run `N` times (once per image), each producing its own bool volume. Those are stacked and merged before coords are extracted.

```mermaid
flowchart TD
    imgs["Input Images\n(N views)"]
    pre["Preprocess each image\n(BiRefNet + crop)"]

    subgraph conds [Per-view conditioning @ 512px]
        c1["cond_v1 (DINOv3)"]
        c2["cond_v2 (DINOv3)"]
        cN["cond_vN (DINOv3)"]
    end

    subgraph stage1 [Stage 1 — Binary Occupancy Fusion]
        ss1["SS Flow → bool vol₁"]
        ss2["SS Flow → bool vol₂"]
        ssN["SS Flow → bool volₙ"]
        stack["Stack bool volumes\n(N × B × 1 × 32³)"]
        union["union: any() across views\n― OR ―\nvote: mean() ≥ threshold"]
        coords["Sparse Coords"]
        ss1 & ss2 & ssN --> stack --> union --> coords
    end

    cond1024p["Primary cond @ 1024px"]
    shape_slat["Shape SLAT (primary cond only)"]
    tex_slat["Texture SLAT (primary cond only)"]
    decode["Decode → MeshWithVoxel → GLB"]

    imgs --> pre --> conds
    conds --> stage1
    pre --> cond1024p
    coords --> shape_slat
    cond1024p --> shape_slat --> tex_slat --> decode

    style union fill:#e6f0ff
    style stack fill:#e6f0ff
```

**Flags:**
| Flag | Default | Effect |
|---|---|---|
| `--fusion union` | — | Voxel is kept if **any** view fired |
| `--fusion vote` | — | Voxel is kept if ≥ `--vote-threshold` fraction of views fired |
| `--vote-threshold` | 0.5 | Majority threshold for `vote` mode |

**Known limitation:** binary thresholding per view discards weak-evidence voxels that might be corroborated across views. This was the primary motivation for logit-level fusion below.

---

## 3. Change B — Logit-Level Occupancy Fusion (`logit-mean` / `logit-max` / `logit-sum`)

Instead of binarising each view's output, the **raw decoder logits** (float scores before the threshold step) are preserved and merged across views. A **single threshold** is applied to the fused volume. Weak-evidence voxels that each view scores just below threshold can collectively cross it.

```mermaid
flowchart TD
    imgs["Input Images (N views)"]
    pre["Preprocess each image"]

    subgraph conds [Per-view conditioning @ 512px]
        c1["cond_v1"]
        c2["cond_v2"]
        cN["cond_vN"]
    end

    subgraph stage1 [Stage 1 — Logit-Level Fusion]
        ss1["SS Flow → logits₁\n(float, 32³)"]
        ss2["SS Flow → logits₂\n(float, 32³)"]
        ssN["SS Flow → logitsₙ\n(float, 32³)"]
        stack["Stack logit volumes\n(N × B × 1 × 32³)"]
        fuse["Fuse:\nlogit-mean → mean()\nlogit-max  → max()\nlogit-sum  → sum()"]
        smooth["Optional Gaussian\nspatial smoothing\n(--logit-smooth-sigma)"]
        thr["Threshold at\n--logit-threshold (default 0.0)"]
        nb["Optional neighbour filter\n(--filter-min-neighbors K)\nremove voxels with < K occupied neighbours"]
        coords["Sparse Coords (N × 4)"]
        ss1 & ss2 & ssN --> stack --> fuse --> smooth --> thr --> nb --> coords
    end

    cond1024p["Primary cond @ 1024px"]
    shape_slat["Shape SLAT (primary cond)"]
    tex_slat["Texture SLAT (primary cond)"]
    decode["Decode → GLB"]

    imgs --> pre --> conds
    conds --> stage1
    pre --> cond1024p
    coords --> shape_slat
    cond1024p --> shape_slat --> tex_slat --> decode

    style fuse fill:#fff0cc
    style smooth fill:#fff0cc
    style thr fill:#fff0cc
    style nb fill:#fff0cc
```

**Flags:**
| Flag | Default | Effect |
|---|---|---|
| `--fusion logit-mean` | — | Mean logit across views → threshold |
| `--fusion logit-max` | — | Max logit across views → threshold |
| `--fusion logit-sum` | — | Sum of logits → threshold |
| `--logit-threshold` | `0.0` | Decision boundary after fusion; lower = more voxels |
| `--logit-smooth-sigma` | `0.0` | 3-D Gaussian blur sigma (voxels) before threshold |
| `--filter-min-neighbors` | `0` | Remove isolated voxels with < K occupied neighbours |
| `--samples-per-view` | `1` | Run diffusion N times per view and average logits (ensemble) |

---

## 4. Change C — Shape SLAT Feature Fusion (`slat-mean` / `slat-norm-weighted` / `slat-max`)

This operates **deeper in the pipeline** — at the geometry representation level rather than the occupancy scaffold level. Instead of conditioning shape SLAT generation on only the first image, the flow model runs **once per view** at 1024 px, producing a full 32-channel SLAT tensor. Those tensors are fused before decoding.

For cascade pipelines (`1024_cascade`, `1536_cascade`), the **low-resolution LR stage** still uses only the primary view (fast, cheap); only the **high-resolution HR stage** is run per-view.

```mermaid
flowchart TD
    imgs["Input Images (N views)"]
    pre["Preprocess each image"]
    coords["Sparse Coords\n(from any occupancy fusion)"]

    subgraph conds1024 [Per-view conditioning @ 1024px]
        c1["cond_v1 @ 1024px"]
        c2["cond_v2 @ 1024px"]
        cN["cond_vN @ 1024px"]
    end

    subgraph stage2 [Stage 2 — SLAT Feature Fusion]
        lr["LR Shape SLAT\n@ 512px (primary only)\ncascade pipelines only"]
        hr1["HR SLAT flow\ncond_v1 → slat₁"]
        hr2["HR SLAT flow\ncond_v2 → slat₂"]
        hrN["HR SLAT flow\ncond_vN → slatₙ"]
        fuse["Fuse SLAT feature vectors:\nslat-mean → mean(slat₁…slatₙ)\nslat-norm-weighted → weight by ‖feat‖\nslat-max → element-wise max"]
        fused_slat["Fused shape_slat\n(SparseTensor, 32-ch)"]
        lr --> hr1 & hr2 & hrN
        hr1 & hr2 & hrN --> fuse --> fused_slat
    end

    cond1024p["Primary cond @ 1024px"]
    tex_slat["Texture SLAT\n(primary cond only)"]
    decode["Decode → GLB"]

    imgs --> pre --> conds1024
    pre --> cond1024p
    coords --> stage2
    conds1024 --> stage2
    fused_slat --> tex_slat
    cond1024p --> tex_slat --> decode

    style fuse fill:#e6ffe6
    style fused_slat fill:#e6ffe6
```

**Flags:**
| Flag | Default | Effect |
|---|---|---|
| `--slat-fusion primary` | default | First-image-only conditioning (no change vs baseline) |
| `--slat-fusion slat-mean` | — | Average 32-ch SLAT features across all views |
| `--slat-fusion slat-norm-weighted` | — | Weight each view's features by their L2 norm per token |
| `--slat-fusion slat-max` | — | Element-wise maximum across views |

---

## 5. Full Multi-Stage Pipeline (All Changes Combined)

Any combination of the above can be composed. This shows the complete flow when both logit-level occupancy fusion **and** SLAT feature fusion are active.

```mermaid
flowchart TD
    imgs["Input Images (N views)"]
    pre["Preprocess\n(BiRefNet + crop, all views)"]

    subgraph cond_block [Conditioning]
        c512["Per-view cond @ 512px\n(for Stage 1)"]
        c1024["Per-view cond @ 1024px\n(for Stage 2 SLAT fusion)"]
        c1024p["Primary cond @ 1024px\n(for Texture SLAT)"]
    end

    subgraph stage1 [Stage 1 — Fused Sparse Structure]
        ss_views["SS Flow × N views\n(return logits or bool)"]
        fusion1["Logit fusion\n(mean / max / sum)\nor binary (union / vote)"]
        smooth1["Optional Gaussian\nsmoothing (sigma S)"]
        thr1["Threshold → Fused bool vol"]
        nbf["Optional neighbour filter\n(min K occupied neighbours)"]
        coords["Sparse Coords"]
        ss_views --> fusion1 --> smooth1 --> thr1 --> nbf --> coords
    end

    subgraph stage2 [Stage 2 — Fused Shape SLAT]
        lr["LR SLAT @ 512\n(primary view, cascade only)"]
        hr_views["HR SLAT Flow × N views"]
        fusion2["SLAT feature fusion\n(mean / norm-weighted / max)"]
        shape_slat["Fused shape_slat"]
        lr --> hr_views --> fusion2 --> shape_slat
    end

    subgraph stage3 [Stage 3 — Texture SLAT]
        tex_flow["Tex SLAT Flow\n(primary view conditioning)"]
        tex_slat["tex_slat"]
        tex_flow --> tex_slat
    end

    decode["Decode Latent\n→ MeshWithVoxel"]
    glb["model.glb"]

    imgs --> pre --> cond_block
    c512 --> stage1
    coords --> stage2
    c1024 --> stage2
    c1024p --> stage3
    shape_slat --> stage3
    shape_slat & tex_slat --> decode --> glb
```

---

## 6. Compare Mode (`--compare`)

When `--compare` is passed, the script runs **both** the single-image baseline and the selected multi-image fusion path and saves both GLB files side-by-side for visual inspection.

```mermaid
flowchart LR
    input["Input Images"]

    subgraph baseline [Single-image baseline]
        b1["images[0] only\npipeline.run()"]
        bglb["single_YYYYMMDD.glb"]
        b1 --> bglb
    end

    subgraph multi [Multi-image fusion]
        m1["All N images\npipeline.run_multi_image()"]
        mglb["multi_logit_mean_YYYYMMDD.glb"]
        m1 --> mglb
    end

    stats["Summary stats:\n• single voxels\n• multi voxels\n• delta %\n• wall-clock times"]

    input --> baseline & multi
    baseline & multi --> stats
```

---

## 7. Output Files

### Layout

All files are written flat into `--output-dir` (default `./out`). There are no sub-folders — each run appends timestamped GLB files to the same directory.

```
{output_dir}/
├── single_2026-05-11T113045.123.glb          # baseline (only with --compare or single-image input)
├── multi_logit_mean_2026-05-11T113412.456.glb
├── multi_logit_mean_sm0.8_2026-05-11T113700.789.glb
└── multi_logit_mean_sm0.8_slat_mean_2026-05-11T114002.111.glb
```

The directory is created automatically if it does not exist.

### File Naming Rules

Every output file is named `{label}_{YYYY-MM-DDTHHMMSS}.{ms}.glb`.

**Baseline file (`--compare` or single image):**

```
single_{timestamp}.glb
```

**Multi-image fusion file:**

The label is assembled from the active flags in order:

| Part | Condition | Example |
|---|---|---|
| `multi_{fusion}` | always | `multi_logit_mean` |
| `thr{value}` | `--logit-threshold` ≠ 0.0 | `thr-0.5` |
| `sm{value}` | `--logit-smooth-sigma` > 0.0 | `sm0.8` |
| `spv{N}` | `--samples-per-view` > 1 | `spv3` |
| `fn{K}` | `--filter-min-neighbors` > 0 | `fn2` |
| `{slat_fusion}` | `--slat-fusion` ≠ `primary` | `slat_mean` |

Parts are joined with `_`. Examples:

```
multi_union_2026-05-11T110000.000.glb
multi_logit_mean_2026-05-11T110000.000.glb
multi_logit_mean_sm0.8_fn2_2026-05-11T110000.000.glb
multi_logit_mean_sm0.8_spv3_fn2_slat_norm_weighted_2026-05-11T110000.000.glb
```

### GLB Contents

Each `.glb` file contains:
- A single decimated mesh (target: `--decimation` faces, default 200 000)
- A baked PBR texture atlas (`--texture-size` px, default 1024)
- Base colour, metallic, roughness, and normal maps packed into WebP-compressed textures

### When files are written

```mermaid
flowchart LR
    start["Script starts"]
    pp["Preprocess images"]
    bl{{"--compare\nor 1 image?"}}
    baseline["Run baseline\n→ single_{ts}.glb"]
    multi_check{{"N > 1 images?"}}
    fusion["Run multi-image fusion\n→ multi_{label}_{ts}.glb"]
    summary["Print summary stats\nto stdout (no file)"]

    start --> pp --> bl
    bl -- yes --> baseline --> multi_check
    bl -- no --> multi_check
    multi_check -- yes --> fusion --> summary
    multi_check -- no --> summary
```

No timestamped flat GLBs are written by the dispatcher; instead each underlying run creates a folder `out/<strategy>_YYYYMMDD_HHMMSS/` with `model.glb`, `preview.png`, and `summary.json`. The summary (voxel counts, timing) is in `summary.json`; logs still go to stdout.

---

## Run topology (split scripts + UI)

The implementation is split so **each strategy runs in its own process** (full RAM/VRAM release between runs):

| Strategy | Script |
|----------|--------|
| Baseline (single image) | [`scripts/runs/baseline.py`](../scripts/runs/baseline.py) |
| P1 (`mean` / `concat` conditioning) | [`scripts/runs/p1_condition_fusion.py`](../scripts/runs/p1_condition_fusion.py) |
| P2 visual hull + SLAT | [`scripts/runs/p2_scaffold.py`](../scripts/runs/p2_scaffold.py) |
| Sparse fusion only (`primary` SLAT) | [`scripts/runs/sparse_fusion.py`](../scripts/runs/sparse_fusion.py) |
| Sparse + SLAT feature fusion | [`scripts/runs/slat_fusion.py`](../scripts/runs/slat_fusion.py) |

Shared helpers: [`scripts/runs/_common.py`](../scripts/runs/_common.py).

Legacy entrypoint [`scripts/test_multi_image_fusion.py`](../scripts/test_multi_image_fusion.py) shells out to the scripts above (CLI unchanged).

**Visualizer:** [`app_multiview.py`](../app_multiview.py) — Gradio UI that scans `./input`, runs a chosen strategy via subprocess, streams logs, and loads the latest `preview.png` / `model.glb` / `summary.json`.


```
# Baseline (single image)
python scripts/test_multi_image_fusion.py front.png --output-dir ./out

# Union / vote (binary)
python scripts/test_multi_image_fusion.py front.png side.png rear.png \
    --fusion union --output-dir ./out

python scripts/test_multi_image_fusion.py front.png side.png rear.png \
    --fusion vote --vote-threshold 0.4 --output-dir ./out

# Logit-level fusion + smoothing + neighbour filter
python scripts/test_multi_image_fusion.py front.png side.png rear.png \
    --fusion logit-mean --logit-smooth-sigma 0.8 \
    --filter-min-neighbors 2 --output-dir ./out

# SLAT feature fusion added on top
python scripts/test_multi_image_fusion.py front.png side.png rear.png \
    --fusion logit-mean --slat-fusion slat-mean --output-dir ./out

# Full combo + compare vs baseline
python scripts/test_multi_image_fusion.py front.png side.png rear.png \
    --fusion logit-mean --logit-smooth-sigma 0.8 \
    --slat-fusion slat-norm-weighted --compare --output-dir ./out
```
