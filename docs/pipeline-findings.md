# TRELLIS.2 — Pipeline Deep-Dive: Code-Grounded Findings

## Collaboration Table

| Note | Info | Status | Comments |
|------|------|--------|----------|
| **Initial thoughts** | TRELLIS.2 uses a sparse structured latent (SLAT) representation that separates *where* geometry exists (sparse occupancy) from *what* geometry exists (latent features). Both stages operate in 3D space with explicit voxel coordinates, making them natural targets for multi-view evidence injection. The O-Voxel format preserves thin structures (masts, antennas, railings) that collapsed in previous pipelines. The texture pipeline is decoupled from geometry, so shape improvements cascade automatically. The existing single-image flow models can be used as frozen backbones while new fusion layers are added around the sparse structure stage — no full retraining required for initial experiments. | 🟡 In progress | Architecture confirmed code-first. Pipeline is modular enough that each improvement strategy can be prototyped independently. |
| **Multi-image sparse occupancy fusion** | Run the sparse structure flow model once per input image to get per-view occupancy predictions, then fuse them (union / weighted vote) before passing `coords` to the Shape SLAT stage. Each view contributes spatial confidence for its visible side. Hook point: `sample_sparse_structure()` in `trellis2_image_to_3d.py`. | 🟡 In progress | Implemented and tested. **Key finding:** the model has no camera awareness — each view independently predicts a *complete* canonical object, not a partial view. Union (~18K voxels) gives good coverage but accumulates per-view hallucinations. Vote ≥0.5 collapses to ~1K voxels because independent runs rarely agree on the same voxels, even for real geometry. Two mitigations added: (1) **intra-view ensemble** (`--samples-per-view N`) — average raw logits over N runs before thresholding; (2) **neighbourhood filter** (`--filter-min-neighbors K`) — drop voxels with fewer than K occupied neighbours via 3D conv, pruning isolated islands while preserving contiguous surfaces. Recommended starting point: union + `--filter-min-neighbors 2`. |
| **Depth-guided occupancy initialization** | Use a monocular depth model (Depth Anything V2, MoGe, or UniDepth) to back-project a depth map into 3D space and bias the sparse structure sampler noise at known-occupied voxels, or use the depth-derived point cloud to directly seed `coords`. Hook point: noise initialisation inside `sample_sparse_structure()`. | 🔵 Planned | Second-highest leverage. Particularly useful for hull thickness and mast depth accuracy. Depth models can be run off-the-shelf with no TRELLIS retraining. |
| **Ship segmentation + symmetry priors** | Segment the vessel from background (or per-part: hull, superstructure, masts) and enforce port/starboard symmetry in the sparse coordinate set by mirroring occupied voxels across the centre plane before SLAT generation. Hook point: post-processing of `coords` returned by `sample_sparse_structure()`. | 🔵 Planned | Third-highest leverage. Ships have strong bilateral symmetry. Mirroring coords is a zero-cost prior that directly halves the hallucination problem on the hidden side. |
| **Multi-image early feature fusion (baseline)** | Pass multiple images into `get_cond()` and average or attention-pool their DINOv3 patch token sequences into a single conditioning tensor. The rest of the pipeline runs unchanged. Hook point: `get_cond()` in `trellis2_image_to_3d.py`. | 🔵 Planned | Weakest geometry improvement but easiest to implement. Useful as a baseline to measure the gain from deeper fusion strategies. |
| **SLAT fusion from multiple views** | Generate a partial Shape SLAT per input view independently, then merge the per-view `SparseTensor` feature maps at matching coordinates (average, max, or learned attention). Hook point: after `sample_shape_slat()`, before `decode_shape_slat()`. Uses `sparse_cat(..., dim=-1)` and a learned merge layer. | 🟠 Exploratory | Much harder than occupancy fusion — requires training or fine-tuning the merge layer. Potentially the most powerful strategy if it works. Attempt after occupancy fusion is validated. |

---

This document walks through every section of `research-topics.md` and anchors each
concept to the actual source code. All file paths are relative to the repo root.

---

## Overview — TRELLIS.2 Is Not a Mesh Generator

The research document's opening claim is exactly right. The pipeline class makes this
explicit through its name-space of models:

```python
# trellis2/pipelines/trellis2_image_to_3d.py  (lines 31-40)
model_names_to_load = [
    'sparse_structure_flow_model',   # Stage 2 — where geometry IS
    'sparse_structure_decoder',
    'shape_slat_flow_model_512',     # Stage 3 — shape latent diffusion
    'shape_slat_flow_model_1024',
    'shape_slat_decoder',
    'tex_slat_flow_model_512',       # Stage 4 — texture latent diffusion
    'tex_slat_flow_model_1024',
    'tex_slat_decoder',
]
```

Eight distinct models. A mesh generator would need one. The mesh is only produced at the
very end by the `shape_slat_decoder`; everything before that is latent-space reasoning.

---

## High-Level Pipeline

The `run()` method is the authoritative source of the pipeline order:

```python
# trellis2/pipelines/trellis2_image_to_3d.py  (lines 536-595)
if preprocess_image:
    image = self.preprocess_image(image)          # 0. Background removal + crop
torch.manual_seed(seed)
cond_512  = self.get_cond([image], 512)           # 1. Image conditioning @ 512px
cond_1024 = self.get_cond([image], 1024)          # 1. Image conditioning @ 1024px

coords = self.sample_sparse_structure(            # 2. Sparse occupancy prediction
    cond_512, ss_res, num_samples, ...
)

shape_slat, res = self.sample_shape_slat_cascade( # 3. Shape SLAT generation
    cond_512, cond_1024, ...
)
tex_slat = self.sample_tex_slat(                  # 4. Texture SLAT generation
    cond_1024, ..., shape_slat, ...
)
out_mesh = self.decode_latent(shape_slat, tex_slat, res)  # 5. Decode to mesh
```

The default mode (`1024_cascade`) uses a two-stage cascade: first sample at 512-resolution
to get a coarse shape, then upsample and re-sample at 1024 for fine detail.

---

## Stage 0 — Image Preprocessing

Before any 3D reasoning, the pipeline removes the background and crops tightly around
the subject:

```python
# trellis2/pipelines/trellis2_image_to_3d.py  (lines 127-162)
def preprocess_image(self, input: Image.Image) -> Image.Image:
    # Background removal via BiRefNet rembg model
    output = self.rembg_model(input)

    # Tight crop around the alpha mask
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    output = output.crop(bbox)

    # Premultiply alpha (white background → transparent)
    output = output[:, :, :3] * output[:, :, 3:4]
```

This is important for ship reconstruction: if the rembg model mis-segments the vessel
(e.g., keeps sea spray or cuts off masts), the quality of every subsequent stage
degrades directly.

---

## Stage 1 — Image Conditioning

The conditioning extractor turns a PIL image into a sequence of patch tokens that
every downstream flow model attends to as cross-attention context.

### What model is used?

```python
# trellis2/modules/image_feature_extractor.py  (lines 59-118)
class DinoV3FeatureExtractor:
    def __init__(self, model_name: str, image_size=512):
        self.model = DINOv3ViTModel.from_pretrained(model_name)
        ...
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)
        for layer_module in self.model.layer:
            hidden_states = layer_module(hidden_states,
                                         position_embeddings=position_embeddings)
        return F.layer_norm(hidden_states, hidden_states.shape[-1:])
```

TRELLIS.2 upgraded from DINOv2 (used in the original TRELLIS) to DINOv3, with RoPE
positional embeddings instead of absolute ones. The output shape is `(B, N, D)` where
N is the number of image patches and D is the hidden dimension.

### How conditioning is built

```python
# trellis2/pipelines/trellis2_image_to_3d.py  (lines 164-186)
def get_cond(self, image, resolution, include_neg_cond=True):
    self.image_cond_model.image_size = resolution
    cond = self.image_cond_model(image)           # (B, N_patches, D)
    neg_cond = torch.zeros_like(cond)             # empty = unconditional
    return {'cond': cond, 'neg_cond': neg_cond}
```

Two resolutions are extracted: 512px for the sparse-structure and low-resolution SLAT
models, 1024px for the high-resolution SLAT model. The negative conditioning is simply
a zero tensor, enabling standard classifier-free guidance (CFG).

**Key insight for multi-image work**: `get_cond` currently takes `[image]` — a list of
one image. It already accepts a list. Passing multiple images here would average their
patch tokens in the transformer, but without spatial awareness. That is Strategy A
(weakest) from the research document.

---

## Stage 2 — Sparse Structure Generation

This stage answers *where* in 3D space geometry exists. It is a **dense** 3D diffusion
problem solved on a coarse grid, not yet sparse.

### The flow model architecture

```python
# trellis2/models/sparse_structure_flow.py  (lines 56-140)
class SparseStructureFlowModel(nn.Module):
    def __init__(self, resolution, in_channels, model_channels,
                 cond_channels, out_channels, num_blocks, ...):
        ...
        # Positional embeddings over a regular 3D grid
        pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
        coords = torch.meshgrid(*[torch.arange(resolution)] * 3, indexing='ij')
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)
        pos_emb = pos_embedder(coords)
        self.register_buffer("pos_emb", pos_emb)

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(model_channels, cond_channels, ...)
            for _ in range(num_blocks)
        ])
        self.out_layer = nn.Linear(model_channels, out_channels)
```

The resolution is 32 (for `1024_cascade` mode) producing a `32³` = 32 768 voxel grid.
Every voxel attends to every other voxel (full 3D attention) and cross-attends to the
image patch tokens.

### Forward pass

```python
# trellis2/models/sparse_structure_flow.py  (lines 224-247)
def forward(self, x, t, cond):
    # x: (B, C, 32, 32, 32) — noisy occupancy latent
    h = x.view(*x.shape[:2], -1).permute(0, 2, 1)  # flatten spatial → (B, 32³, C)
    h = self.input_layer(h)
    h = h + self.pos_emb[None]                       # add 3D position embeddings
    t_emb = self.t_embedder(t)                        # sinusoidal timestep → MLP
    for block in self.blocks:
        h = block(h, t_emb, cond)                     # self-attn + cross-attn to image
    h = self.out_layer(h)
    h = h.permute(0, 2, 1).view(B, C, 32, 32, 32)    # back to volumetric
    return h
```

### Sampling and thresholding to get coordinates

```python
# trellis2/pipelines/trellis2_image_to_3d.py  (lines 188-235)
def sample_sparse_structure(self, cond, resolution, num_samples, sampler_params):
    flow_model = self.models['sparse_structure_flow_model']
    noise = torch.randn(num_samples, in_channels, reso, reso, reso)

    z_s = self.sparse_structure_sampler.sample(flow_model, noise, **cond, ...)

    # Decode latent → occupancy logits → threshold
    decoded = self.models['sparse_structure_decoder'](z_s) > 0
    # Downsample to desired resolution if needed
    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
    return coords   # shape: (N_occupied, 4)  — [batch, x, y, z]
```

`coords` is the critical output: a list of `(batch_idx, x, y, z)` integer voxels that
are predicted occupied. This becomes the *support set* for all subsequent sparse
operations. If a region is missing here, it **cannot** be recovered later.

**Key insight**: This is where multi-image fusion has the highest leverage. Depth maps
or frontal/lateral views could each independently vote on occupancy, and the votes could
be fused before or during this diffusion step.

---

## Stage 3 — Shape SLAT Generation

Given the sparse coordinates, this stage assigns a learned latent feature vector to
every occupied voxel via diffusion over a `SparseTensor`.

### The SparseTensor — what it actually is

```python
# trellis2/modules/sparse/basic.py  (lines 343-360)
class SparseTensor(VarLenTensor):
    """
    Sparse tensor — coords: (N, 4) [batch, x, y, z]
                    feats:  (N, C) latent vectors
    NOTE: Data corresponding to the same batch must be contiguous.
    Coords must be in [0, 1023].
    """
```

Concretely, a Shape SLAT is:

```python
SparseTensor(
    coords=torch.tensor([[0, 12, 8, 45], [0, 12, 8, 46], ...]),   # (N, 4)
    feats =torch.randn(N, C),                                       # (N, C)
)
```

Where N might be ~10 000–50 000 occupied voxels and C is the latent channel dimension
(likely 8–32). The **coordinates are fixed** throughout SLAT diffusion; only `feats`
change at each denoising step.

### The SLAT flow model

```python
# trellis2/models/structured_latent_flow.py  (lines 15-82)
class SLatFlowModel(nn.Module):
    def __init__(self, resolution, in_channels, model_channels,
                 cond_channels, out_channels, num_blocks, ...):
        ...
        self.input_layer  = sp.SparseLinear(in_channels, model_channels)
        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels, cond_channels,
                attn_mode='full', use_rope=(pe_mode=="rope"), ...
            )
            for _ in range(num_blocks)
        ])
        self.out_layer = sp.SparseLinear(model_channels, out_channels)
```

Every operation (`SparseLinear`, `SparseConv3d`, sparse attention) is aware of the
sparse coordinate structure — only occupied voxels are computed.

### Forward pass

```python
# trellis2/models/structured_latent_flow.py  (lines 169-199)
def forward(self, x: SparseTensor, t, cond, concat_cond=None):
    if concat_cond is not None:
        x = sp.sparse_cat([x, concat_cond], dim=-1)  # for texture: append shape feats
    h = self.input_layer(x)                           # project features up
    t_emb = self.t_embedder(t)                        # timestep embedding
    if self.pe_mode == "ape":
        pe = self.pos_embedder(h.coords[:, 1:])       # positional embed from coords
        h = h + pe
    for block in self.blocks:
        h = block(h, t_emb, cond)                     # sparse self-attn + cross-attn
    h = self.out_layer(h)
    return h                                           # SparseTensor, same coords
```

The image conditioning `cond` is a `VarLenTensor` of patch tokens. Each sparse voxel
token cross-attends to all image patch tokens at every layer.

### Sampling and denormalization

```python
# trellis2/pipelines/trellis2_image_to_3d.py  (lines 237-275)
def sample_shape_slat(self, cond, flow_model, coords, sampler_params):
    noise = SparseTensor(
        feats=torch.randn(coords.shape[0], flow_model.in_channels),
        coords=coords,
    )
    slat = self.shape_slat_sampler.sample(flow_model, noise, **cond, ...)
    
    # Denormalize from training distribution
    std  = torch.tensor(self.shape_slat_normalization['std'])[None]
    mean = torch.tensor(self.shape_slat_normalization['mean'])[None]
    slat = slat * std + mean
    return slat
```

### The cascade: 512 → 1024

```python
# trellis2/pipelines/trellis2_image_to_3d.py  (lines 277-364)
def sample_shape_slat_cascade(self, lr_cond, cond, flow_model_lr, flow_model,
                               lr_resolution, resolution, coords, ...):
    # 1. Generate coarse SLAT at 512 resolution
    slat = self.shape_slat_sampler.sample(flow_model_lr, noise, **lr_cond, ...)
    slat = slat * std + mean  # denormalize

    # 2. Use decoder to predict finer subdivision coordinates
    hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)

    # 3. Quantize to target grid resolution
    quant_coords = torch.cat([
        hr_coords[:, :1],
        ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
    ], dim=1)
    coords = quant_coords.unique(dim=0)  # de-duplicate

    # 4. Re-sample at high resolution with finer coordinate set
    slat = self.shape_slat_sampler.sample(flow_model, noise_hr, **cond, ...)
    slat = slat * std + mean
    return slat, hr_resolution
```

The decoder's `upsample()` method runs partial decoding to predict where to split
coarse voxels, generating a denser coordinate set. The high-res flow model then fills
those coordinates with refined feature vectors.

---

## What a Shape SLAT Token Encodes

The research document speculates about token content. Here is the concrete evidence:

The `SparseUnetVaeEncoder` maps **mesh geometry** (voxelized via O-Voxel) to a latent
space:

```python
# trellis2/models/sc_vaes/sparse_unet_vae.py  (lines 297-395)
class SparseUnetVaeEncoder(nn.Module):
    def forward(self, x: sp.SparseTensor, sample_posterior=False):
        h = self.input_layer(x)
        for res_level in self.blocks:
            for block in res_level:
                h = block(h)          # SparseResBlocks with SparseConv3d
        h = self.to_latent(h)         # project to 2*latent_channels
        mean, logvar = h.feats.chunk(2, dim=-1)
        z = mean  # or reparameterized sample
        return h.replace(z)
```

The training objective for the shape VAE (`trellis2/trainers/vae/shape_vae.py`) is to
reconstruct the original O-Voxel geometry from the latent. Through this,  each latent
vector at coordinate `(x, y, z)` must encode everything the decoder needs to reconstruct
local geometry — surface orientation, curvature, local connectivity. This confirms the
research document's table of encoded information.

The `SparseUnetVaeDecoder` uses **predicted subdivision** (`pred_subdiv=True`) at each
upsampling step:

```python
# trellis2/models/sc_vaes/sparse_unet_vae.py  (lines 488-507)
def forward(self, x, guide_subs=None, return_subs=True):
    h = self.from_latent(x)
    subs = []
    for i, res in enumerate(self.blocks):
        for j, block in enumerate(res):
            if i < len(self.blocks)-1 and j == len(res)-1:
                h, sub = block(h)   # upsample + predict subdivision mask
                subs.append(sub)
            else:
                h = block(h)
    h = self.output_layer(h)
    return h, subs  # decoded geometry + subdivision hints per scale
```

The subdivision mask `subs` is passed to the texture decoder as `guide_subs` — this is
how shape and texture decoding share geometric structure.

---

## Stage 4 — Texture SLAT Generation

Texture generation is explicitly conditioned on the shape latent:

```python
# trellis2/pipelines/trellis2_image_to_3d.py  (lines 391-432)
def sample_tex_slat(self, cond, flow_model, shape_slat, sampler_params):
    # Normalize shape_slat back to diffusion space
    shape_slat_norm = (shape_slat - mean) / std

    # Noise only the texture channels; shape channels are concatenated as condition
    in_channels = flow_model.in_channels
    noise = shape_slat.replace(
        feats=torch.randn(N, in_channels - shape_slat.feats.shape[1])
    )
    slat = self.tex_slat_sampler.sample(
        flow_model, noise,
        concat_cond=shape_slat_norm,   # <-- shape features appended to every token
        **cond, ...
    )
```

Inside `SLatFlowModel.forward()`:

```python
# trellis2/models/structured_latent_flow.py  (line 177-178)
if concat_cond is not None:
    x = sp.sparse_cat([x, concat_cond], dim=-1)  # tex_noise ‖ shape_feats per token
```

Each texture token has the shape latent for the same voxel concatenated directly. This
means the texture model **knows the geometry** at every voxel before generating colour
and material information. Improving shape quality therefore directly improves texture.

---

## Stage 5 — Decoding Shape SLAT to Mesh

The `shape_slat_decoder` (`SparseUnetVaeDecoder`) converts the sparse latent back to
a volumetric representation:

```python
# trellis2/pipelines/trellis2_image_to_3d.py  (lines 366-389)
def decode_shape_slat(self, slat, resolution):
    self.models['shape_slat_decoder'].set_resolution(resolution)
    meshes, subs = self.models['shape_slat_decoder'](slat, return_subs=True)
    return meshes, subs   # subs = subdivision hints per scale
```

The mesh extraction uses the **Flexible Dual Grid** algorithm from `o-voxel`:

```python
# o-voxel/o_voxel/convert/flexible_dual_grid.py  (lines 142-283)
def flexible_dual_grid_to_mesh(coords, dual_vertices, intersected_flag,
                                split_weight, aabb, ...):
    # For each surface-crossing edge, form a quad from 4 adjacent voxel centers
    edge_neighbor_voxel = coords + edge_neighbor_voxel_offset   # (N, 3, 4, 3)
    connected_voxel_indices = hashmap_lookup(...)                 # (L, 4)
    # Split each quad into 2 triangles, choosing the diagonal that aligns normals
    mesh_triangles = torch.where(align0 > align1,
                                  atempt_triangles_0,
                                  atempt_triangles_1).reshape(-1, 3)
    return mesh_vertices, mesh_triangles
```

This is a variant of Dual Marching Cubes (DMC): instead of placing iso-surface vertices
at cube edges, it places them at **dual voxel positions** optimised via a Quadric Error
Function (QEF). This gives sharper edges and better thin-structure preservation than
standard Marching Cubes.

---

## Stage 6 — Texture Decoding

```python
# trellis2/pipelines/trellis2_image_to_3d.py  (lines 434-453)
def decode_tex_slat(self, slat, subs):
    ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
    return ret   # SparseTensor with PBR material feats
```

The texture decoder's `guide_subs` are the subdivision masks from the shape decoder —
so the texture voxels follow exactly the same spatial topology as the geometry voxels.

### PBR attribute layout

```python
# trellis2/pipelines/trellis2_image_to_3d.py  (lines 73-78)
self.pbr_attr_layout = {
    'base_color': slice(0, 3),   # RGB
    'metallic':   slice(3, 4),
    'roughness':  slice(4, 5),
    'alpha':      slice(5, 6),
}
```

Each texture voxel stores 6 channels: base colour (3) + metallic + roughness + alpha.

---

## The SparseTensor Data Structure — Deep Dive

The `SparseTensor` is the backbone of all SLAT operations. Understanding it is essential
for any extension work.

```python
# trellis2/modules/sparse/basic.py  (lines 343-450)
class SparseTensor(VarLenTensor):
    # Supports torchsparse, spconv, or plain-dict backends
    # coords: (N, 4) int32 — [batch_idx, x, y, z]
    # feats:  (N, C) float  — latent vectors

    def replace(self, feats, coords=None) -> 'SparseTensor':
        # Create new SparseTensor with different feats but same coords
        ...

    def to_dense(self) -> torch.Tensor:
        # Scatter sparse feats into dense (B, C, X, Y, Z) tensor
        ret = torch.zeros(*self.shape, *spatial_shape)
        ret[batch_idx, :, x, y, z] = feats
        return ret
```

For **SLAT editing** (section in research doc):

```python
# Direct feature manipulation — valid because coords are explicit integers
slat.feats[mask] = edited_features
# or
slat = slat.replace(feats=new_feats)
```

For **multi-view SLAT fusion**, two SparseTensors with the same coordinates can be
combined:

```python
# trellis2/modules/sparse/basic.py  (lines 797-821)
def sparse_cat(inputs: List[SparseTensor], dim=0) -> SparseTensor:
    # dim=0: batch concatenation
    # dim=-1: feature concatenation (same coords)
    feats = torch.cat([inp.feats for inp in inputs], dim=dim)
    output = inputs[0].replace(feats)
    return output
```

---

## Sparse Transformer Architecture

### Self-attention on sparse tokens

```python
# trellis2/modules/sparse/attention/modules.py  (lines 99-141)
def forward(self, x: SparseTensor, context=None):
    if self._type == "self":
        qkv = self.to_qkv(x)       # project each sparse token
        if self.attn_mode == "full":
            h = sparse_scaled_dot_product_attention(qkv)
        elif self.attn_mode == "windowed":
            h = sparse_windowed_scaled_dot_product_self_attention(
                qkv, self.window_size, shift_window=self.shift_window
            )
    else:  # cross-attention
        q  = self.to_q(x)          # sparse tokens query
        kv = self.to_kv(context)   # image patch tokens are keys/values
        h  = sparse_scaled_dot_product_attention(q, kv)
```

The `"windowed"` attention mode is used in the VAE encoder/decoder for efficiency —
each voxel only attends to spatially nearby voxels. The flow model uses `"full"`
attention since long-range geometric reasoning is needed (e.g., mast ↔ hull alignment).

The support for `"double_windowed"` (half the heads use a shifted window) is a
direct implementation of the Swin Transformer idea in 3D sparse space — it allows
attention across window boundaries without full quadratic complexity.

---

## O-Voxel (Omni-Voxel) Representation

O-Voxel is a custom voxel format in `o-voxel/` that bridges mesh geometry and sparse
voxel operations.

### Spatial ordering

```python
# o-voxel/o_voxel/serialize.py
def encode_seq(coords, permute=[0,1,2], mode='z_order'):
    """Encode 3D coords into a 30-bit Z-order (Morton) or Hilbert code."""
    return _C.z_order_encode_cuda(x, y, z)
```

Z-order (Morton) encoding maps 3D coordinates to 1D such that spatially close voxels
remain close in memory. This is critical for:
- Cache-efficient sparse convolution
- Locality in windowed attention
- Efficient serialisation to `.vxz` files

### Mesh → O-Voxel → SLAT training data

The data pipeline:

```python
# data_toolkit/voxelize_pbr.py — not shown but exists
# Calls: mesh_to_flexible_dual_grid()

# o-voxel/o_voxel/convert/flexible_dual_grid.py  (lines 29-139)
def mesh_to_flexible_dual_grid(vertices, faces, grid_size=512, ...):
    # Voxelise mesh surface, placing a dual vertex inside each surface voxel
    # Uses QEF (Quadric Error Function) to optimally position dual vertices
    ret = _C.mesh_to_flexible_dual_grid_cpu(vertices, faces, ...)
    return occupied_coords, dual_vertices, intersected_flag
```

This is the inverse of mesh extraction — it converts a training mesh into the O-Voxel
representation that the VAE encoder is trained on.

---

## VAE-Style Latent Compression

The Shape SLAT is the output of a trained VAE encoder. During **training**, meshes are
encoded:

```python
# trellis2/trainers/vae/shape_vae.py (schema, not shown in full)
# Training objective:
#   L = reconstruction_loss(decoder(encoder(x)), x) + kl_weight * KL(q(z|x) || N(0,1))

# Encoder (SparseUnetVaeEncoder):
z, mean, logvar = encoder(x_sparse, sample_posterior=True, return_raw=True)
kl_loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).mean()
```

During **inference**, the diffusion model generates samples directly in this latent space
without needing the encoder — this is the standard LDM (Latent Diffusion Model) pattern.

The normalization statistics stored in the pipeline config:

```python
self.shape_slat_normalization = {'mean': [...], 'std': [...]}
```

are used to standardize latents before diffusion and de-standardize after:

```python
slat = slat * std + mean   # denormalize after sampling
shape_slat_norm = (shape_slat - mean) / std  # re-normalize for texture model
```

---

## Flow Matching — The Diffusion Algorithm

TRELLIS.2 uses **Flow Matching** (not DDPM) for all generative stages:

```python
# trellis2/pipelines/samplers/flow_euler.py  (lines 53-81)
def sample_once(self, model, x_t, t, t_prev, cond, **kwargs):
    """Euler step: x_{t-1} = x_t - (t - t_prev) * v_pred"""
    pred_v = model(x_t, t, cond, **kwargs)   # predict velocity field
    pred_x_0, pred_eps = self._v_to_xstart_eps(x_t, t, pred_v)
    pred_x_prev = x_t - (t - t_prev) * pred_v
    return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

def sample(self, model, noise, cond, steps=50, ...):
    t_seq = np.linspace(1, 0, steps + 1)   # t: 1 → 0 (noise → clean)
    for t, t_prev in t_pairs:
        out = self.sample_once(model, sample, t, t_prev, cond)
        sample = out.pred_x_prev
```

The velocity parameterization `v` maps a straight path from noise at `t=1` to clean
data at `t=0`, which is more training-stable than score-based DDPM.

Classifier-free guidance is mixed in:

```python
# trellis2/pipelines/samplers/classifier_free_guidance_mixin.py
pred_v_cond   = model(x_t, t, cond, **kwargs)
pred_v_uncond = model(x_t, t, neg_cond, **kwargs)
pred_v = pred_v_uncond + guidance_strength * (pred_v_cond - pred_v_uncond)
```

---

## Decoding Is Representation-Agnostic

The research document correctly identifies that SLAT decodes into different outputs.
The code shows this directly:

- `shape_slat_decoder` (`SparseUnetVaeDecoder`) outputs a `SparseTensor` of O-Voxel
  dual-vertex positions → then `flexible_dual_grid_to_mesh()` extracts triangles
- `tex_slat_decoder` outputs a `SparseTensor` of PBR material attributes
- The same SLAT could in principle be decoded by a different decoder (Gaussian, NeRF)
  since the latent structure is independent of the output format

---

## Shape SLAT vs Texture SLAT — Separation of Concerns

```
shape_slat = sample_shape_slat(cond_512, flow_model_512, coords)
tex_slat   = sample_tex_slat(cond_1024, flow_model_1024, shape_slat)
                                                         ^^^^^^^^^^^
                                                         depends on shape
```

The dependency is one-directional. This means:
1. Shape can be improved independently without retraining texture
2. Any improvement to shape quality propagates automatically to texture
3. Texture generation has access to full geometric context per voxel

---

## Multi-Image Extension — Where to Hook In

Based on the code, here are the precise hook points for the four strategies:

### Strategy A — Early Feature Fusion (weakest)
```python
# Hook into get_cond()
def get_cond_multi(self, images, resolution):
    # images: list of PIL Images from different viewpoints
    conds = [self.image_cond_model([img]) for img in images]
    # Average or attention-pool patch token sequences
    cond = torch.stack(conds).mean(0)   # naive average
    neg_cond = torch.zeros_like(cond)
    return {'cond': cond, 'neg_cond': neg_cond}
```

### Strategy B — Sparse Occupancy Fusion (best)
```python
# Hook into sample_sparse_structure()
def sample_sparse_structure_multi(self, conds_per_view, resolution, ...):
    # Generate separate occupancy predictions per view
    occupancies = []
    for cond in conds_per_view:
        z_s = self.sparse_structure_sampler.sample(flow_model, noise, **cond)
        decoded = decoder(z_s) > 0
        occupancies.append(decoded)
    
    # Fuse: union (conservative) or weighted vote
    fused = torch.stack(occupancies).any(dim=0)   # union
    # OR: fused = torch.stack(occupancies).float().mean(0) > threshold  # vote
    coords = torch.argwhere(fused)[:, [0,2,3,4]].int()
    return coords
```

### Strategy C — SLAT Fusion
```python
# Generate partial SLATs per view, fuse before decode
slat_front  = sample_shape_slat(cond_front, ...)
slat_rear   = sample_shape_slat(cond_rear, ...)

# Merge: since coords are explicit, fuse matching coordinates
shared_mask = ...  # coords in both tensors
slat_fused_feats = (slat_front.feats + slat_rear.feats) / 2
slat_fused = slat_front.replace(feats=slat_fused_feats)
```

### Strategy D — Geometry Constraint Injection
```python
# Inject depth-derived occupancy as a prior
depth = depth_model(image)   # e.g. Depth Anything V2
depth_coords = depth_to_voxel_coords(depth, camera_params)

# Use depth_coords to bias the sparse structure sampler
# (e.g. mask out noise at known-occupied voxels)
noise[known_occupied] = 0.0   # force start close to GT
```

---

## Marine Vessel Reconstruction — Specific Implications

### Why TRELLIS.2 is suited to ships

| Ship feature          | Relevant mechanism                                              |
|-----------------------|-----------------------------------------------------------------|
| Hull (large surface)  | Full 3D attention preserves long-range surface consistency      |
| Masts (thin vertical) | O-Voxel QEF preserves thin structures; sparse coords don't average out |
| Antennas/radar        | O-Voxel intersected-flag marks surface voxels precisely         |
| Port/starboard symmetry | Spatial coordinates enable symmetry regularisation in latent space |
| Repeated structures (railings) | Transformer priors learn repetitive patterns |

### Biggest failure mode in practice

The sparse structure generation step uses **only one image**. For a vessel:
- Visible side → accurate occupancy
- Hidden side → hallucinated from learned priors

The `coords` tensor going into `sample_shape_slat()` will simply be missing voxels for
the hidden half of the hull. Since coords are fixed during SLAT generation, no latent
diffusion can recover missing spatial support.

This maps directly to **Strategy B** being the highest-value intervention.

---

## Summary — Where Each Concept Lives in the Code

| Research Concept                  | File                                            | Key Symbol                              |
|-----------------------------------|-------------------------------------------------|-----------------------------------------|
| Image conditioning                | `modules/image_feature_extractor.py`            | `DinoV3FeatureExtractor`                |
| Sparse structure flow             | `models/sparse_structure_flow.py`               | `SparseStructureFlowModel`              |
| Sparse structure VAE              | `models/sparse_structure_vae.py`                | `SparseStructureEncoder/Decoder`        |
| SparseTensor                      | `modules/sparse/basic.py`                       | `SparseTensor`, `sparse_cat`            |
| Shape SLAT flow model             | `models/structured_latent_flow.py`              | `SLatFlowModel`, `ElasticSLatFlowModel` |
| Shape SLAT VAE                    | `models/sc_vaes/sparse_unet_vae.py`             | `SparseUnetVaeEncoder/Decoder`          |
| Sparse attention (self + cross)   | `modules/sparse/attention/modules.py`           | `SparseMultiHeadAttention`              |
| Windowed sparse attention         | `modules/sparse/attention/windowed_attn.py`     | `sparse_windowed_scaled_dot_product_*`  |
| Flow matching sampler             | `pipelines/samplers/flow_euler.py`              | `FlowEulerCfgSampler`                   |
| O-Voxel / Flexible Dual Grid      | `o-voxel/o_voxel/convert/flexible_dual_grid.py`| `mesh_to_flexible_dual_grid`            |
| O-Voxel Z-order serialization     | `o-voxel/o_voxel/serialize.py`                  | `encode_seq` (z_order / hilbert)        |
| Full pipeline orchestration       | `pipelines/trellis2_image_to_3d.py`             | `Trellis2ImageTo3DPipeline.run()`       |
| PBR texture layout                | `pipelines/trellis2_image_to_3d.py`             | `pbr_attr_layout`                       |
