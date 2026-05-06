# TRELLIS.2 — In-Depth Shape SLAT and Pipeline Explanation

## Overview

TRELLIS.2 is not fundamentally an image-to-mesh model.

The core idea is:

```text
image(s)
→ sparse structured 3D latent representation
→ decoded 3D representation

The most important internal representation is the:

Shape SLAT (Structured LATent)

The mesh is only a decoded output generated from this latent representation.

Main References
Official TRELLIS.2 Repository

https://github.com/microsoft/TRELLIS.2

Main Pipeline File

https://github.com/microsoft/TRELLIS.2/blob/main/trellis2/pipelines/trellis2_image_to_3d.py

TRELLIS Original Research Repository

https://github.com/microsoft/TRELLIS

TRELLIS DeepWiki Documentation

https://deepwiki.com/microsoft/TRELLIS/3.1-image-to-3d-pipeline

HuggingFace TRELLIS.2 Space

https://huggingface.co/spaces/microsoft/TRELLIS.2

High-Level Pipeline

The TRELLIS.2 image-to-3D process is approximately:

Input image
    ↓
Vision encoder / conditioning extraction
    ↓
Sparse structure generation
    ↓
Shape SLAT generation
    ↓
Texture SLAT generation
    ↓
Decoder
    ↓
Mesh / Gaussian / Radiance Field

Important distinction:

Shape != Mesh

Instead:

Shape = latent spatial representation
Mesh = decoded interpretation
Stage 1 — Image Conditioning

The input image is converted into conditioning embeddings.

Example from the pipeline:

cond_512 = self.get_cond([image], 512)

Reference:
https://github.com/microsoft/TRELLIS.2/blob/main/trellis2/pipelines/trellis2_image_to_3d.py

This conditioning likely contains:

semantic information
visual appearance
object category priors
topology priors
approximate geometry priors
learned multiview statistics

This stage is NOT yet 3D geometry.

It is closer to:

CLIP/DINO-style learned visual embeddings

with geometry-aware priors.

Stage 2 — Sparse Structure Generation

TRELLIS does not immediately generate dense voxels.

Instead, it predicts:

Which regions of 3D space probably contain geometry

This creates a sparse occupancy structure.

Instead of:

1024³ dense voxel grids

TRELLIS stores only active spatial regions:

coords = [
    [x1, y1, z1],
    [x2, y2, z2],
    ...
]

This is extremely important for:

memory efficiency
scalability
thin structure preservation
sparse detail handling
Stage 3 — Shape SLAT Generation

Example from the pipeline:

shape_slat = self.sample_shape_slat(
    cond_512,
    model,
    coords,
    params
)

Reference:
https://github.com/microsoft/TRELLIS.2/blob/main/trellis2/pipelines/trellis2_image_to_3d.py

What Is a Shape SLAT?

SLAT = Structured LATent

A Shape SLAT is essentially:

Sparse spatial tensor

where each occupied coordinate stores a learned latent feature vector.

Conceptually:

ShapeSLAT = {
    coords: Nx3 voxel coordinates,
    features: NxC latent vectors
}

Where:

N = number of occupied sparse voxels
3 = x,y,z coordinates
C = latent feature dimension

The important part:

These are NOT:
- mesh vertices
- triangles
- colors
- textures

They are:

compressed learned geometric descriptors
What Information Does a Shape SLAT Token Contain?

A single Shape SLAT feature vector likely encodes:

Information	Description
Local geometry	shape structure and curvature
Occupancy confidence	whether geometry exists
Surface orientation	normal/directional hints
Topology priors	connectivity relationships
Multi-scale shape detail	coarse + fine geometry
Contextual relationships	neighboring spatial structure
Edge/sharpness priors	corners, thin structures
Semantic priors	"mast", "wheel", "antenna", etc
Surface continuity	smoothness constraints
Implicit material hints	reflective vs diffuse structure cues

The exact latent semantics are learned during training.

Why “Structured” Latents Matter

Traditional latent diffusion:

latent = unstructured vector

TRELLIS:

latent = spatially organized sparse 3D tensor

This means:

The latent itself already has 3D spatial meaning

This enables:

local editing
local regeneration
region refinement
multiview fusion
geometric constraints
spatial conditioning
Why Shape SLAT Is Important for Multi-Image Input

Current TRELLIS flow:

single image
→ hallucinated 3D prior
→ Shape SLAT
→ mesh

But Shape SLAT is spatial.

This means multiple images could contribute to the SAME latent space.

Potential future flow:

front image
→ front geometry evidence

left image
→ left geometry evidence

rear image
→ rear geometry evidence

all merged into unified Shape SLAT

This is likely the BEST location to improve reconstruction quality.

Possible Operations on Shape SLAT
1. Multi-View Fusion

Most important improvement area.

Each image could provide:

partial spatial confidence

Example:

Region	Best Input
Bow	front image
Stern	rear image
Port side	left image
Starboard	right image

Then merge latent features spatially.

2. Depth Injection

Depth maps are highly compatible with Shape SLAT generation.

Potential benefits:

better occupancy prediction
improved hull thickness
antenna separation
more accurate mast depth
reduced hallucination

Useful models:

Depth Anything V2
MoGe
UniDepth
Marigold

References:

https://github.com/DepthAnything/Depth-Anything-V2

https://github.com/microsoft/MoGe

https://github.com/lpiccinelli-eth/UniDepth

https://github.com/prs-eth/Marigold

3. Edge / Canny Constraints

Ships contain strong geometric edges:

railings
hull boundaries
deck edges
radar structures
antennas

Edge maps could improve:

sparse occupancy initialization

before Shape SLAT generation.

4. SLAT Editing

Since coordinates are explicit:

shape_slat.features[mask] = edited_features

Potential operations:

remove region
regenerate region
upscale region
repair damaged geometry
smooth local structure
5. Symmetry Enforcement

Very important for vessels.

Example:

port side ↔ starboard side symmetry

Could improve:

hull consistency
mast alignment
deck reconstruction
railing placement
Shape SLAT vs Texture SLAT

TRELLIS separates geometry from appearance.

Pipeline structure:

shape_slat = sample_shape_slat(...)
tex_slat = sample_tex_slat(..., shape_slat)

Reference:
https://github.com/microsoft/TRELLIS.2/blob/main/trellis2/pipelines/trellis2_image_to_3d.py

Meaning:

Texture generation depends on geometry generation

This is important because geometry quality can be improved independently first.

TRELLIS.2 O-Voxel Representation

TRELLIS.2 introduces:

O-Voxel (Omni-Voxel)

Reference:
https://github.com/microsoft/TRELLIS.2

Designed to better handle:

arbitrary topology
thin structures
disconnected geometry
sparse detail
high-resolution occupancy

This is extremely important for:

ships
antennas
radar domes
masts
cranes
railings

Traditional occupancy fields struggle heavily with these structures.

Why TRELLIS Performs Better Than Older Pipelines

Older pipelines typically relied on:

single implicit field
single NeRF
single occupancy function

Problems:

blurry geometry
topology collapse
weak thin structures
poor scaling

TRELLIS instead uses:

sparse structured 3D latent representations

which preserve:

locality
topology
sparsity
fine detail
structural continuity
Most Important Insight

The key realization:

TRELLIS is NOT primarily an image-to-mesh model

It is:

image(s)
→ structured sparse 3D latent
→ decoded representation

Meaning:

The latent stage is the real reconstruction stage

That is where:

multiview fusion
depth conditioning
geometric constraints
symmetry priors
segmentation guidance
occupancy refinement

should be injected.

NOT after mesh generation.

Recommended Future Research Direction
Best Direction
Multi-image sparse occupancy fusion

before Shape SLAT generation.

Second Best
Depth-guided occupancy initialization

using:

Depth Anything V2
MoGe
UniDepth
Marigold
Third Best
Ship segmentation + symmetry priors

before sparse voxel generation.

Suggested Future Pipeline

Potential improved architecture:

Input images
    ↓
Canonicalization
    ↓
Multiview alignment
    ↓
Depth estimation
    ↓
Sparse occupancy fusion
    ↓
Shape SLAT generation
    ↓
Texture SLAT generation
    ↓
Decoder

instead of:

single image
→ hallucinated latent
→ mesh

This would likely produce significantly better reconstruction quality for marine structures and thin geometric objects.
The really important areas are:

1. Sparse Transformer Architecture

TRELLIS is not just storing sparse voxels.

It runs a transformer directly over sparse spatial tokens.

Meaning:

token = spatial voxel latent

instead of:

token = word

or:

token = image patch

This changes EVERYTHING.

Sparse Attention

Traditional transformers:

attention complexity = O(N²)

Impossible for dense 3D.

TRELLIS solves this with:

sparse spatial attention

Only occupied regions participate.

This means:

ship hull
→ active tokens

empty ocean around it
→ ignored

This is why TRELLIS can scale to high-detail geometry.

What Each Token REALLY Represents

A Shape SLAT token is probably closer to:

localized geometric field descriptor

than a voxel.

It likely encodes:

occupancy probability
signed distance hints
local curvature
neighboring topology
feature continuity
learned semantic shape priors
directional structure information

The latent is NOT binary occupancy.

It is a learned continuous representation.

Important Distinction — SLAT vs Occupancy Grid

Occupancy grid:

voxel[x,y,z] = 0 or 1

Shape SLAT:

voxel[x,y,z] = learned feature vector

Example:

[
  0.12,
  -0.88,
  1.72,
  ...
]

Maybe 32–256 dimensions.

Those dimensions are learned during VAE training.

2. Shape SLAT Is Probably Hierarchical

This is VERY important.

TRELLIS likely stores:

coarse global structure
+
fine local detail

simultaneously.

Meaning:

Scale	Information
Large-scale	hull shape
Medium-scale	deck structures
Small-scale	antennas, rails

This explains why TRELLIS preserves topology surprisingly well.

3. The Sparse Structure Generator Is Separate

This is one of the least understood but most important parts.

Pipeline:

image
→ sparse occupancy prediction
→ SLAT generation

The sparse occupancy stage determines:

WHERE geometry exists

before:

WHAT geometry exists

This is critical.

Why This Matters For Multi-Image Input

You could fuse multiple images BEFORE occupancy generation.

Example:

front image
→ frontal occupancy confidence

left image
→ port-side occupancy confidence

depth map
→ geometric constraints

Then generate a better sparse structure.

This might improve TRELLIS more than modifying the SLAT itself.

4. O-Voxel Representation (TRELLIS.2)

TRELLIS.2 introduces:

Omni-Voxel (O-Voxel)

This is NOT just a sparse voxel grid.

It is likely:

direction-aware sparse spatial representation

optimized for:

arbitrary topology
disconnected structures
thin geometry
high-detail sparsity

This matters enormously for ships.

Why Thin Structures Are Hard

Most 3D diffusion systems fail on:

antennas
rails
radar poles
cables
ladders

because:

thin structures occupy tiny spatial volume

Dense voxel systems lose them.

O-Voxel specifically targets this issue.

5. TRELLIS Uses VAE-Style Latent Compression

This is extremely important.

TRELLIS does NOT diffuse directly on meshes.

It diffuses in:

compressed latent geometry space

Meaning:

mesh
→ encoder
→ Shape SLAT latent
→ decoder
→ mesh

The diffusion model operates INSIDE the latent space.

Why This Matters

The latent learns:

valid geometry manifold

Meaning:

connected surfaces
plausible topology
smooth transitions
semantic structure

instead of random triangles.

6. Decoding Is Separate From Representation

This is one of the most important concepts.

The Shape SLAT itself is representation-agnostic.

It can theoretically decode into:

Decoder	Output
Marching cubes	mesh
Gaussian decoder	Gaussian splats
NeRF decoder	radiance field
SDF decoder	signed distance field

Meaning:

SLAT ≠ mesh

Again.

This is huge architecturally.

7. Texture SLAT Is Conditioned On Geometry

TRELLIS separates:

geometry generation

from:

appearance generation

This is MUCH more stable.

Pipeline:

shape latent
→ texture latent

instead of trying to solve both simultaneously.

Why This Is Important

If geometry is wrong:

texture generation collapses

So improving Shape SLAT quality directly improves textures.

8. Attention Probably Uses Spatial Neighborhoods

Very important for future modifications.

TRELLIS attention is likely:

spatially localized

instead of global.

Meaning:

mast token
↔ nearby mast tokens

NOT

mast token
↔ random hull token

This preserves local geometry consistency.

9. Shape SLAT Can Potentially Support Partial Regeneration

This is VERY interesting for reconstruction systems.

Because SLAT is sparse + spatial:

you could theoretically regenerate ONLY:

stern
radar mast
deck
bridge

without regenerating the whole model.

10. Potential Multi-Image Extension Strategies

This is probably what you actually care about.

There are several places where multiple images could be injected.

Strategy A — Early Feature Fusion
multiple images
→ combined conditioning embedding
→ normal TRELLIS pipeline

Simplest approach.

Weakest geometry improvement.

Strategy B — Sparse Occupancy Fusion (BEST)
multiple images
→ multiple occupancy predictions
→ fused sparse structure
→ Shape SLAT generation

Probably the strongest option.

Strategy C — SLAT Fusion
image A → partial SLAT
image B → partial SLAT
merge SLATs

Much harder.

Potentially extremely powerful.

Strategy D — Geometry Constraint Injection

Inject:

depth maps
segmentation
edge maps
symmetry priors

during sparse structure generation.

Probably easier than retraining full multi-image TRELLIS.

11. Marine Vessel Reconstruction Implications

TRELLIS is actually unusually well-suited for ships because ships are:

Property	TRELLIS Advantage
Sparse thin structures	O-Voxel
Large connected surfaces	spatial continuity
Symmetry	latent spatial reasoning
Long geometry	sparse scaling
Repeated structures	transformer priors
Biggest Failure Mode

Single-image hallucination.

Example:

visible side = accurate
hidden side = guessed

For vessels this becomes catastrophic because:

deck layouts matter
mast placement matters
hull symmetry matters

This is EXACTLY where multi-image fusion helps.

12. The Most Important Architectural Insight

TRELLIS is really:

3D latent scene reasoning

NOT:

mesh generation

The mesh is only a final export format.

The actual intelligence exists in:

sparse structured latent geometry space

That is where future improvements should happen.

What I Would Research Next

For your marine pipeline specifically:

Highest-value research
multi-image sparse occupancy fusion

before Shape SLAT generation.

Second highest
depth-guided sparse occupancy initialization
Third highest
symmetry-aware latent constraints

for port/starboard consistency.

Most Likely Best Future Pipeline
Input images
    ↓
View alignment
    ↓
Depth estimation
    ↓
Segmentation
    ↓
Sparse occupancy fusion
    ↓
Shape SLAT generation
    ↓
Texture SLAT generation
    ↓
Decoder
    ↓
Mesh

This is likely MUCH more powerful than simply feeding more images into the existing single-image encoder.