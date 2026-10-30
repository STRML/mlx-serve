# idol — speridlabs/idol (Apache-2.0): one image to a 3D Gaussian splat

A photo or render of one object in, up to a million 3D Gaussians out, as a GLB. `geometry: "discs"` (the default) is a
triangle mesh any viewer draws solid: one opaque quad per Gaussian, its colour a texel of a baked atlas. `"splat"` is the
ratified `KHR_gaussian_splatting` extension on one POINTS primitive (the app asks for it: it draws its own discs and
saves a `.ply` from the attributes). The
chain: BiRefNet matte (only when the image has no real alpha) -> crop to 1024 on black -> DINOv3 ViT-H+/16 tokens and a
sample of the FLUX.2 VAE's posterior (4101 tokens each, CLS and registers first) -> a rectified-flow denoiser over an
8192 x 16 latent plus a 5-value camera (Euler, classifier-free guidance, learned per-token RoPE) -> the octree decoder
places the anchors (a CPU systematic-sampling descent) -> the Gaussian decoder turns each into 32.

## Files

| File | Owns |
|---|---|
| `config.zig` | MLX-free: claims (`model_type: idol`, or the release's `config.json` by its denoiser class), the bills |
| `plugin.zig` | the media table, the request fields and their named refusals, the tiny pack |
| `idol.zig` | the engine: load, conditioning, the sampler, decode; the env-gated oracles |
| `denoiser.zig` / `decoders.zig` / `layers.zig` | the three transformers, the octree descent and Gaussian activation, their shared blocks |
| `birefnet.zig` | BiRefNet (Swin-L, deformable ASPP decoder as gathers), f32 channels-last |
| `image.zig` | `preprocess_image` in Pillow's arithmetic (premultiplied resize, `MinFilter(3)`, `paste`) |
| `splat.zig` | the disc-mesh GLB, the splat GLB (glTF y-up) and the standard 3DGS `.ply` (the reference's `save_ply` axes) |

## Rules

- The conditioning is a SAMPLE of the VAE posterior (`latent_dist.sample()`), not its mean: the scatter is as large as
  the signal (cos 0.84 between the two). `flux2`'s `VaeEncoder.encodeSample` draws it; the mean path is for oracles.
- DINOv3's `last_hidden_state` is the checkpoint's own affine final norm, then a non-affine LayerNorm (`affine_final`);
  TRELLIS.2 reads the block output without it.
- Reference numerics: Linear of the blocks and refiners f16 (`castLinears`), norms, embeddings, RoPE angles, the
  heads and the RePo layers f32. RoPE rotates BEFORE the q/k RMS norm, on adjacent (even, odd) pairs.
- The denoiser's image context depends on neither the latent nor the time: it is refined once per branch (the
  unconditional branch is all-zero features), never per step.
- Widths come from `config.json` and the weights (the tiny pack is 128 wide at 64 px); head width is 64 everywhere.
- The octree RNG (systematic sampling, in-cell jitter) is ours: only its distribution matches the reference's.
- A converted pack loads `birefnet.safetensors` as released (OIHW is turned to OHWI at load, batch norms folded); the
  release's own folder plus `dinov3/` and `vae/` loads too, its f32 cast at load. Quantized packs: Linear whose input
  is a multiple of 64 (`tests/convert_idol_weights.py --bits 8`); the VAE's middle attention is always 8-bit (flux2's
  loader reads only quantized ones there).
- The FLUX.2 VAE ships from the ungated klein repo (the same VAE as FLUX.2-dev's, whose repo is gated).
- Viewers ignore per-vertex colour (Preview draws it white) and Preview draws a POINTS primitive as bare dots: the disc
  mesh bakes colour into a texture with nearest sampling. It is opaque, so glassy or hazy regions show one layer of the
  blend (windshields speckle); only a blended renderer fixes that.
- The GLB's axes are glTF's y-up; the PLY keeps the reference's `save_ply` turn (its +y points down in a viewer).

- Published packs: `ddalcu/Idol-MLX-Serve-8bit` (the app's download, 3.0 GB) and `ddalcu/Idol-MLX-Serve-fp16` (3.6 GB, the
  quality reference), both from `tests/convert_idol_weights.py` with the Idol `LICENSE`, a `NOTICE` and the DINOv3 licence.

## Proofs

- Tiny pack (`conformanceFixture`): every model at the smallest width its code takes; a job answers a GLB.
- Reference oracles (env-gated `IDOL_*`, `tests/dump_idol_fixtures.py` on MPS/CPU): the crop (bit-exact from the
  matte), BiRefNet's matte (IoU 0.9999), DINOv3 and VAE tokens, one guided velocity (cos 0.99998), octree logits,
  the Gaussian activation (exact), a 4-step sampler (cos 0.998), the end-to-end run (`IDOL_PACK`, `IDOL_IMAGE`,
  `IDOL_OUT` writes `splat.ply` + `splat.glb`). `tests/test_idol_3d.sh` is the HTTP surface.

## The row from the root architecture table

speridlabs/idol (Apache-2.0): `model_type: idol` (the release's `config.json` has none: claimed by its denoiser's
class `LatentSeqMMFlowModel`), `.mesh` modality, a Gaussian splat as a disc-mesh or `KHR_gaussian_splatting` GLB. Pack adds
`dinov3/` and `vae/` beside the release's four weight files.
