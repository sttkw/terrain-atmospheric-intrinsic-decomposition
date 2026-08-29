# Terrain Atmospheric Intrinsic Decomposition

Minimal training and inference code for terrain intrinsic decomposition and
atmospheric editing. This public release uses:

- [`ShunTatsukawa/TAID-Dataset`](https://huggingface.co/datasets/ShunTatsukawa/TAID-Dataset)
- [`ShunTatsukawa/TAID-AtmosEdit`](https://huggingface.co/datasets/ShunTatsukawa/TAID-AtmosEdit)

The repository intentionally excludes cluster launch files, containers,
experiments, ablations, evaluation dumps, optimizer snapshots, and generated
outputs.

## Installation

Python 3.11 or 3.12 and a CUDA-capable PyTorch environment are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Both datasets and the InstructPix2Pix base model are public, so an HF token is
not normally required. Set `HF_TOKEN` only if your environment requires one.

## 1. Intrinsic decomposition

The decomposition model predicts one of four components per denoising run:
Albedo (`A`), Diffuse Shading (`D`), Specular Shading (`S`), or Volume (`V`).
It retains the 10-channel U-Net input and `v_prediction` used by the original
`rgbx/outputs/terrain-ip2p/rgbx` model.

`TAID-Dataset` stores one scene per row. `Input/A/S/V` are linear 8-bit PNG,
`D` is a linear float32 HWC NPY in `[0, 5]`, and `water_mask` is a one-hot RGB
PNG (`R=water`, `G=terrain`, `B=sky`). The loader converts `D` to
`log1p(D)/log1p(5)` for the diffusion target.

### Training

The following reproduces the public-data training configuration corresponding
to the 24,000-step RGBX run:

```bash
accelerate launch decomposition/train.py \
  --dataset_name ShunTatsukawa/TAID-Dataset \
  --pretrained_model_name_or_path timbrooks/instruct-pix2pix \
  --output_dir outputs/decomposition/rgbx \
  --resolution 512 \
  --max_train_steps 24000 \
  --train_batch_size 6 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine \
  --lr_warmup_steps 0 \
  --checkpointing_steps 1000 \
  --checkpoints_total_limit 10 \
  --loss_weight_diffusion 1.0 \
  --loss_weight_recon 0.5 \
  --loss_weight_direct 1.0 \
  --loss_weight_water 2.0 \
  --random_flip \
  --gradient_checkpointing \
  --enable_vae_slicing \
  --enable_vae_tiling \
  --mixed_precision bf16 \
  --allow_tf32
```

Run the data-format test without downloading the base model:

```bash
python decomposition/train.py --self_test
```

### Inference

Generate all four maps with reconstruction correction:

```bash
python decomposition/inference.py \
  --mode corr \
  --model_path outputs/decomposition/rgbx \
  --input_image path/to/input.png \
  --output_dir outputs/decomposition/example \
  --dtype bf16 \
  --input_is_linear
```

Generate one component:

```bash
python decomposition/inference.py \
  --mode normal \
  --model_path outputs/decomposition/rgbx \
  --input_image path/to/input.png \
  --prompt "Diffuse Shading" \
  --output_path outputs/decomposition/diffuse.exr \
  --input_is_linear
```

`--water_mask` is optional at inference. If omitted, both mask-conditioning
channels are zero. For behavior matching the mask-conditioned training run,
supply a compatible one-hot RGB segmentation mask.

## 2. Atmospheric editing

The atmospheric editor predicts per-component log changes with a 12-channel to
9-channel U-Net:

```text
input  = [D, S, V, broadcast(delta_log_parameters)]
output = [delta_log_D, delta_log_S, delta_log_V]
```

### Training

Defaults reproduce the architecture and hyperparameters stored in
`atmos/checkpoints/diff_ver4`: 512 px, base width 32, 100 epochs, batch size 24,
cosine LR, `lambda_delta=0`, `lambda_x=1`, global spatial deltas, and bf16.

```bash
python atmosphere/train.py \
  --dataset ShunTatsukawa/TAID-AtmosEdit \
  --checkpoint_root outputs/atmosphere/checkpoints
```

### Inference

```bash
python atmosphere/infer.py \
  --checkpoint outputs/atmosphere/checkpoints/diff_ver1/terrain_difference.pt \
  --diff_shading path/to/D.npy \
  --spec_shading path/to/S.png \
  --volume path/to/V.png \
  --p_cur 1.0 1.0 1.0 \
  --p_tgt 1.2 0.8 1.1 \
  --out_diff_shading outputs/atmosphere/D_edited.npy \
  --out_spec_shading outputs/atmosphere/S_edited.png \
  --out_volume outputs/atmosphere/V_edited.png
```

Inputs and NPY outputs are linear. PNG values are interpreted as linear byte
values; no gamma transfer is applied by the data loaders.

## Reproducibility scope

The public datasets intentionally use portable PNG/NPY payloads and do not
contain every field from the internal training datasets:

- `TAID-Dataset` includes the water/terrain/sky segmentation mask used for
  conditioning and the water-specific loss. PNG quantization remains the main
  data-format difference from the internal float tensor representation.
- `TAID-AtmosEdit` stores `S/V` as linear PNG clipped to `[0, 1]`. The original
  internal atmospheric dataset used float16 HDR tensors, including values above
  one. The architecture and training procedure are reproduced, but exact
  diff_ver4 weights cannot be regenerated from the clipped public payloads.

These differences are explicit so results obtained from the public datasets are
not presented as bitwise reproduction of the internal checkpoints.

## Security

Dataset NPY data is loaded with `allow_pickle=False`, and model checkpoints are
loaded using PyTorch's `weights_only=True` mode. Authentication tokens are never
stored in newly produced checkpoints.
