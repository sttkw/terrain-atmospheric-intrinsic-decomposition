# Atmosphere-Aware Intrinsic Decomposition from a Single Terrain Image with Latent Diffusion Models

[Project Page](https://sttkw.github.io/terrain-atmospheric-intrinsic-decomposition/)
&nbsp;|&nbsp;
[Paper](docs/static/pdf/paper.pdf)

Minimal training and inference code for terrain intrinsic decomposition and
atmospheric editing. This public release uses:

- [`ShunTatsukawa/TAID-Dataset`](https://huggingface.co/datasets/ShunTatsukawa/TAID-Dataset)
- [`ShunTatsukawa/TAID-AtmosEdit`](https://huggingface.co/datasets/ShunTatsukawa/TAID-AtmosEdit)

Pretrained weights are on [`ShunTatsukawa/TAID-Models`](https://huggingface.co/ShunTatsukawa/TAID-Models):

| File | Model |
| --- | --- |
| [`decomposition/`](https://huggingface.co/ShunTatsukawa/TAID-Models/tree/main/decomposition) | Intrinsic decomposition U-Net (step 18,000) |
| [`atmosphere/terrain_difference.pt`](https://huggingface.co/ShunTatsukawa/TAID-Models/blob/main/atmosphere/terrain_difference.pt) | Atmospheric editor |

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

## Quick start

`demo.py` downloads the pretrained weights, decomposes a terrain photo, and
edits its atmosphere:

```bash
python demo.py \
  --input_image demo_image/2.jpg \
  --water_mask demo_image/2.png \
  --p_control 0 -3 0 \
  --output_dir outputs/demo/2
```

`demo_image/<n>.jpg` are sRGB photos and `demo_image/<n>.png` their one-hot
water/terrain/sky masks (`--water_mask` is optional). `--p_control` is the
log-scale change of air, aerosol and ozone density: `0` keeps a parameter,
negative values thin it and positive values thicken it (range `[-3, 3]`).

Results are written as linear EXR plus gamma-2.2 PNG:

```text
outputs/demo/2/decomposition/{albedo,diffuse_shading,specular_shading,volume,reconstruction}
outputs/demo/2/atmosphere/{diffuse_shading,specular_shading,volume,edited}
```

`edited` is the recomposed image `A * D' + S' + V'`. A GPU with about 24 GB is
recommended for the decomposition step.

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

`TAID-AtmosEdit` stores 10,000 rows: one scene per `seed` (1..1000) with ten
atmospheric conditions per scene (`p_idx` 0..9). `D` is a linear float32 HWC NPY
in `[0, 5]`, `S` and `V` are linearly quantized 8-bit RGB PNG, and
`s_density` / `s_aerosol` / `s_ozone` are the atmospheric parameters. Each
training sample pairs two conditions of the same scene, so the editor only ever
sees the atmosphere change, never a change of terrain.

### Training

The defaults reproduce the atmospheric editor used in the paper: 512 px, base
width 32, 100 epochs, batch size 24, cosine LR from 1e-4, `lambda_delta=0`,
`lambda_x=1`, global spatial deltas, and bf16. The split is seed-wise, so a
validation scene never appears in training; `--seed 42 --val_ratio 0.1` holds
out the same 100 of the 1,000 scene seeds as the published model.

```bash
python atmosphere/train.py \
  --dataset ShunTatsukawa/TAID-AtmosEdit \
  --checkpoint_root outputs/atmosphere/checkpoints
```

Each run writes to a new `diff_ver<N>` directory: the best-validation
checkpoint, a snapshot every `--save_every` epochs, and `val_seeds.csv` with
the held-out seeds. Per-epoch losses go to `--log_csv` and TensorBoard.

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

## License

The source code in this repository is licensed under the [MIT License](LICENSE).
The `TAID-Dataset` and `TAID-AtmosEdit` datasets are separately licensed under
the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
(CC BY 4.0).
