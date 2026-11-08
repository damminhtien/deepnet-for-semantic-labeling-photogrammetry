# Public model weights

**Status: CURRENT** — Final single-model state dictionaries for the four systems in the
Potsdam benchmark comparison are available from the
[Hugging Face model repository](https://huggingface.co/damminhtien/deepnet-potsdam-segmentation).
Use the immutable Hub revision below when recording a reproduction.

| Release field | Value |
| --- | --- |
| Hub revision | `fc4e205b1d3a8338ad15d3a6dd659b6ddbf3797c` |
| Release manifest | [`manifest.json`](https://huggingface.co/damminhtien/deepnet-potsdam-segmentation/blob/fc4e205b1d3a8338ad15d3a6dd659b6ddbf3797c/manifest.json) |
| Checksum list | [`SHA256SUMS`](https://huggingface.co/damminhtien/deepnet-potsdam-segmentation/blob/fc4e205b1d3a8338ad15d3a6dd659b6ddbf3797c/SHA256SUMS) |
| Local release record | [`model-weights-manifest.json`](../artifacts/public/model-weights-manifest.json) |

The release is scoped to the final weights for the four systems in the README benchmark
table. It does not include selection/intermediate checkpoints, optimizer state, or
unrelated linear-probe artifacts. Upstream weights are not mirrored separately unless
they are part of a released full model state.

The scores below are measured on the same previously examined 14-scene Potsdam
benchmark cohort and 467,862,855 valid pixels, using six-class eroded references.
They are results of the recorded model recipes, not a new blind test or an
architecture-only comparison. Source inputs and evaluation details are in each linked
experiment record.

| Model | Input recorded for the run | OA / six-class mIoU | Exported weights |
| --- | --- | ---: | --- |
| DINOv3 ViT-B/16 | Verified RGB bands 1/2/3 from RGBIR | 92.0397% / 81.0932% ([CLAIM-001](../artifacts/public/evidence-manifest.json)) | [`model.safetensors`](https://huggingface.co/damminhtien/deepnet-potsdam-segmentation/blob/fc4e205b1d3a8338ad15d3a6dd659b6ddbf3797c/models/dinov3-vitb16-rgb/model.safetensors) · [campaign](experiments/potsdam-dinov3-rgbir-v2-final-review-2026-11-06.md) |
| DINOv2 ViT-B/14 | Historical three-band file configured as IRRG; stored values audited as NIR/green/blue | 91.4785% / 79.6122% ([CLAIM-003](../artifacts/public/evidence-manifest.json)) | [`model.safetensors`](https://huggingface.co/damminhtien/deepnet-potsdam-segmentation/blob/fc4e205b1d3a8338ad15d3a6dd659b6ddbf3797c/models/dinov2-vitb14-irrg/model.safetensors) · [review](experiments/potsdam-dinov2-coverage-refit-final-review-2026-11-05.md) |
| PSPNet ResNet101 | Verified RGB | 91.2834% / 79.4235% ([CLAIM-002](../artifacts/public/evidence-manifest.json)) | [`model.safetensors`](https://huggingface.co/damminhtien/deepnet-potsdam-segmentation/blob/fc4e205b1d3a8338ad15d3a6dd659b6ddbf3797c/models/pspnet-resnet101-rgb/model.safetensors) · [report](experiments/potsdam-pspnet-rgb-v1.md) |
| SAM ViT-B + rank-4 LoRA | Historical three-band file catalogued as IRRG | 87.8142% / 73.7024% ([CLAIM-004](../artifacts/public/evidence-manifest.json)) | [`model.safetensors`](https://huggingface.co/damminhtien/deepnet-potsdam-segmentation/blob/fc4e205b1d3a8338ad15d3a6dd659b6ddbf3797c/models/sam-vitb-lora-irrg/model.safetensors) · [review](experiments/isprs-dinov2-joint-validation-review-2026-11-01.md) |

Scores are shown as OA / six-class mIoU.

## Download and load

Each download is pinned to the release commit so a future update cannot silently change
the file:

```bash
python -m pip install huggingface_hub safetensors
```

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

repo_id = "damminhtien/deepnet-potsdam-segmentation"
revision = "fc4e205b1d3a8338ad15d3a6dd659b6ddbf3797c"
filename = "models/dinov3-vitb16-rgb/model.safetensors"

path = hf_hub_download(repo_id, filename, revision=revision)
state_dict = load_file(path, device="cpu")
```

Construct the matching architecture using the model specification and implementation
in this repository, then load the returned mapping with strict key checking. DINOv2 and
DINOv3 model specs, study configs, and evaluation protocols are stored alongside the
weights on the Hub. PSPNet and SAM construction details are in their `model-info.json`
files and linked reports. The SAM export contains only LoRA and decoder tensors; obtain
the official [SAM ViT-B base checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
separately and verify its SHA-256 from the Hub `model-info.json` before applying the
released trainable state.

The DINOv3 and DINOv2 files contain their trained encoder and segmentation-model state,
so these exports are not LoRA-only deltas. All four files use the `safetensors` format
and omit optimizer/scaler state, training examples, source rasters, labels, and
prediction maps. The artifact manifest records each exported-file SHA-256, byte size,
tensor count, and source checkpoint hash.

## Licenses and data rights

The Hub bundle carries separate license notices for the included model families. The
[DINOv3 License Agreement](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
is included in the release and applies to its DINOv3-derived model materials; keep that
notice with any permitted redistribution. DINOv2 and SAM upstream code and model weights
are released under [Apache-2.0](https://github.com/facebookresearch/dinov2) and
[Apache-2.0](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE),
respectively. The PSPNet export includes TorchVision-derived ResNet weights and its
[BSD-3-Clause notice](https://github.com/pytorch/vision/blob/main/LICENSE); upstream
weight and source-data terms also apply. Read the exact notices bundled with the model
files before reuse or redistribution; this repository does not grant one common license
for the bundle.

Potsdam imagery and labels are not part of the release. Obtain them from the
[official ISPRS dataset page](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
and follow the current access and usage terms there. The [dataset description](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
specifies the available RGB, IRRG, RGBIR, DSM, and label assets; the DINOv2 and SAM runs
used historical three-band files, while the DINOv3 and PSPNet runs used verified RGB.
Release of trained weights does not redistribute the imagery or grant rights to it.
