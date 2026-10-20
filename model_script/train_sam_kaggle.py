"""Fine-tune SAM ViT-B with attention LoRA for six-class Potsdam labeling."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_runtime_dependencies():
    core_modules = ("numpy", "torch", "torchvision")
    missing_core = [name for name in core_modules if importlib.util.find_spec(name) is None]
    if missing_core:
        raise RuntimeError(
            "Kaggle GPU image is missing required core packages: "
            + ", ".join(missing_core)
            + ". Select the standard Kaggle GPU image; torch is not installed from PyPI here."
        )

    optional_packages = (
        ("tifffile", "tifffile"),
        ("PIL", "pillow"),
        ("cv2", "opencv-python-headless"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
    )
    if not Path("/kaggle/input").is_dir():
        return

    for module_name, package_name in optional_packages:
        if importlib.util.find_spec(module_name) is None:
            print(f"Installing missing Kaggle runtime package: {package_name}", flush=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", package_name],
                check=True,
            )


def verify_source_package(project_root):
    manifest_path = project_root / "source_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Kaggle source package has no file inventory")
    for record in records:
        source = project_root / record["path"]
        if not source.is_file() or source.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Missing or wrong-sized Kaggle source file: {source}")
        if sha256_file(source) != record["sha256"]:
            raise RuntimeError(f"Kaggle source checksum mismatch: {source}")
    expected = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected != manifest.get("source_sha256"):
        raise RuntimeError("Kaggle source package fingerprint does not match")
    return sha256_file(manifest_path)


ensure_runtime_dependencies()

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_CANDIDATES = [SCRIPT_DIR, SCRIPT_DIR.parent]
KAGGLE_INPUT = Path("/kaggle/input")
if KAGGLE_INPUT.is_dir():
    PROJECT_ROOT_CANDIDATES.extend(
        package_dir.parent
        for package_dir in KAGGLE_INPUT.rglob("semantic_labeling")
        if package_dir.is_dir() and (package_dir / "__init__.py").is_file()
    )
PROJECT_ROOT = next(
    (
        root.resolve()
        for root in PROJECT_ROOT_CANDIDATES
        if (root / "semantic_labeling" / "__init__.py").is_file()
    ),
    None,
)
if PROJECT_ROOT is None:
    raise ModuleNotFoundError(
        "Could not find semantic_labeling in the kernel files or attached Kaggle "
        "datasets; attach the SAM source dataset to this kernel."
    )
SOURCE_PACKAGE_MANIFEST_SHA256 = verify_source_package(PROJECT_ROOT)
if (
    PROJECT_ROOT.is_relative_to(KAGGLE_INPUT.resolve())
    and SOURCE_PACKAGE_MANIFEST_SHA256 is None
):
    raise RuntimeError("The attached SAM source package is missing its checksum manifest")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic_labeling.constants import IGNORE_LABEL, LABELS, N_CLASSES, PALETTE
from semantic_labeling.losses import (
    inverse_sqrt_class_weights,
    weighted_ce_macro_dice_loss,
)
from semantic_labeling.sam_lora import SAMViTSemanticSegmenter

SAM_REPOSITORY = "https://github.com/facebookresearch/segment-anything.git"
SAM_COMMIT = "dca509fe793f601edb92606367a655c15ac00fdf"
SAM_INSTALL_URL = f"git+{SAM_REPOSITORY}@{SAM_COMMIT}"
SAM_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)
DEFAULT_PATCH_SIZE = 512
DEFAULT_STEPS_PER_EPOCH = 256
DEFAULT_EPOCHS = 30
DEFAULT_RUN_ID = "sam-vitb-lora-potsdam-20261020-rerun-03"
DEFAULT_OUTPUT_ROOT = Path("/kaggle/working")
DEFAULT_CHECKPOINT_PATH = Path("/tmp/sam-vitb-lora-cache/sam_vit_b_01ec64.pth")
CE_LOSS_WEIGHT = 0.7
DICE_LOSS_WEIGHT = 0.3
TRAIN_IDS_KEY = "train_ids"
VALIDATION_IDS_KEY = "validation_ids"
TEST_IDS_KEY = "held_out_test_ids"


def atomic_json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, path)


def find_data_root(requested_root):
    if requested_root is not None:
        root = Path(requested_root).expanduser().resolve()
        if not (root / "bundle_manifest.json").is_file():
            raise FileNotFoundError(f"Missing bundle_manifest.json under {root}")
        return root

    input_root = Path("/kaggle/input")
    roots = sorted(
        {manifest.parent.resolve() for manifest in input_root.rglob("bundle_manifest.json")}
    )
    if len(roots) == 1:
        return roots[0]
    if len(roots) > 1:
        raise RuntimeError(
            "More than one SAM Potsdam bundle is mounted: "
            + ", ".join(str(root) for root in roots)
        )

    tar_archives = sorted(input_root.rglob("*.tar"))
    if tar_archives:
        extraction_root = Path("/kaggle/working/sam-potsdam-data")
        extraction_root.mkdir(parents=True, exist_ok=True)
        for archive_path in tar_archives:
            relative_path = archive_path.relative_to(input_root)
            dataset_root = extraction_root / relative_path.parts[0]
            archive_target = dataset_root / archive_path.stem
            archive_target.mkdir(parents=True, exist_ok=True)
            resolved_archive_target = archive_target.resolve()
            with tarfile.open(archive_path, "r:*") as archive:
                for member in archive.getmembers():
                    member_path = Path(member.name)
                    resolved_path = (archive_target / member_path).resolve()
                    if (
                        member_path.is_absolute()
                        or ".." in member_path.parts
                        or (
                            resolved_path != resolved_archive_target
                            and resolved_archive_target not in resolved_path.parents
                        )
                    ):
                        raise ValueError(
                            f"Unsafe path in Kaggle dataset archive: {member.name}"
                        )
                    if member.issym() or member.islnk():
                        raise ValueError(
                            f"Unexpected link in Kaggle dataset archive: {member.name}"
                        )
                archive.extractall(archive_target)
        roots = sorted(
            {
                manifest.parent.resolve()
                for manifest in extraction_root.rglob("bundle_manifest.json")
            }
        )
        if len(roots) == 1:
            return roots[0]
    if not roots:
        raise FileNotFoundError(
            "Could not find a SAM bundle or archive under /kaggle/input; "
            "pass --data-root explicitly"
        )
    raise RuntimeError("Could not resolve one SAM Potsdam data bundle")


def verify_data_bundle(data_root):
    manifest_path = data_root / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("SAM data bundle has no file inventory")

    for record in records:
        path = data_root / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Data bundle file missing or size-mismatched: {path}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Data bundle checksum mismatch: {path}")

    fingerprint_payload = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()
    if fingerprint != manifest.get("bundle_sha256"):
        raise RuntimeError("Data bundle fingerprint does not match its inventory")
    return manifest


def ensure_segment_anything():
    package_root = Path(tempfile.gettempdir()) / f"sam-source-{SAM_COMMIT}"
    package = package_root / "segment_anything" / "__init__.py"
    if not package.is_file():
        print(f"Installing official Segment Anything source at {SAM_COMMIT}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(package_root),
                SAM_INSTALL_URL,
            ],
            check=True,
        )
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    return package_root


def download_checkpoint(checkpoint_path):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.is_file() and checkpoint_path.stat().st_size > 100_000_000:
        return

    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".part")
    print("Downloading the official SAM ViT-B checkpoint", flush=True)
    with urllib.request.urlopen(SAM_CHECKPOINT_URL, timeout=60) as response:
        total_size = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        with temporary_path.open("wb") as target:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                downloaded += len(chunk)
                if total_size and downloaded % (32 * 1024 * 1024) < len(chunk):
                    print(
                        f"checkpoint_downloaded_bytes={downloaded}/{total_size}",
                        flush=True,
                    )
    if temporary_path.stat().st_size < 100_000_000:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded SAM checkpoint is unexpectedly small")
    os.replace(temporary_path, checkpoint_path)


def read_tiff(path):
    try:
        import tifffile

        return tifffile.imread(path)
    except ImportError:
        from skimage import io

        return io.imread(path)


def read_class_map(path):
    with np.load(path, allow_pickle=False) as archive:
        labels = archive["labels"]
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Expected a 2D integer class map at {path}")
    if np.any((labels < 0) | (labels > IGNORE_LABEL)):
        raise ValueError(f"Class map contains values outside 0..{IGNORE_LABEL}: {path}")
    return labels.astype(np.uint8, copy=False)


def compute_training_class_counts(data_root, scene_ids):
    """Count non-ignored pixels from the training scenes only."""
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for scene_id in scene_ids:
        labels = read_class_map(
            data_root / "labels" / "train" / f"{scene_id}.npz"
        )
        valid = (labels >= 0) & (labels < N_CLASSES)
        counts += np.bincount(
            labels[valid].astype(np.int64, copy=False), minlength=N_CLASSES
        )
        del labels
    if np.any(counts == 0):
        missing = [LABELS[index] for index in np.flatnonzero(counts == 0)]
        raise ValueError("Training split has no pixels for classes: " + ", ".join(missing))
    return counts


class PotsdamPatchDataset(Dataset):
    """Sample random aligned 512-pixel patches from the 18 training scenes."""

    def __init__(self, data_root, scene_ids, patch_size, epoch_size):
        if not scene_ids:
            raise ValueError("At least one training scene is required")
        if patch_size <= 0 or epoch_size <= 0:
            raise ValueError("Patch size and epoch size must be positive")
        self.data_root = data_root
        self.scene_ids = list(scene_ids)
        self.patch_size = patch_size
        self.epoch_size = epoch_size
        self.cache = {}

    def __len__(self):
        return self.epoch_size

    def _load_scene(self, scene_id):
        if scene_id not in self.cache:
            image_path = (
                self.data_root / "images" / "train" / f"{scene_id}.tif"
            )
            label_path = (
                self.data_root / "labels" / "train" / f"{scene_id}.npz"
            )
            image = read_tiff(image_path)
            labels = read_class_map(label_path)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"Expected three-band IRRG image for {scene_id}, got {image.shape}"
                )
            if image.dtype != np.uint8:
                raise ValueError(
                    f"Expected uint8 IRRG pixels for {scene_id}, got {image.dtype}"
                )
            if image.shape[:2] != labels.shape:
                raise ValueError(
                    f"Image/label shape mismatch for {scene_id}: "
                    f"{image.shape[:2]} != {labels.shape}"
                )
            if min(image.shape[:2]) < self.patch_size:
                raise ValueError(f"Scene {scene_id} is smaller than the patch size")
            self.cache[scene_id] = image, labels
        return self.cache[scene_id]

    def __getitem__(self, _index):
        scene_id = random.choice(self.scene_ids)
        image, labels = self._load_scene(scene_id)
        height, width = labels.shape
        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)
        image = image[top : top + self.patch_size, left : left + self.patch_size]
        labels = labels[top : top + self.patch_size, left : left + self.patch_size]

        if random.random() < 0.5:
            image = image[::-1]
            labels = labels[::-1]
        if random.random() < 0.5:
            image = image[:, ::-1]
            labels = labels[:, ::-1]

        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        labels = np.ascontiguousarray(labels.astype(np.int64, copy=False))
        return torch.from_numpy(image), torch.from_numpy(labels)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_optimizer(model, learning_rate, decoder_learning_rate, weight_decay):
    lora_parameters = []
    decoder_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".lora_a" in name or ".lora_b" in name:
            lora_parameters.append(parameter)
        else:
            decoder_parameters.append(parameter)
    if not lora_parameters or not decoder_parameters:
        raise RuntimeError("Expected trainable LoRA and decoder parameters")
    return torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": learning_rate, "name": "lora"},
            {
                "params": decoder_parameters,
                "lr": decoder_learning_rate,
                "name": "decoder",
            },
        ],
        weight_decay=weight_decay,
    )


def build_warmup_cosine_scheduler(
    optimizer, total_steps, warmup_fraction, min_learning_rate
):
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")

    warmup_steps = int(total_steps * warmup_fraction)

    def learning_rate_lambda(base_learning_rate):
        if min_learning_rate <= 0 or min_learning_rate >= base_learning_rate:
            raise ValueError("min_learning_rate must be positive and below all base rates")

        def schedule(step):
            if warmup_steps and step < warmup_steps:
                return min((step + 1) / warmup_steps, 1.0)
            decay_steps = max(total_steps - warmup_steps - 1, 1)
            progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            learning_rate = min_learning_rate + (
                base_learning_rate - min_learning_rate
            ) * cosine
            return learning_rate / base_learning_rate

        return schedule

    schedules = [
        learning_rate_lambda(group["lr"]) for group in optimizer.param_groups
    ]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedules)
    return scheduler, warmup_steps


def cuda_smoke_test(model, dataset, class_weights, device):
    model.train()
    sample_inputs, sample_targets = dataset[0]
    inputs = sample_inputs.unsqueeze(0).to(device, dtype=torch.float32)
    targets = sample_targets.unsqueeze(0).to(device)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(inputs)
        loss = weighted_ce_macro_dice_loss(
            logits,
            targets,
            class_weights,
            ignore_index=IGNORE_LABEL,
            ce_weight=CE_LOSS_WEIGHT,
            dice_weight=DICE_LOSS_WEIGHT,
        )
    loss.backward()
    torch.cuda.synchronize(device)
    trainable_gradients = [
        parameter.grad
        for parameter in model.trainable_parameters()
        if parameter.grad is not None
    ]
    if not trainable_gradients or not all(
        torch.isfinite(gradient).all().item() for gradient in trainable_gradients
    ):
        raise RuntimeError("SAM CUDA smoke run produced missing or non-finite gradients")
    smoke = {
        "input_shape": list(inputs.shape),
        "logit_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "trainable_parameter_count": model.trainable_parameter_count(),
        "gradient_tensors": len(trainable_gradients),
        "seconds": time.perf_counter() - started,
        "device": torch.cuda.get_device_name(device),
        "passed": True,
    }
    model.zero_grad(set_to_none=True)
    print("CUDA smoke: " + json.dumps(smoke, sort_keys=True), flush=True)
    return smoke


def train_one_epoch(
    model, loader, optimizer, scheduler, scaler, class_weights, device, epoch
):
    model.train()
    losses = []
    started = time.perf_counter()
    for step, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(inputs)
            loss = weighted_ce_macro_dice_loss(
                logits,
                targets,
                class_weights,
                ignore_index=IGNORE_LABEL,
                ce_weight=CE_LOSS_WEIGHT,
                dice_weight=DICE_LOSS_WEIGHT,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before_step:
            scheduler.step()
        losses.append(float(loss.detach().cpu()))
        if step % 32 == 0 or step == len(loader):
            print(
                f"epoch={epoch:02d} step={step:04d}/{len(loader)} "
                f"loss_window={np.mean(losses[-32:]):.5f} "
                f"elapsed_seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )
    torch.cuda.synchronize(device)
    learning_rates = {
        group.get("name", str(index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }
    return float(np.mean(losses)), time.perf_counter() - started, learning_rates


def confusion_counts(labels, predictions):
    valid = labels != IGNORE_LABEL
    if not np.any(valid):
        return np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    truth = labels[valid].astype(np.int64, copy=False)
    predicted = predictions[valid].astype(np.int64, copy=False)
    return np.bincount(
        truth * N_CLASSES + predicted, minlength=N_CLASSES * N_CLASSES
    ).reshape(N_CLASSES, N_CLASSES)


def summarize_confusion(matrix):
    true_positives = np.diag(matrix).astype(np.float64)
    unions = matrix.sum(axis=0) + matrix.sum(axis=1) - true_positives
    iou = np.divide(
        true_positives,
        unions,
        out=np.full(N_CLASSES, np.nan, dtype=np.float64),
        where=unions > 0,
    )
    total = int(matrix.sum())
    return {
        "pixels_scored": total,
        "overall_accuracy_percent": (
            float(true_positives.sum() * 100.0 / total) if total else None
        ),
        "mean_iou_percent": (
            float(np.nanmean(iou) * 100.0) if np.any(np.isfinite(iou)) else None
        ),
        "per_class_iou_percent": {
            str(class_id): (
                float(iou[class_id] * 100.0)
                if np.isfinite(iou[class_id])
                else None
            )
            for class_id in range(N_CLASSES)
        },
        "confusion_matrix": matrix.tolist(),
    }


def pad_image_patch(image, patch_size):
    height, width = image.shape[:2]
    pad_height = patch_size - height
    pad_width = patch_size - width
    if pad_height < 0 or pad_width < 0:
        raise ValueError("Image patch exceeds the configured patch size")
    if pad_height or pad_width:
        image = np.pad(
            image,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode="edge",
        )
    return image


def save_preview(image, labels, predictions, output_path):
    from PIL import Image

    palette = np.zeros((N_CLASSES, 3), dtype=np.uint8)
    for class_id in range(N_CLASSES):
        palette[class_id] = PALETTE[class_id]
    base = image.astype(np.float32)
    color_mask = palette[predictions.astype(np.int64)]
    overlay = np.clip(base * 0.55 + color_mask.astype(np.float32) * 0.45, 0, 255)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay.astype(np.uint8)).save(output_path)
    Image.fromarray(color_mask).save(output_path.with_name("prediction_" + output_path.name))
    truth_mask = np.zeros_like(color_mask)
    valid = labels != IGNORE_LABEL
    truth_mask[valid] = palette[labels[valid].astype(np.int64)]
    Image.fromarray(truth_mask).save(output_path.with_name("ground_truth_" + output_path.name))


@torch.no_grad()
def evaluate_split(
    model,
    data_root,
    scene_ids,
    split_name,
    patch_size,
    device,
    output_dir,
    *,
    save_previews=True,
):
    matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    model.eval()
    preview_saved = False
    scene_results = []
    for scene_index, scene_id in enumerate(scene_ids, start=1):
        image_path = data_root / "images" / split_name / f"{scene_id}.tif"
        label_path = data_root / "labels" / split_name / f"{scene_id}.npz"
        image = read_tiff(image_path)
        labels = read_class_map(label_path)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Invalid IRRG image for {scene_id}: {image.shape} {image.dtype}")
        if image.shape[:2] != labels.shape:
            raise ValueError(f"Image/label shape mismatch for {scene_id}")

        scene_matrix = np.zeros_like(matrix)
        height, width = labels.shape
        for top in range(0, height, patch_size):
            for left in range(0, width, patch_size):
                input_patch = image[
                    top : top + patch_size, left : left + patch_size
                ]
                input_patch = pad_image_patch(input_patch, patch_size)
                inputs = torch.from_numpy(
                    np.ascontiguousarray(input_patch.transpose(2, 0, 1))
                ).unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(inputs)
                predicted = logits.argmax(dim=1)[0].cpu().numpy()
                valid_height = min(patch_size, height - top)
                valid_width = min(patch_size, width - left)
                truth = labels[top : top + valid_height, left : left + valid_width]
                predicted = predicted[:valid_height, :valid_width]
                scene_matrix += confusion_counts(truth, predicted)

                if (
                    save_previews
                    and not preview_saved
                    and scene_id == scene_ids[0]
                    and top == 0
                    and left == 0
                ):
                    save_preview(
                        image[:valid_height, :valid_width],
                        truth,
                        predicted,
                        output_dir / f"{split_name}_{scene_id}_overlay.png",
                    )
                    preview_saved = True
        matrix += scene_matrix
        scene_metrics = summarize_confusion(scene_matrix)
        scene_metrics["scene_id"] = scene_id
        scene_results.append(scene_metrics)
        print(
            f"{split_name} scene={scene_id} "
            f"({scene_index}/{len(scene_ids)}) "
            f"accuracy={scene_metrics['overall_accuracy_percent']} "
            f"mIoU={scene_metrics['mean_iou_percent']}",
            flush=True,
        )
        del image, labels

    result = summarize_confusion(matrix)
    result["scene_ids"] = list(scene_ids)
    result["scenes"] = scene_results
    result["metric"] = "overall pixel accuracy and mean IoU; label 6 ignored"
    print(
        f"{split_name} aggregate: "
        + json.dumps(
            {
                "accuracy": result["overall_accuracy_percent"],
                "mean_iou": result["mean_iou_percent"],
                "pixels_scored": result["pixels_scored"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def trainable_state_dict(model):
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def restore_trainable_state(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    missing_trainable = sorted(trainable_names.intersection(missing))
    if missing_trainable or unexpected:
        raise RuntimeError(
            "Resume adapter state mismatch: "
            f"missing trainable={missing_trainable}, unexpected={unexpected}"
        )


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint_atomic(path, checkpoint):
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def write_artifact_sha256_manifest(output_dir):
    records = []
    for path in sorted(Path(output_dir).rglob("*")):
        if not path.is_file() or path.name == "artifact_sha256_manifest.json":
            continue
        records.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    atomic_json_write(
        Path(output_dir) / "artifact_sha256_manifest.json",
        {"schema_version": 1, "files": records},
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH
    )
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--decoder-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args()


def main():
    args = parse_args()
    import cv2
    import matplotlib
    import PIL
    import tifffile
    import tqdm

    runtime = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "tifffile_version": tifffile.__version__,
        "pillow_version": PIL.__version__,
        "opencv_version": cv2.__version__,
        "matplotlib_version": matplotlib.__version__,
        "tqdm_version": tqdm.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
    }
    print("runtime_preflight=" + json.dumps(runtime, sort_keys=True), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("This Kaggle SAM training run requires a CUDA GPU")
    if args.epochs <= 0 or args.steps_per_epoch <= 0:
        raise ValueError("Epochs and steps per epoch must be positive")
    if args.early_stopping_patience <= 0:
        raise ValueError("early-stopping-patience must be positive")
    if args.learning_rate <= 0 or args.decoder_learning_rate <= 0:
        raise ValueError("Learning rates must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative")
    if not 0 <= args.warmup_fraction < 1:
        raise ValueError("warmup-fraction must be in [0, 1)")
    if args.min_learning_rate <= 0 or args.min_learning_rate >= min(
        args.learning_rate, args.decoder_learning_rate
    ):
        raise ValueError("min-learning-rate must be positive and below both base rates")
    if args.batch_size != 1:
        raise ValueError(
            "The validated Kaggle memory profile uses batch size 1; "
            "change it only after a CUDA smoke run"
        )
    if args.patch_size % 16:
        raise ValueError("Patch size must be divisible by SAM ViT-B's 16-pixel patch")

    data_root = find_data_root(args.data_root)
    bundle_manifest = verify_data_bundle(data_root)
    split = bundle_manifest["split"]
    train_ids = split[TRAIN_IDS_KEY]
    validation_ids = split[VALIDATION_IDS_KEY]
    test_ids = split[TEST_IDS_KEY]
    if (len(train_ids), len(validation_ids), len(test_ids)) != (18, 6, 14):
        raise ValueError("Expected the repository's fixed 18/6/14 scene split")

    train_class_counts = compute_training_class_counts(data_root, train_ids)
    train_class_frequencies = train_class_counts / train_class_counts.sum()
    class_weights = inverse_sqrt_class_weights(train_class_frequencies)

    sam_package_root = ensure_segment_anything()

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / args.run_id
    ).expanduser().resolve()
    manifest_path = output_dir / "run_manifest.json"
    if args.resume_from:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Resume requires the original run manifest in the output directory"
            )
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("run_id") != args.run_id:
            raise RuntimeError("Resume output directory belongs to a different run ID")
    elif output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint or DEFAULT_CHECKPOINT_PATH
    checkpoint_path = checkpoint_path.expanduser().resolve()
    download_checkpoint(checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)

    seed_everything(args.seed)
    device = torch.device("cuda")
    model = SAMViTSemanticSegmenter.from_pretrained(
        checkpoint_path,
        num_classes=N_CLASSES,
        rank=args.rank,
        alpha=args.alpha,
        gradient_checkpointing=True,
    ).to(device)
    optimizer = build_optimizer(
        model,
        learning_rate=args.learning_rate,
        decoder_learning_rate=args.decoder_learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler, warmup_steps = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=args.epochs * args.steps_per_epoch,
        warmup_fraction=args.warmup_fraction,
        min_learning_rate=args.min_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda")

    training_config = {
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "target_epochs": args.epochs,
        "max_optimizer_steps": args.epochs * args.steps_per_epoch,
        "sampler": "uniform random training scene and aligned random crop per step",
        "random_patches_per_epoch": args.steps_per_epoch,
        "seed": args.seed,
        "optimizer": "AdamW",
        "learning_rates": {
            "lora": args.learning_rate,
            "decoder": args.decoder_learning_rate,
        },
        "weight_decay": args.weight_decay,
        "scheduler": {
            "name": "linear_warmup_then_cosine",
            "warmup_fraction": args.warmup_fraction,
            "warmup_steps": warmup_steps,
            "total_steps": args.epochs * args.steps_per_epoch,
            "minimum_learning_rate": args.min_learning_rate,
        },
        "early_stopping_patience": args.early_stopping_patience,
        "loss": "0.7 * weighted cross entropy + 0.3 * macro soft Dice",
        "loss_coefficients": {
            "weighted_cross_entropy": CE_LOSS_WEIGHT,
            "macro_soft_dice": DICE_LOSS_WEIGHT,
        },
        "class_weighting": "mean-one inverse-square-root frequency, clipped to [0.5, 2.0]",
        "class_counts_train": train_class_counts.tolist(),
        "class_frequencies_train": train_class_frequencies.tolist(),
        "class_weights": class_weights.tolist(),
        "ignore_label": IGNORE_LABEL,
    }
    training_config_sha256 = hashlib.sha256(
        json.dumps(training_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    start_epoch = 1
    training_history = []
    validation_history = []
    best_validation_miou = None
    best_validation_metrics = None
    best_epoch = None
    epochs_without_improvement = 0
    created_utc = datetime.now(UTC).isoformat()
    saved = None
    if args.resume_from:
        saved = torch.load(
            args.resume_from.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if saved.get("sam_checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError("Resume checkpoint uses a different SAM base checkpoint")
        if saved.get("dataset_bundle_sha256") != bundle_manifest["bundle_sha256"]:
            raise RuntimeError("Resume checkpoint belongs to a different data bundle")
        if saved.get("training_config_sha256") != training_config_sha256:
            raise RuntimeError("Resume checkpoint has a different training configuration")
        if saved.get("run_id") != args.run_id:
            raise RuntimeError("Resume checkpoint belongs to a different run ID")
        restore_trainable_state(model, saved["trainable_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        restore_rng_state(saved["rng_state"])
        start_epoch = int(saved["epoch"]) + 1
        training_history = saved.get("training_history", [])
        validation_history = saved.get("validation_history", [])
        best_validation_miou = saved.get("best_validation_miou")
        best_validation_metrics = saved.get("best_validation_metrics")
        best_epoch = saved.get("best_epoch")
        epochs_without_improvement = saved.get("epochs_without_improvement", 0)
        created_utc = run_manifest.get("created_utc", created_utc)
        if start_epoch > args.epochs:
            raise ValueError(
                f"Resume checkpoint already completed epoch {start_epoch - 1}, "
                f"target is {args.epochs}"
            )

    train_dataset = PotsdamPatchDataset(
        data_root, train_ids, args.patch_size, args.steps_per_epoch
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    rng_state_before_smoke = capture_rng_state()
    smoke = cuda_smoke_test(model, train_dataset, class_weights, device)
    restore_rng_state(rng_state_before_smoke)

    if saved is not None:
        best_checkpoint_path = output_dir / "best.pt"
        if best_epoch is None or not best_checkpoint_path.is_file():
            raise RuntimeError("Resume checkpoint has no matching best checkpoint")

    if saved is None:
        run_manifest = {
            "run_id": args.run_id,
            "status": "running",
            "created_utc": created_utc,
            "sam_model": "SAM ViT-B",
            "sam_repository": SAM_REPOSITORY,
            "sam_repository_commit": SAM_COMMIT,
            "sam_package_root": str(sam_package_root),
            "sam_checkpoint_url": SAM_CHECKPOINT_URL,
            "sam_checkpoint_sha256": checkpoint_sha,
            "trainer_script_sha256": sha256_file(Path(__file__).resolve()),
            "source_package_manifest_sha256": SOURCE_PACKAGE_MANIFEST_SHA256,
            "model_wrapper_sha256": sha256_file(
                PROJECT_ROOT / "semantic_labeling" / "sam_lora.py"
            ),
            "class_contract_sha256": sha256_file(
                PROJECT_ROOT / "semantic_labeling" / "constants.py"
            ),
            "loss_module_sha256": sha256_file(
                PROJECT_ROOT / "semantic_labeling" / "losses.py"
            ),
            "dataset_bundle_sha256": bundle_manifest["bundle_sha256"],
            "source_git_revision": bundle_manifest.get("source_git_revision"),
            "source_worktree_changes": bundle_manifest.get(
                "source_worktree_changes", []
            ),
            "backend": "kaggle",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_count": torch.cuda.device_count(),
            "python_version": platform.python_version(),
            "runtime_preflight": runtime,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_version": torch.version.cuda,
            "train_ids": train_ids,
            "validation_ids": validation_ids,
            "held_out_test_ids": test_ids,
            "input": {
                "bands": "three-channel Potsdam IRRG TIFF, in stored channel order",
                "uses_dsm": False,
                "uses_ndsm": False,
                "normalization": "official SAM ViT-B pixel_mean and pixel_std in 0..255 scale",
                "note": "IRRG is not natural RGB; spectral domain differs from SAM pretraining",
            },
            "task": "supervised six-class semantic segmentation",
            "classes": N_CLASSES,
            "class_names": list(LABELS),
            "ignore_label": IGNORE_LABEL,
            "lora": {
                "targets": [
                    "image_encoder.blocks.*.attn.qkv",
                    "image_encoder.blocks.*.attn.proj",
                ],
                "rank": args.rank,
                "alpha": args.alpha,
                "original_encoder_weights_frozen": True,
            },
            "decoder": "four ConvNormAct upsampling stages and a six-class 1x1 head",
            "trainable_parameter_count": model.trainable_parameter_count(),
            "training_config": training_config,
            "training_config_sha256": training_config_sha256,
            "starting_epoch": start_epoch,
            "early_stopped": False,
            "cuda_smoke": smoke,
            "data_root": str(data_root),
            "run_directory": str(output_dir),
        }
    else:
        run_manifest.update(
            {
                "status": "running",
                "training_config": training_config,
                "training_config_sha256": training_config_sha256,
                "cuda_smoke": smoke,
            }
        )
    atomic_json_write(manifest_path, run_manifest)

    def checkpoint_payload(epoch, latest_validation_metrics):
        return {
            "epoch": epoch,
            "trainable_state_dict": trainable_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "training_history": training_history,
            "validation_history": validation_history,
            "best_validation_miou": best_validation_miou,
            "best_validation_metrics": best_validation_metrics,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "validation_metrics": latest_validation_metrics,
            "sam_checkpoint_sha256": checkpoint_sha,
            "dataset_bundle_sha256": bundle_manifest["bundle_sha256"],
            "training_config_sha256": training_config_sha256,
            "run_id": run_manifest["run_id"],
            "created_utc": created_utc,
        }

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            mean_loss, seconds, learning_rates = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                class_weights,
                device,
                epoch,
            )
            epoch_record = {
                "epoch": epoch,
                "mean_loss": mean_loss,
                "seconds": seconds,
                "learning_rates": learning_rates,
                "completed_utc": datetime.now(UTC).isoformat(),
            }
            training_history.append(epoch_record)

            validation = evaluate_split(
                model,
                data_root,
                validation_ids,
                "validation",
                args.patch_size,
                device,
                output_dir,
                save_previews=False,
            )
            validation_score = validation["mean_iou_percent"]
            if validation_score is None or not math.isfinite(validation_score):
                raise RuntimeError("Validation did not produce a finite six-class mIoU")
            validation_record = {
                "epoch": epoch,
                "mean_iou_percent": validation_score,
                "metrics": validation,
                "completed_utc": datetime.now(UTC).isoformat(),
            }
            validation_history.append(validation_record)

            improved = (
                best_validation_miou is None
                or validation_score > best_validation_miou
            )
            if improved:
                best_validation_miou = validation_score
                best_validation_metrics = validation
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            checkpoint = checkpoint_payload(epoch, validation)
            if improved:
                save_checkpoint_atomic(output_dir / "best.pt", checkpoint)
            save_checkpoint_atomic(output_dir / "last.pt", checkpoint)

            with (output_dir / "training_log.jsonl").open(
                "a", encoding="utf-8"
            ) as log:
                log.write(json.dumps(epoch_record, sort_keys=True) + "\n")
            with (output_dir / "validation_log.jsonl").open(
                "a", encoding="utf-8"
            ) as log:
                log.write(json.dumps(validation_record, sort_keys=True) + "\n")

            run_manifest.update(
                {
                    "last_completed_epoch": epoch,
                    "last_epoch_loss": mean_loss,
                    "last_validation_miou_percent": validation_score,
                    "best_epoch": best_epoch,
                    "best_validation_miou_percent": best_validation_miou,
                    "epochs_without_improvement": epochs_without_improvement,
                    "training_history": training_history,
                    "validation_history": validation_history,
                }
            )
            atomic_json_write(manifest_path, run_manifest)
            print(
                f"completed epoch={epoch}/{args.epochs} loss={mean_loss:.5f} "
                f"validation_mIoU={validation_score:.4f}% "
                f"best_epoch={best_epoch} minutes={seconds / 60.0:.1f}",
                flush=True,
            )
            if epochs_without_improvement >= args.early_stopping_patience:
                run_manifest["early_stopped"] = True
                print(
                    f"early stopping after {epochs_without_improvement} epochs "
                    "without validation mIoU improvement",
                    flush=True,
                )
                break

        best_checkpoint_path = output_dir / "best.pt"
        if not best_checkpoint_path.is_file():
            raise RuntimeError("Training ended without a validation-best checkpoint")
        best_checkpoint = torch.load(
            best_checkpoint_path, map_location="cpu", weights_only=False
        )
        if best_checkpoint.get("training_config_sha256") != training_config_sha256:
            raise RuntimeError("Best checkpoint has a different training configuration")
        restore_trainable_state(model, best_checkpoint["trainable_state_dict"])

        validation = evaluate_split(
            model,
            data_root,
            validation_ids,
            "validation",
            args.patch_size,
            device,
            output_dir,
            save_previews=True,
        )
        selected_validation = best_checkpoint.get("best_validation_metrics")
        if (
            selected_validation is None
            or validation["confusion_matrix"]
            != selected_validation["confusion_matrix"]
        ):
            raise RuntimeError(
                "Re-evaluated validation confusion matrix differs from best checkpoint"
            )
        if not math.isclose(
            validation["mean_iou_percent"],
            best_checkpoint["best_validation_miou"],
            rel_tol=0,
            abs_tol=1e-5,
        ):
            raise RuntimeError("Re-evaluated validation mIoU differs from best checkpoint")

        held_out = evaluate_split(
            model,
            data_root,
            test_ids,
            "held_out",
            args.patch_size,
            device,
            output_dir,
            save_previews=True,
        )
        baseline = {
            "run_id": "sam-vitb-lora-potsdam-20261020-rerun-02",
            "validation_miou_percent": 62.70276804696081,
            "held_out_miou_percent": 60.021338817378165,
        }
        results = {
            "selected_checkpoint": "best.pt",
            "selected_checkpoint_sha256": sha256_file(best_checkpoint_path),
            "selected_epoch": best_checkpoint["best_epoch"],
            "selection_metric": "six-class validation mean IoU; label 6 ignored",
            "validation": validation,
            "held_out_test": held_out,
            "baseline_comparison": {
                **baseline,
                "validation_miou_delta_percentage_points": (
                    validation["mean_iou_percent"]
                    - baseline["validation_miou_percent"]
                ),
                "held_out_miou_delta_percentage_points": (
                    held_out["mean_iou_percent"] - baseline["held_out_miou_percent"]
                ),
                "held_out_used_for_selection": False,
            },
            "evaluated_utc": datetime.now(UTC).isoformat(),
        }
        atomic_json_write(output_dir / "metrics.json", results)
        run_manifest.update(
            {
                "status": "complete",
                "best_epoch": best_checkpoint["best_epoch"],
                "early_stopped": bool(run_manifest.get("early_stopped", False)),
                "best_validation_miou_percent": validation["mean_iou_percent"],
                "metrics": results,
                "completed_utc": datetime.now(UTC).isoformat(),
                "best_checkpoint_sha256": sha256_file(best_checkpoint_path),
                "last_checkpoint_sha256": sha256_file(output_dir / "last.pt"),
            }
        )
        atomic_json_write(manifest_path, run_manifest)
        write_artifact_sha256_manifest(output_dir)
        print(json.dumps(results, indent=2), flush=True)
    except Exception as error:
        run_manifest.update(
            {
                "status": "failed",
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "last_completed_epoch": run_manifest.get("last_completed_epoch"),
                "last_checkpoint_sha256": (
                    sha256_file(output_dir / "last.pt")
                    if (output_dir / "last.pt").is_file()
                    else None
                ),
                "failed_utc": datetime.now(UTC).isoformat(),
            }
        )
        atomic_json_write(manifest_path, run_manifest)
        raise


if __name__ == "__main__":
    main()
