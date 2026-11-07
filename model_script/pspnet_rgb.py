"""PSPNet comparison on the frozen Potsdam RGB study data and schedule."""

from __future__ import annotations

import logging
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from deepnet.data.contracts import canonical_hash
from deepnet.data.evaluation import summarize_confusion
from deepnet.data.raster import sha256_file
from deepnet.potsdam_rgbir.data import CacheBundle
from deepnet.potsdam_rgbir.engine import (
    fit_small,
    restore_panel,
    score_patches,
    score_scenes,
    timed,
    train_step,
)
from deepnet.potsdam_rgbir.phases import check_refit_cache
from deepnet.potsdam_rgbir.protocol import (
    Budget,
    freeze,
    load_protocol,
    matrix_metrics,
    read_json,
)
from deepnet.potsdam_rgbir.sampling import loader, schedule, training_index
from deepnet.runtime import configure_torch_runtime, seed_everything
from deepnet.training_utils import load_checkpoint, save_checkpoint, write_json
from semantic_labeling.losses import inverse_sqrt_class_weights

STUDY = "potsdam-pspnet-rgb-v1"
WEIGHT_NAME = "resnet101-cd907fc2.pth"
WEIGHT_SHA = "cd907fc2a0de2292b2ba2b27d2216bc672d84632d1cf20058d01c0c03b5e020e"
HOURS = {"selection": 12, "refit": 3, "benchmark": 2}
ROLES = {
    "selection": ["train", "validation"],
    "refit": ["train"],
    "benchmark": ["test"],
}
PREVIOUS = {"refit": "selection", "benchmark": "refit"}
LOG = logging.getLogger(__name__)


def conv(in_channels, out_channels, kernel=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.ReLU(inplace=True),
    )


class RGBPSPNet(nn.Module):
    """ImageNet RGB ResNet101 with pyramid pooling and a stride-four detail path."""

    def __init__(self, weights=None):
        super().__init__()
        from torchvision.models import resnet101

        self.encoder = resnet101(
            weights=None, replace_stride_with_dilation=[False, False, True]
        )
        if weights is not None:
            if sha256_file(weights) != WEIGHT_SHA:
                raise ValueError("ResNet101 ImageNet-1K V2 checksum differs")
            self.encoder.load_state_dict(
                torch.load(weights, map_location="cpu", weights_only=True)
            )
        del self.encoder.fc
        self.reduce = conv(2048, 256)
        self.pyramid = nn.ModuleList(
            nn.Sequential(nn.AdaptiveAvgPool2d(n), conv(256, 64)) for n in (1, 2, 3, 6)
        )
        self.context = conv(512, 128, 3)
        self.detail = conv(256, 32)
        self.head = nn.Sequential(
            conv(160, 128, 3), nn.Dropout2d(0.1), nn.Conv2d(128, 6, 1)
        )
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def train(self, mode=True):
        super().train(mode)
        # Retain pretrained running statistics with microbatch one.
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    @staticmethod
    def resize(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, image):
        x = (image - self.mean) / self.std
        e = self.encoder
        detail = e.layer1(e.maxpool(e.relu(e.bn1(e.conv1(x)))))
        top = self.reduce(e.layer4(e.layer3(e.layer2(detail))))
        context = torch.cat(
            [
                top,
                *(self.resize(branch(top), top.shape[-2:]) for branch in self.pyramid),
            ],
            1,
        )
        context = self.resize(self.context(context), detail.shape[-2:])
        return self.resize(
            self.head(torch.cat((context, self.detail(detail)), 1)), image.shape[-2:]
        )


def optimizer(model, runtime, protocol, refit):
    factor = 0.5 if refit else 1
    groups = [
        {
            "params": list(model.encoder.parameters()),
            "lr": factor * 1e-4,
            "initial_lr": factor * 1e-4,
        },
        {
            "params": [
                p
                for name, p in model.named_parameters()
                if not name.startswith("encoder.")
            ],
            "lr": factor * 3e-4,
            "initial_lr": factor * 3e-4,
        },
    ]
    return torch.optim.AdamW(
        groups, weight_decay=protocol["optimizer"]["weight_decay"]
    ), torch.amp.GradScaler(
        "cuda", init_scale=1024, enabled=runtime.device_type == "cuda"
    )


def new_model(runtime, weights=None):
    return RGBPSPNet(weights).to(
        device=runtime.device, memory_format=torch.channels_last
    )


def draws_for(train, index, protocol, pass_number, runtime):
    draws = schedule(
        index,
        train,
        protocol["seed"],
        pass_number,
        jitter=protocol["sampling"]["center_jitter"],
    )
    return loader(
        train,
        draws,
        protocol["seed"],
        pass_number,
        protocol["augmentation"]["baseline"],
        workers=runtime.num_workers,
        cuda=runtime.device_type == "cuda",
    )


def pilot(out, cache, protocol, runtime, weights, pretrained, index, budget):
    model = new_model(runtime, pretrained)
    with (
        Budget(out / "pilot", 1, budget.devices) as trial,
        cache.access("train", auxiliary=False) as train,
    ):
        fit = fit_small(model, train, index, runtime, protocol, trial, weights)
        write_json(out / "pilot-fit.json", fit)
        if not fit["passed"]:
            raise ValueError("PSPNet did not pass the eight-patch 90% train mIoU gate")
        opt, scaler = optimizer(model, runtime, protocol, False)
        iterator = iter(draws_for(train, index, protocol, 0, runtime))
        model.train()
        durations = []
        for step in range(24):
            trial.check()
            start = time.monotonic()
            train_step(
                model,
                [next(iterator) for _ in range(4)],
                opt,
                scaler,
                weights,
                runtime,
                protocol,
                step,
                864,
            )
            torch.cuda.synchronize(runtime.device)
            durations.append(time.monotonic() - start)
        del iterator
        with cache.access("validation", auxiliary=False) as val:
            _, probe = timed(
                lambda: score_patches(
                    model,
                    val,
                    restore_panel(protocol["probe_panel"]),
                    runtime,
                    protocol,
                    trial,
                ),
                runtime,
            )
            _, scene = timed(
                lambda: score_scenes(
                    model,
                    val,
                    runtime,
                    protocol,
                    trial,
                    out / ".scratch",
                    scene_ids=[next(iter(val.scene_shapes))],
                ),
                runtime,
            )
    update = statistics.median(durations[4:])
    # Account for all allocated GPUs and reserve setup/serialization time per phase.
    forecasts = {
        "selection": 1.25
        * (protocol["passes"] * (864 * update + probe + 12) + 12 * scene + 180),
        "refit": 1.25 * (1152 * update + 240),
        "benchmark": 1.25 * (14 * scene + 240),
    }
    report = {
        "update_seconds": update,
        "probe_seconds": probe,
        "scene_seconds": scene,
        "forecast_gpu_hours": {
            phase: v * budget.devices / 3600 for phase, v in forecasts.items()
        },
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(runtime.device),
    }
    write_json(out / "pilot.json", report)
    if forecasts["selection"] > budget.remaining() or any(
        forecasts[p] * budget.devices > HOURS[p] * 3600 for p in ("refit", "benchmark")
    ):
        raise TimeoutError(
            "Matched eight-pass PSPNet schedule does not fit the phase budgets"
        )
    LOG.info(
        "PSPNet pilot passed; estimated allocated GPU-hours: %s",
        report["forecast_gpu_hours"],
    )


def verified_previous(path, phase, settings, protocol):
    info = read_json(path / "manifest.json")
    checkpoint = path / ("best.pt" if phase == "selection" else "final.pt")
    expected = 864 * protocol["passes"] if phase == "selection" else 1152
    if (
        info["status"] != "complete"
        or info["phase"] != phase
        or info["updates"] != expected
        or info["source_sha256"] != settings["source_sha256"]
        or info["reference_protocol"] != protocol["protocol_hash"]
        or info["checkpoint_sha256"] != sha256_file(checkpoint)
        or info["cache_hash"] != settings["cache_hashes"][phase]
    ):
        raise ValueError(
            f"Prerequisite {phase} does not match the completed comparison"
        )
    saved = load_checkpoint(checkpoint, info["binding"])
    expected_step = info["best_update"] if phase == "selection" else 1152
    if saved["progress"]["updates"] != expected_step:
        raise ValueError("Prerequisite checkpoint progress differs")
    return info, saved["model"]


def train(out, cache, protocol, runtime, settings, budget, reference, previous):
    phase = settings["phase"]
    refit = phase == "refit"
    passes, steps = (1, 1152) if refit else (protocol["passes"], 864)
    weights = torch.tensor(
        inverse_sqrt_class_weights(protocol["train_class_support"]), dtype=torch.float32
    )
    pretrained = None
    initial = None
    seed_everything(protocol["seed"])
    with cache.access("train", auxiliary=False) as access:
        if refit:
            check_refit_cache(cache, protocol)
            initial, state = verified_previous(
                previous, "selection", settings, protocol
            )
            index = training_index(access)
        else:
            index = read_json(reference / "training-index.json")
            if canonical_hash(index) != protocol["training_index_hash"]:
                raise ValueError("Selection training index differs from DINOv3")
            pretrained = Path("/tmp") / WEIGHT_NAME
            if not pretrained.exists():
                torch.hub.download_url_to_file(
                    f"https://download.pytorch.org/models/{WEIGHT_NAME}",
                    pretrained,
                    hash_prefix=WEIGHT_SHA,
                )
            pilot(out, cache, protocol, runtime, weights, pretrained, index, budget)
            seed_everything(protocol["seed"])
        model = new_model(runtime, pretrained)
        if initial is not None:
            model.load_state_dict(state)
            del state
        opt, scaler = optimizer(model, runtime, protocol, refit)
        binding = canonical_hash(
            {
                "phase": phase,
                "source": settings["source_sha256"],
                "reference": protocol["protocol_hash"],
                "weights": WEIGHT_SHA,
                "cache": cache.manifest["cache_hash"],
                "seed": protocol["seed"],
                "initial": initial["checkpoint_sha256"] if initial else None,
            }
        )
        history, best, update, presentations = [], None, 0, 0
        checkpoint = out / ("final.pt" if refit else "best.pt")
        for n in range(passes):
            dataset = draws_for(access, index, protocol, n, runtime)
            if len(dataset) != 4 * steps:
                raise ValueError(
                    "Coverage schedule differs from the matched experiment"
                )
            iterator = iter(dataset)
            model.train()
            matrix, loss_sum = np.zeros((6, 6), dtype=np.int64), 0.0
            for _ in range(steps):
                budget.check()
                loss, counts = train_step(
                    model,
                    [next(iterator) for _ in range(4)],
                    opt,
                    scaler,
                    weights,
                    runtime,
                    protocol,
                    update,
                    passes * steps,
                )
                update += 1
                matrix += counts
                loss_sum += loss
                if update % 128 == 0:
                    LOG.info(
                        "%s update=%d/%d loss=%.5f", phase, update, passes * steps, loss
                    )
                if update % 288 == 0:
                    save_checkpoint(
                        out / "last.pt",
                        model,
                        {"updates": update},
                        binding,
                        optimizer=opt,
                        scaler=scaler,
                    )
            del iterator, dataset
            presentations += int(matrix.sum())
            record = {
                "pass": n + 1,
                "updates": update,
                "train_loss": loss_sum / steps,
                "train_metrics": summarize_confusion(matrix),
            }
            if not refit:
                with cache.access("validation", auxiliary=False) as val:
                    record["probe"] = score_patches(
                        model,
                        val,
                        restore_panel(protocol["probe_panel"]),
                        runtime,
                        protocol,
                        budget,
                    )
                    if n + 1 in protocol["full_validation_passes"]:
                        metrics = score_scenes(
                            model, val, runtime, protocol, budget, out / ".scratch"
                        )
                        checked = matrix_metrics(
                            metrics["confusion_matrix"], protocol["validation_pixels"]
                        )
                        ranking = (
                            checked["overall_accuracy"],
                            checked["mean_iou"],
                            -update,
                        )
                        record["validation"] = metrics
                        if best is None or ranking > best:
                            best = ranking
                            save_checkpoint(
                                checkpoint, model, {"updates": update}, binding
                            )
                        LOG.info(
                            "Full validation update=%d OA=%.6f mIoU=%.6f",
                            update,
                            *ranking[:2],
                        )
            history.append(record)
            write_json(out / "history.json", history)
        if refit:
            save_checkpoint(checkpoint, model, {"updates": update}, binding)
        elif best is None:
            raise ValueError("Selection requires full validation")
        write_json(
            out / "manifest.json",
            {
                "status": "complete",
                "phase": phase,
                "updates": update,
                "passes": passes,
                "reference_protocol": protocol["protocol_hash"],
                "source_sha256": settings["source_sha256"],
                "cache_hash": cache.manifest["cache_hash"],
                "binding": binding,
                "checkpoint_sha256": sha256_file(checkpoint),
                "best_update": -best[2] if best else None,
                "initial_checkpoint_sha256": initial["checkpoint_sha256"]
                if initial
                else WEIGHT_SHA,
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "train_scene_ids": list(cache.split.train_scene_ids),
                "valid_pixel_presentations": presentations,
            },
        )


def benchmark(out, cache, protocol, runtime, settings, budget, previous):
    info, state = verified_previous(previous, "refit", settings, protocol)
    model = new_model(runtime)
    model.load_state_dict(state)
    if (
        sum(s["valid_pixels"] for s in cache.manifest["scenes"].values())
        != protocol["heldout_pixels"]
    ):
        raise ValueError("Historical benchmark support differs")
    marker = {
        "checkpoint_sha256": info["checkpoint_sha256"],
        "inference": "identity",
        "cache_hash": cache.manifest["cache_hash"],
        "reference_protocol": protocol["protocol_hash"],
        "heldout_evaluation_count": 1,
    }
    if (out / "attempt.json").exists():
        raise FileExistsError("Historical benchmark attempt already exists")
    freeze(out / "attempt.json", marker)
    with cache.access("test", auxiliary=False) as access:
        result, seconds = timed(
            lambda: score_scenes(
                model, access, runtime, protocol, budget, out / ".scratch", profile=True
            ),
            runtime,
        )
    matrix_metrics(result["confusion_matrix"], protocol["heldout_pixels"])
    write_json(out / "metrics.json", {**result, **marker, "inference_seconds": seconds})


def run(settings, cache_path, out, reference, previous, started):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    phase = settings["phase"]
    protocol = load_protocol(reference)
    lock = read_json(reference / "selection-lock.json")
    if (
        canonical_hash(lock) != settings["selection_lock_hash"]
        or lock["protocol_hash"] != protocol["protocol_hash"]
        or lock["recipe"] != {"backbone": "B", "auxiliary": False, "enhanced": False}
        or lock["inference"] != "identity"
        or protocol["passes"] != 8
    ):
        raise ValueError("Reference must be the frozen DINOv3 RGB winner")
    runtime = configure_torch_runtime("cuda", num_workers=2)
    devices = torch.cuda.device_count()
    if devices != settings["allocation"]:
        raise ValueError(f"Unexpected allocated device count: {devices}")
    with Budget(out, HOURS[phase], devices, started=started) as budget:
        cache = CacheBundle.load(cache_path)
        if (
            cache.manifest["included_splits"] != ROLES[phase]
            or cache.manifest["cache_hash"] != settings["cache_hashes"][phase]
            or cache.manifest["selection_split_hash"]
            != protocol["selection_split_hash"]
        ):
            raise ValueError("The attached cache does not match the locked DINOv3 role")
        if phase == "benchmark":
            benchmark(out, cache, protocol, runtime, settings, budget, previous)
        else:
            train(out, cache, protocol, runtime, settings, budget, reference, previous)
