"""Synchronized, mask-aware RGBIR/DSM augmentation in raw sensor space."""

from __future__ import annotations

import random
from dataclasses import replace

import cv2
import numpy as np


def augment(access, draw, *, seed: int, recipe: dict):
    rng = random.Random(seed)
    turns, flip = rng.randrange(4), rng.choice((False, True))
    affine = draw.rare_class is None and rng.random() < recipe["affine_probability"]
    halo = recipe["halo"] if affine else 0
    source = (
        access.read_context(draw.scene, draw.window, halo)
        if halo
        else access.read_sample(draw.scene, draw.window)
    )
    image, labels, valid = (
        source.image.copy(),
        source.labels.copy(),
        source.valid_mask.copy(),
    )
    optical_count = 4 if image.shape[0] == 6 else 3
    if affine:
        h, w = draw.window.height, draw.window.width
        offset_y = draw.window.row - source.window.row
        offset_x = draw.window.col - source.window.col
        angle = recipe["rotation"]
        low, high = recipe["scale"]
        translation = recipe["translation"]
        matrix = cv2.getRotationMatrix2D(
            (offset_x + (w - 1) / 2, offset_y + (h - 1) / 2),
            rng.uniform(-angle, angle),
            rng.uniform(low, high),
        )
        matrix[:, 2] += (
            -offset_x + rng.uniform(-translation, translation) * w,
            -offset_y + rng.uniform(-translation, translation) * h,
        )

        def warp(array, nearest=False, fill=0):
            return cv2.warpAffine(
                array,
                matrix,
                (w, h),
                flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=fill,
            )

        support = warp(valid.astype(np.float32))
        optical = [
            warp(c * valid) / np.maximum(support, 1e-6) for c in image[:optical_count]
        ]
        if optical_count == 4:
            available = image[5] * valid
            dsm_support = warp(available)
            mask = warp(available, nearest=True) > 0.5
            relief = np.where(
                mask, warp(image[4] * available) / np.maximum(dsm_support, 1e-6), 0
            )
            optical.extend((relief, mask))
        image = np.stack(optical).astype(np.float32)
        labels = warp(labels.astype(np.uint8), nearest=True, fill=6).astype(np.int64)
        valid = warp(valid.astype(np.uint8), nearest=True).astype(bool)
    image = np.rot90(image, turns, axes=(-2, -1))
    labels, valid = np.rot90(labels, turns), np.rot90(valid, turns)
    if flip:
        image, labels, valid = image[..., ::-1], labels[:, ::-1], valid[:, ::-1]
    image, labels, valid = image.copy(), labels.copy(), valid.copy()
    if rng.random() < recipe["gain_probability"]:
        low, high = recipe["gain_range"]
        image[:optical_count] = np.clip(
            image[:optical_count] * rng.uniform(low, high), 0, 1
        )
    if rng.random() < recipe["gamma_probability"]:
        image[:optical_count] = image[:optical_count] ** rng.uniform(
            *recipe["gamma_range"]
        )
    if rng.random() < recipe["noise_probability"]:
        noise = np.random.default_rng(seed).normal(
            0, recipe["noise_std"], image[:optical_count].shape
        )
        image[:optical_count] = np.clip(image[:optical_count] + noise, 0, 1)
    if optical_count == 4 and rng.random() < recipe["dsm_dropout"]:
        image[4:] = 0
    image[:, ~valid] = 0
    labels[~valid] = 6
    return replace(
        source, image=image, labels=labels, valid_mask=valid, window=draw.window
    )
