"""Shared random-patch dataset for the Potsdam experiment notebooks."""

import random

import numpy as np
import torch
from skimage import io

from .constants import INPUT_SCALE
from .labels import convert_from_color
from .preprocessing import assemble_input, normalize_input
from .sampling import get_random_pos

DEFAULT_MAIN_FOLDER = "../input/potsdamvaihingen/"
DEFAULT_DATA_FOLDER = (
    DEFAULT_MAIN_FOLDER
    + "3_ortho_irrg/3_Ortho_IRRG/top_potsdam_{}_IRRG.tif"
)
DEFAULT_DSM_FOLDER = DEFAULT_MAIN_FOLDER + "1_dsm/1_DSM/dsm_potsdam_0{}.tif"
DEFAULT_NDSM_FOLDER = (
    DEFAULT_MAIN_FOLDER
    + "1_dsm_normalisation/1_DSM_normalisation/"
    + "dsm_potsdam_0{}_normalized_lastools.jpg"
)
DEFAULT_LABEL_FOLDER = (
    "../input/label-all-potsdam-vaihingen/5_labels_all/"
    + "top_potsdam_{}_label.tif"
)


class ISPRS_dataset(torch.utils.data.Dataset):
    """Sample random, augmented patches from aligned Potsdam raster paths."""

    def __init__(
        self,
        train_ids,
        dsm_ids,
        data_files=DEFAULT_DATA_FOLDER,
        label_files=DEFAULT_LABEL_FOLDER,
        dsm_files=DEFAULT_DSM_FOLDER,
        ndsm_files=DEFAULT_NDSM_FOLDER,
        cache=False,
        augmentation=True,
        window_size=(256, 256),
        epoch_size=10000,
        input_scale=INPUT_SCALE,
    ):
        if not train_ids:
            raise ValueError("At least one training tile is required")
        if len(train_ids) != len(dsm_ids):
            raise ValueError("TOP/label and DSM/nDSM tile lists must have equal lengths")
        if epoch_size <= 0:
            raise ValueError(f"Epoch size must be positive, got {epoch_size}")

        self.data_files = [data_files.format(tile_id) for tile_id in train_ids]
        self.label_files = [label_files.format(tile_id) for tile_id in train_ids]
        self.dsm_files = [dsm_files.format(tile_id) for tile_id in dsm_ids]
        self.ndsm_files = [ndsm_files.format(tile_id) for tile_id in dsm_ids]
        self.cache = cache
        self.augmentation = augmentation
        self.window_size = window_size
        self.epoch_size = epoch_size
        self.input_scale = input_scale
        self.data_cache = {}
        self.label_cache = {}

    def __len__(self):
        return self.epoch_size

    @staticmethod
    def data_augmentation(*arrays, flip=True, mirror=True):
        will_flip = flip and random.random() < 0.5
        will_mirror = mirror and random.random() < 0.5
        results = []
        for array in arrays:
            if will_flip:
                array = array[::-1, :] if array.ndim == 2 else array[:, ::-1, :]
            if will_mirror:
                array = array[:, ::-1] if array.ndim == 2 else array[:, :, ::-1]
            results.append(np.ascontiguousarray(array))
        return tuple(results)

    def _load_data(self, index):
        if self.cache and index in self.data_cache:
            return self.data_cache[index]

        top = io.imread(self.data_files[index])
        dsm = io.imread(self.dsm_files[index])
        ndsm = io.imread(self.ndsm_files[index])
        image = assemble_input(top, dsm, ndsm)
        image = normalize_input(image, self.input_scale)
        data = np.ascontiguousarray(image.transpose(2, 0, 1))
        if self.cache:
            self.data_cache[index] = data
        return data

    def _load_label(self, index):
        if self.cache and index in self.label_cache:
            return self.label_cache[index]

        label = convert_from_color(io.imread(self.label_files[index]))
        if self.cache:
            self.label_cache[index] = label
        return label

    def __getitem__(self, index):
        # Each index denotes one random sample from the configured epoch.
        tile_index = random.randrange(len(self.data_files))
        data = self._load_data(tile_index)
        label = self._load_label(tile_index)

        top, bottom, left, right = get_random_pos(data, self.window_size)
        data_patch = data[:, top:bottom, left:right]
        label_patch = label[top:bottom, left:right]
        if self.augmentation:
            data_patch, label_patch = self.data_augmentation(data_patch, label_patch)
        else:
            data_patch = np.ascontiguousarray(data_patch)
            label_patch = np.ascontiguousarray(label_patch)

        return torch.from_numpy(data_patch), torch.from_numpy(label_patch.astype(np.int64, copy=False))
