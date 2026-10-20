import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from model_script.train_sam_kaggle import (
    PotsdamPatchDataset,
    apply_spatial_augmentation,
    sample_crop_origin,
)


class PotsdamPatchSamplingTests(unittest.TestCase):
    def test_targeted_crop_contains_selected_clutter_pixel(self):
        labels = np.zeros((16, 16), dtype=np.uint8)
        labels[0, 0] = 5
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        dataset = PotsdamPatchDataset(
            Path(tempfile.gettempdir()),
            ["tile"],
            patch_size=8,
            epoch_size=1,
            clutter_patch_fraction=1.0,
        )
        dataset.cache["tile"] = image, labels

        for _ in range(10):
            _, patch_labels = dataset[0]
            self.assertEqual(patch_labels.shape, (8, 8))
            self.assertTrue(torch.any(patch_labels == 5))

    def test_crop_origin_contains_requested_point_at_each_edge(self):
        for point in ((0, 0), (0, 9), (9, 0), (9, 9), (5, 6)):
            top, left = sample_crop_origin(10, 10, 4, target_point=point)
            self.assertLessEqual(top, point[0])
            self.assertLess(point[0], top + 4)
            self.assertLessEqual(left, point[1])
            self.assertLess(point[1], left + 4)

    def test_spatial_augmentation_keeps_image_and_label_aligned(self):
        labels = np.arange(16, dtype=np.uint8).reshape(4, 4)
        image = np.repeat(labels[:, :, None], 3, axis=2)

        for rotation_k in range(4):
            for flip_horizontal in (False, True):
                augmented_image, augmented_labels = apply_spatial_augmentation(
                    image, labels, rotation_k, flip_horizontal
                )
                np.testing.assert_array_equal(
                    augmented_image[:, :, 0], augmented_labels
                )
                self.assertEqual(set(np.unique(augmented_labels)), set(range(16)))

    def test_invalid_clutter_patch_fraction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "clutter-patch-fraction"):
            PotsdamPatchDataset(
                Path(tempfile.gettempdir()),
                ["tile"],
                patch_size=8,
                epoch_size=1,
                clutter_patch_fraction=1.1,
            )


if __name__ == "__main__":
    unittest.main()
