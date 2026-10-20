"""Regression tests for SAM's class-balanced segmentation losses."""

import unittest

import torch

from semantic_labeling.constants import IGNORE_LABEL, N_CLASSES
from semantic_labeling.losses import (
    inverse_sqrt_class_weights,
    weighted_ce_macro_dice_loss,
)


class InverseSqrtClassWeightsTests(unittest.TestCase):
    def test_weights_are_mean_one_and_clipped(self):
        frequencies = torch.tensor(
            [0.2847, 0.2488, 0.2478, 0.1464, 0.0161, 0.0561]
        )

        weights = inverse_sqrt_class_weights(frequencies)

        self.assertEqual(weights.shape, (N_CLASSES,))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreaterEqual(float(weights.min()), 0.5)
        self.assertLessEqual(float(weights.max()), 2.0)
        self.assertGreater(float(weights[5]), float(weights[0]))

    def test_rejects_nonpositive_or_nonfinite_frequencies(self):
        for frequencies in ([0.5, 0.5, 0, 0.0, 0.0, 0.0], [float("nan")] * 6):
            with self.subTest(frequencies=frequencies), self.assertRaises(ValueError):
                inverse_sqrt_class_weights(frequencies)


class WeightedCrossEntropyDiceLossTests(unittest.TestCase):
    def setUp(self):
        self.class_weights = torch.ones(N_CLASSES)

    def test_ignore_pixels_do_not_change_loss(self):
        torch.manual_seed(7)
        logits = torch.randn(1, N_CLASSES, 2, 2)
        targets = torch.tensor([[[0, 1], [2, 3]]])
        extended_logits = torch.cat(
            [logits, torch.randn(1, N_CLASSES, 2, 1)], dim=3
        )
        extended_targets = torch.cat(
            [targets, torch.full((1, 2, 1), IGNORE_LABEL)], dim=2
        )

        original_loss = weighted_ce_macro_dice_loss(
            logits, targets, self.class_weights
        )
        extended_loss = weighted_ce_macro_dice_loss(
            extended_logits, extended_targets, self.class_weights
        )

        self.assertTrue(torch.allclose(original_loss, extended_loss, atol=1e-6))

    def test_absent_classes_produce_finite_loss_and_gradients(self):
        logits = torch.randn(1, N_CLASSES, 2, 2, requires_grad=True)
        targets = torch.tensor([[[0, 1], [0, 1]]])

        loss = weighted_ce_macro_dice_loss(logits, targets, self.class_weights)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_all_ignored_targets_return_zero_with_zero_gradient(self):
        logits = torch.randn(1, N_CLASSES, 2, 2, requires_grad=True)
        targets = torch.full((1, 2, 2), IGNORE_LABEL)

        loss = weighted_ce_macro_dice_loss(logits, targets, self.class_weights)
        loss.backward()

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))


if __name__ == "__main__":
    unittest.main()
