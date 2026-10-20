import unittest

import torch

from model_script.train_sam_kaggle import build_warmup_cosine_scheduler


class WarmupCosineSchedulerTests(unittest.TestCase):
    def test_warmup_starts_below_base_and_decay_reaches_minimum(self):
        lora_parameter = torch.nn.Parameter(torch.ones(()))
        decoder_parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD(
            [
                {"params": [lora_parameter], "lr": 1e-4},
                {"params": [decoder_parameter], "lr": 3e-4},
            ]
        )

        scheduler, warmup_steps = build_warmup_cosine_scheduler(
            optimizer,
            total_steps=20,
            warmup_fraction=0.1,
            min_learning_rate=1e-6,
        )

        self.assertEqual(warmup_steps, 2)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-5)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1.5e-4)

        for _ in range(20):
            optimizer.step()
            scheduler.step()

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1e-6)


if __name__ == "__main__":
    unittest.main()
