from pathlib import Path

import pytest
import torch

from deepnet.training_utils import save_checkpoint, write_json
from model_script.pspnet_rgb import verified_previous


@pytest.mark.parametrize("checkpoint_update", [1151, 1152])
def test_benchmark_requires_completed_refit_checkpoint(
    tmp_path: Path, checkpoint_update: int
):
    digest = save_checkpoint(
        tmp_path / "final.pt",
        torch.nn.Linear(1, 1),
        {"updates": checkpoint_update},
        "binding",
    )
    write_json(
        tmp_path / "manifest.json",
        {
            "status": "complete",
            "phase": "refit",
            "updates": 1152,
            "source_sha256": "source",
            "reference_protocol": "reference",
            "checkpoint_sha256": digest,
            "cache_hash": "refit-cache",
            "binding": "binding",
        },
    )
    settings = {"source_sha256": "source", "cache_hashes": {"refit": "refit-cache"}}
    protocol = {"protocol_hash": "reference"}
    if checkpoint_update != 1152:
        with pytest.raises(ValueError, match="checkpoint progress"):
            verified_previous(tmp_path, "refit", settings, protocol)
    else:
        record, state = verified_previous(tmp_path, "refit", settings, protocol)
        assert record["checkpoint_sha256"] == digest
        assert set(state) == {"weight", "bias"}
