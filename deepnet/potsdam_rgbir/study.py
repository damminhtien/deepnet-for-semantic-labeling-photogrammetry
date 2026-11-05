"""Run one frozen RGBIR campaign phase with explicit, role-scoped inputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .phases import heldout, refit, select
from .preflight import preflight
from .protocol import CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, required=True)
    common.add_argument("--cache", type=Path, required=True)
    common.add_argument("--pretrained", type=Path, required=True)
    common.add_argument("--pretrained-large", type=Path)
    common.add_argument("--device", default="cuda")
    pilot = commands.add_parser("preflight", parents=[common])
    pilot.add_argument("--config", type=Path, default=CONFIG)
    pilot.add_argument("--calibration", type=Path, required=True)
    arm = commands.add_parser("select", parents=[common])
    arm.add_argument("--arm", choices="ABCLD", required=True)
    commands.add_parser("refit", parents=[common])
    commands.add_parser("heldout", parents=[common])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    {"preflight": preflight, "select": select, "refit": refit, "heldout": heldout}[
        args.phase
    ](args)


if __name__ == "__main__":
    main()
