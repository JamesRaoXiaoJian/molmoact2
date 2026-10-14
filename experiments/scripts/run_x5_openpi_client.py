"""Molmo X5 hardware client; --help never imports hardware modules."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_entry(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Remaining arguments go to the original OpenPI X5 main. Required: "
            "--left-camera, --right-camera, --head-camera[-serial], and both "
            "--*-gripper-open-readout; MAE also needs --tactile-serials (two). "
            "Set task with --prompt. Motion requires --enable-motion, "
            "--confirm-motion MOVE_X5, --gravity-compensation-setup. "
            "Default is 5-step synchronous execution; --rtc selects 30 Hz asynchronous "
            "RTC (queue threshold 18, execution horizon 20). "
            "Even without motion, real SDK initialization/exit sends hold/gain/damping."
        ),
    )
    parser.add_argument("--mode", choices=("rgb", "mae"), required=True)
    parser.add_argument(
        "--openpi-root",
        type=Path,
        default=Path.home() / "lwm_ws/openpi-x5-arx5-beta/openpi",
    )
    args, remaining = parser.parse_known_args(argv)
    flags = {arg.split("=", 1)[0] for arg in remaining}
    required = {
        "--left-camera",
        "--right-camera",
        "--left-gripper-open-readout",
        "--right-gripper-open-readout",
    }
    if required - flags or not flags.intersection(
        {"--head-camera", "--head-camera-serial"}
    ):
        parser.error(
            "Explicit RGB camera routes and both current gripper readouts are required"
        )
    if args.mode == "mae" and "--tactile-serials" not in flags:
        parser.error("MAE requires --tactile-serials LEFT_LEFT RIGHT_RIGHT")
    if args.mode == "rgb" and "--tactile-serials" in flags:
        parser.error("RGB does not consume tactile cameras")
    return args, remaining


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args, remaining = parse_entry(argv)
    from x5_client_logging import ClientRunLog

    with ClientRunLog(args.mode, argv) as run_log:
        _run_control(args, remaining, run_log)


def _run_control(args, remaining, run_log):
    sys.dont_write_bytecode = True
    sys.path[:0] = [
        str(args.openpi_root),
        str(args.openpi_root / "src"),
        str(args.openpi_root / "packages/openpi-client/src"),
        str(args.openpi_root / "examples/x5_client"),
    ]
    from x5_openpi_control import load_control_main

    module = load_control_main(args.openpi_root, args.mode)
    run_log.attach_control(module)
    original_argv = sys.argv
    sys.argv = ["molmo-x5-client", *remaining]
    try:
        module.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
