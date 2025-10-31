#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import sys

from easy_handeye2.handeye_calibration import save_calibration
from easy_handeye2.handeye_calibration_backend_opencv import HandeyeCalibrationBackendOpenCV
from easy_handeye2.handeye_dataset import load_dataset
from easy_handeye2.handeye_metrics import save_metrics_report
from rosidl_runtime_py import message_to_yaml


class _Logger:
    def info(self, msg: str):  # noqa: D401 - concise logging proxy
        print(msg)

    def warn(self, msg: str):
        print(f'WARN: {msg}', file=sys.stderr)

    def error(self, msg: str):
        print(f'ERROR: {msg}', file=sys.stderr)


class _Node:
    def __init__(self):
        self._logger = _Logger()

    def get_logger(self):
        return self._logger


def _resume_dataset(args) -> int:
    dataset = load_dataset(args.name)
    if dataset is None:
        print(f'Dataset {args.name} not found', file=sys.stderr)
        return 1
    print(f'Dataset: {dataset.metadata.name}')
    print(f'Created: {dataset.metadata.created_at}')
    print(f'Samples: {len(dataset.samples.samples)}')
    print(f'Gate config: {dataset.metadata.gate}')
    print(f'Diversity config: {dataset.metadata.diversity}')
    return 0


def _bundle_adjust(args) -> int:
    dataset = load_dataset(args.name)
    if dataset is None:
        print(f'Dataset {args.name} not found', file=sys.stderr)
        return 1
    node = _Node()
    backend = HandeyeCalibrationBackendOpenCV()
    calibration = backend.compute_calibration(node, dataset.parameters, dataset.samples, algorithm='Daniilidis')
    if calibration is None:
        print('Calibration failed', file=sys.stderr)
        return 2
    metrics = save_metrics_report(dataset.parameters, calibration, dataset.samples)
    print(f'Metrics stored at {metrics.report_path}')
    if args.output:
        path = pathlib.Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(message_to_yaml(calibration))
        print(f'Refined calibration saved to {path}')
    else:
        save_path = save_calibration(calibration)
        print(f'Refined calibration saved to {save_path}')
    return 0


def main():
    parser = argparse.ArgumentParser(description='easy_handeye2 dataset utilities')
    sub = parser.add_subparsers(dest='command', required=True)

    resume = sub.add_parser('resume', help='Inspect a persisted dataset')
    resume.add_argument('name', help='Dataset name')
    resume.set_defaults(func=_resume_dataset)

    bundle = sub.add_parser('bundle-adjust', help='Re-run Daniilidis calibration with bundle adjustment')
    bundle.add_argument('name', help='Dataset name')
    bundle.add_argument('--output', help='Optional output file path for the refined calibration')
    bundle.set_defaults(func=_bundle_adjust)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == '__main__':
    main()
