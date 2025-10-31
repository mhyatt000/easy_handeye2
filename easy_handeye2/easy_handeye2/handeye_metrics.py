from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass
from typing import List

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from easy_handeye2_msgs.msg import SampleList

from .handeye_geometry import rotation_angle_deg_from_matrix, transform_to_matrix, transform_translation


@dataclass
class CalibrationMetrics:
    translation_errors: np.ndarray
    rotation_errors_deg: np.ndarray
    condition_numbers: np.ndarray
    pose_spread: np.ndarray
    rotation_spread_deg: float
    report_path: pathlib.Path
    summary_path: pathlib.Path


def _compute_residuals(calibration_matrix: np.ndarray, samples: SampleList) -> tuple[np.ndarray, np.ndarray]:
    translation_errors: List[float] = []
    rotation_errors: List[float] = []
    for sample in samples.samples:
        robot = transform_to_matrix(sample.robot)
        tracking = transform_to_matrix(sample.tracking)
        lhs = robot @ calibration_matrix
        rhs = calibration_matrix @ tracking
        delta = np.linalg.inv(rhs) @ lhs
        translation_errors.append(float(np.linalg.norm(delta[:3, 3])))
        rotation_errors.append(rotation_angle_deg_from_matrix(delta[:3, :3]))
    return np.array(translation_errors), np.array(rotation_errors)


def _compute_condition_numbers(samples: SampleList) -> np.ndarray:
    if len(samples.samples) < 2:
        return np.array([])
    conds = []
    for idx in range(len(samples.samples) - 1):
        a_i = transform_to_matrix(samples.samples[idx].robot)
        a_j = transform_to_matrix(samples.samples[idx + 1].robot)
        b_i = transform_to_matrix(samples.samples[idx].tracking)
        b_j = transform_to_matrix(samples.samples[idx + 1].tracking)
        rel_a = np.linalg.inv(a_i) @ a_j
        rel_b = b_i @ np.linalg.inv(b_j)
        mat = rel_a[:3, :3] - rel_b[:3, :3]
        conds.append(np.linalg.cond(mat))
    return np.array(conds)


def _pose_spread(samples: SampleList) -> tuple[np.ndarray, float]:
    if not samples.samples:
        return np.zeros(3), 0.0
    translations = np.array([transform_translation(s.robot) for s in samples.samples])
    spread = translations.max(axis=0) - translations.min(axis=0)
    rotations_deg = []
    for sample in samples.samples:
        rotations_deg.append(rotation_angle_deg_from_matrix(transform_to_matrix(sample.robot)[:3, :3]))
    rotation_span = max(rotations_deg) - min(rotations_deg) if rotations_deg else 0.0
    return spread, rotation_span


def _ensure_dir(path: pathlib.Path):
    os.makedirs(path, exist_ok=True)


def compute_metrics(calibration, samples: SampleList) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    calibration_matrix = transform_to_matrix(calibration.transform)
    translation_errors, rotation_errors = _compute_residuals(calibration_matrix, samples)
    conds = _compute_condition_numbers(samples)
    pose_spread, rotation_span = _pose_spread(samples)
    return translation_errors, rotation_errors, conds, pose_spread, rotation_span


def save_metrics_report(parameters, calibration, samples: SampleList) -> CalibrationMetrics:
    translation_errors, rotation_errors, conds, pose_spread, rotation_span = compute_metrics(calibration, samples)
    report_dir = pathlib.Path(os.path.expanduser('~/.ros2/easy_handeye2/reports'))
    _ensure_dir(report_dir)
    dataset_dir = report_dir / parameters.name
    _ensure_dir(dataset_dir)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(translation_errors, bins=max(len(translation_errors) // 2, 10))
    axes[0].set_title('Reprojection residual (m)')
    axes[0].set_xlabel('meters')

    translations = np.array([transform_translation(s.robot) for s in samples.samples]) if samples.samples else np.zeros((0, 3))
    if translations.size:
        axes[1].scatter(translations[:, 0], translations[:, 1], c=np.arange(len(translations)))
    axes[1].set_title('Pose spread (X/Y)')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')

    if conds.size:
        axes[2].plot(conds)
    axes[2].set_title('Condition number per pair')
    axes[2].set_xlabel('Pair index')
    axes[2].set_ylabel('Cond')

    fig.tight_layout()
    report_path = dataset_dir / 'metrics.png'
    fig.savefig(report_path, dpi=180)
    plt.close(fig)

    summary_path = dataset_dir / 'metrics.json'
    summary = {
        'translation_mean': float(translation_errors.mean()) if translation_errors.size else 0.0,
        'translation_std': float(translation_errors.std()) if translation_errors.size else 0.0,
        'rotation_mean_deg': float(rotation_errors.mean()) if rotation_errors.size else 0.0,
        'rotation_std_deg': float(rotation_errors.std()) if rotation_errors.size else 0.0,
        'condition_mean': float(conds.mean()) if conds.size else 0.0,
        'condition_std': float(conds.std()) if conds.size else 0.0,
        'translation_spread': pose_spread.tolist(),
        'rotation_span_deg': rotation_span,
    }
    with open(summary_path, 'w') as handle:
        json.dump(summary, handle, indent=2)

    return CalibrationMetrics(
        translation_errors=translation_errors,
        rotation_errors_deg=rotation_errors,
        condition_numbers=conds,
        pose_spread=pose_spread,
        rotation_spread_deg=rotation_span,
        report_path=report_path,
        summary_path=summary_path,
    )
