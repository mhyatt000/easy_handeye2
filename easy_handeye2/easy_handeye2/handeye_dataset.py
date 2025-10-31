from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Iterable, Optional

import yaml
from easy_handeye2_msgs.msg import Sample, SampleList

from . import SAMPLES_DIRECTORY
from .handeye_calibration import HandeyeCalibrationParameters
from .handeye_geometry import PoseDiff, pose_difference, rotation_distance_deg, translation_distance


@dataclass
class SampleGateConfig:
    min_translation: float = 0.005
    min_rotation_deg: float = 2.0
    min_interval_sec: float = 0.5


@dataclass
class DiversityConfig:
    min_translation: float = 0.01
    min_rotation_deg: float = 5.0
    max_samples: Optional[int] = 250


@dataclass
class SampleDatasetMetadata:
    name: str
    created_at: str
    gate: SampleGateConfig
    diversity: DiversityConfig
    parameters: HandeyeCalibrationParameters

    def as_dict(self):
        return {
            'name': self.name,
            'created_at': self.created_at,
            'gate': self.gate.__dict__,
            'diversity': self.diversity.__dict__,
            'parameters': {
                'name': self.parameters.name,
                'calibration_type': self.parameters.calibration_type,
                'robot_base_frame': self.parameters.robot_base_frame,
                'robot_effector_frame': self.parameters.robot_effector_frame,
                'tracking_base_frame': self.parameters.tracking_base_frame,
                'tracking_marker_frame': self.parameters.tracking_marker_frame,
                'freehand_robot_movement': self.parameters.freehand_robot_movement,
            },
        }


class SampleGate:
    def __init__(self, config: SampleGateConfig):
        self.config = config
        self._last_timestamp_ns: Optional[int] = None

    def accept(self, node, sample: Sample, existing: SampleList) -> bool:
        if node is None:
            return True
        now_ns = node.get_clock().now().nanoseconds
        if self._last_timestamp_ns is not None:
            delta = (now_ns - self._last_timestamp_ns) * 1e-9
            if delta < self.config.min_interval_sec:
                return False
        if not existing.samples:
            self._last_timestamp_ns = now_ns
            return True
        last = existing.samples[-1]
        if translation_distance(sample.robot, last.robot) < self.config.min_translation:
            return False
        if rotation_distance_deg(sample.robot, last.robot) < self.config.min_rotation_deg:
            return False
        if translation_distance(sample.tracking, last.tracking) < self.config.min_translation:
            return False
        if rotation_distance_deg(sample.tracking, last.tracking) < self.config.min_rotation_deg:
            return False
        self._last_timestamp_ns = now_ns
        return True


class DiversityFilter:
    def __init__(self, config: DiversityConfig):
        self.config = config

    def _closest_pair(self, samples: Iterable[Sample]) -> Optional[tuple[int, int]]:
        indices = list(range(len(samples)))
        if len(indices) < 2:
            return None
        best: Optional[tuple[int, int, float]] = None
        for i in indices:
            for j in indices:
                if j <= i:
                    continue
                s_i = samples[i]
                s_j = samples[j]
                dist = translation_distance(s_i.robot, s_j.robot) + translation_distance(s_i.tracking, s_j.tracking)
                dist += 0.01 * (rotation_distance_deg(s_i.robot, s_j.robot) + rotation_distance_deg(s_i.tracking, s_j.tracking))
                if best is None or dist < best[2]:
                    best = (i, j, dist)
        if best is None:
            return None
        return best[0], best[1]

    def enforce(self, samples: SampleList) -> list[int]:
        removed = []
        if self.config.min_translation > 0 or self.config.min_rotation_deg > 0:
            filtered = []
            for sample in samples.samples:
                if any(self._too_close(sample, kept) for kept in filtered):
                    removed.append(sample)
                    continue
                filtered.append(sample)
            samples.samples = filtered
        if self.config.max_samples and len(samples.samples) > self.config.max_samples:
            while len(samples.samples) > self.config.max_samples:
                idx_pair = self._closest_pair(samples.samples)
                if idx_pair is None:
                    break
                _, idx_remove = idx_pair
                samples.samples.pop(idx_remove)
        return removed

    def _too_close(self, new_sample: Sample, existing: Sample) -> bool:
        if translation_distance(new_sample.robot, existing.robot) < self.config.min_translation:
            return True
        if rotation_distance_deg(new_sample.robot, existing.robot) < self.config.min_rotation_deg:
            return True
        if translation_distance(new_sample.tracking, existing.tracking) < self.config.min_translation:
            return True
        if rotation_distance_deg(new_sample.tracking, existing.tracking) < self.config.min_rotation_deg:
            return True
        return False


class SampleDataset:
    def __init__(self, node, parameters: HandeyeCalibrationParameters, gate: SampleGate, diversity: DiversityFilter,
                 auto_save: bool = True):
        self.node = node
        self.parameters = parameters
        self.gate = gate
        self.diversity = diversity
        self.auto_save = auto_save
        self.samples = SampleList()
        self.metadata = SampleDatasetMetadata(
            name=parameters.name,
            created_at=dt.datetime.utcnow().isoformat(),
            gate=gate.config,
            diversity=diversity.config,
            parameters=parameters,
        )
        self.persist_path = pathlib.Path(SAMPLES_DIRECTORY / f'{parameters.name}.samples')
        self.meta_path = pathlib.Path(SAMPLES_DIRECTORY / f'{parameters.name}.meta.json')

    def add(self, sample: Sample) -> bool:
        if not self.gate.accept(self.node, sample, self.samples):
            return False
        self.samples.samples.append(sample)
        removed = self.diversity.enforce(self.samples)
        if removed and self.node is not None:
            self.node.get_logger().info(f'Pruned {len(removed)} redundant samples')
        if self.auto_save:
            self.save()
        return True

    def remove(self, index: int):
        if 0 <= index < len(self.samples.samples):
            self.samples.samples.pop(index)
            if self.auto_save:
                self.save()

    def load(self) -> bool:
        if not self.persist_path.exists():
            return False
        with open(self.persist_path) as handle:
            sample_yaml = yaml.safe_load(handle.read()) or {}
        new_samples = SampleList()
        for entry in sample_yaml.get('samples', []):
            sample = Sample()
            sample.robot.translation.x = entry['robot']['translation']['x']
            sample.robot.translation.y = entry['robot']['translation']['y']
            sample.robot.translation.z = entry['robot']['translation']['z']
            sample.robot.rotation.x = entry['robot']['rotation']['x']
            sample.robot.rotation.y = entry['robot']['rotation']['y']
            sample.robot.rotation.z = entry['robot']['rotation']['z']
            sample.robot.rotation.w = entry['robot']['rotation']['w']
            sample.tracking.translation.x = entry['tracking']['translation']['x']
            sample.tracking.translation.y = entry['tracking']['translation']['y']
            sample.tracking.translation.z = entry['tracking']['translation']['z']
            sample.tracking.rotation.x = entry['tracking']['rotation']['x']
            sample.tracking.rotation.y = entry['tracking']['rotation']['y']
            sample.tracking.rotation.z = entry['tracking']['rotation']['z']
            sample.tracking.rotation.w = entry['tracking']['rotation']['w']
            new_samples.samples.append(sample)
        self.samples = new_samples
        if self.meta_path.exists():
            with open(self.meta_path) as handle:
                meta = json.load(handle)
            self.metadata.name = meta.get('name', self.metadata.name)
            self.metadata.created_at = meta.get('created_at', self.metadata.created_at)
            params_meta = meta.get('parameters')
            if params_meta:
                self.parameters = HandeyeCalibrationParameters(
                    name=params_meta['name'],
                    calibration_type=params_meta['calibration_type'],
                    robot_base_frame=params_meta['robot_base_frame'],
                    robot_effector_frame=params_meta['robot_effector_frame'],
                    tracking_base_frame=params_meta['tracking_base_frame'],
                    tracking_marker_frame=params_meta['tracking_marker_frame'],
                    freehand_robot_movement=params_meta['freehand_robot_movement'],
                )
                self.metadata.parameters = self.parameters
            self.metadata.gate = SampleGateConfig(**meta.get('gate', self.metadata.gate.__dict__))
            self.metadata.diversity = DiversityConfig(**meta.get('diversity', self.metadata.diversity.__dict__))
            self.gate.config = self.metadata.gate
            self.diversity.config = self.metadata.diversity
        return True

    def save(self) -> bool:
        if not os.path.exists(SAMPLES_DIRECTORY):
            os.makedirs(SAMPLES_DIRECTORY)
        with open(self.persist_path, 'w') as handle:
            yaml.safe_dump({'samples': [self._sample_to_dict(s) for s in self.samples.samples]}, handle, sort_keys=False)
        with open(self.meta_path, 'w') as handle:
            json.dump(self.metadata.as_dict(), handle, indent=2)
        return True

    def _sample_to_dict(self, sample: Sample) -> dict:
        return {
            'robot': {
                'translation': {
                    'x': sample.robot.translation.x,
                    'y': sample.robot.translation.y,
                    'z': sample.robot.translation.z,
                },
                'rotation': {
                    'x': sample.robot.rotation.x,
                    'y': sample.robot.rotation.y,
                    'z': sample.robot.rotation.z,
                    'w': sample.robot.rotation.w,
                },
            },
            'tracking': {
                'translation': {
                    'x': sample.tracking.translation.x,
                    'y': sample.tracking.translation.y,
                    'z': sample.tracking.translation.z,
                },
                'rotation': {
                    'x': sample.tracking.rotation.x,
                    'y': sample.tracking.rotation.y,
                    'z': sample.tracking.rotation.z,
                    'w': sample.tracking.rotation.w,
                },
            },
        }

    def difference_to_last(self, sample: Sample) -> Optional[PoseDiff]:
        if not self.samples.samples:
            return None
        return pose_difference(sample.robot, self.samples.samples[-1].robot)


def load_dataset(name: str) -> Optional[SampleDataset]:
    samples_path = pathlib.Path(SAMPLES_DIRECTORY / f'{name}.samples')
    meta_path = pathlib.Path(SAMPLES_DIRECTORY / f'{name}.meta.json')
    if not samples_path.exists() or not meta_path.exists():
        return None
    with open(meta_path) as handle:
        meta = json.load(handle)
    params = HandeyeCalibrationParameters(
        name=meta['parameters']['name'],
        calibration_type=meta['parameters']['calibration_type'],
        robot_base_frame=meta['parameters']['robot_base_frame'],
        robot_effector_frame=meta['parameters']['robot_effector_frame'],
        tracking_base_frame=meta['parameters']['tracking_base_frame'],
        tracking_marker_frame=meta['parameters']['tracking_marker_frame'],
        freehand_robot_movement=meta['parameters']['freehand_robot_movement'],
    )
    gate_cfg = SampleGateConfig(**meta['gate'])
    diversity_cfg = DiversityConfig(**meta['diversity'])
    dataset = SampleDataset(None, params, SampleGate(gate_cfg), DiversityFilter(diversity_cfg))
    dataset.persist_path = samples_path
    dataset.meta_path = meta_path
    dataset.metadata.created_at = meta['created_at']
    dataset.metadata.parameters = params
    dataset.parameters = params
    dataset.load()
    return dataset
