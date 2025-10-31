import math
from typing import Iterable

import numpy as np
import transforms3d as tfs
from geometry_msgs.msg import Transform

from easy_handeye2_msgs.msg import Sample


TRANSLATION_TOLERANCE_M = 0.003
ROTATION_TOLERANCE_RAD = math.radians(3)


def translation_distance(t1: Transform, t2: Transform) -> float:
    v1 = np.array((t1.translation.x, t1.translation.y, t1.translation.z))
    v2 = np.array((t2.translation.x, t2.translation.y, t2.translation.z))
    return float(np.linalg.norm(v1 - v2))


def _normalized_quaternion(q: Iterable[float]) -> np.ndarray:
    arr = np.array(tuple(q), dtype=float)
    return arr / tfs.quaternions.qnorm(arr)


def quaternion_log(q: Iterable[float]) -> np.ndarray:
    normalized = _normalized_quaternion(q)
    unit = tfs.quaternions.fillpositive(np.array(normalized[1:4]))
    vec = unit[1:4]
    if np.allclose(vec, np.zeros(3)):
        return np.zeros(3)
    angle = math.acos(float(np.clip(unit[0], -1.0, 1.0)))
    return angle * vec / np.linalg.norm(vec)


def quaternion_distance(q1: Iterable[float], q2: Iterable[float]) -> float:
    unit_1 = tfs.quaternions.fillpositive(np.array(_normalized_quaternion(q1)[1:4]))
    unit_2 = tfs.quaternions.fillpositive(np.array(_normalized_quaternion(q2)[1:4]))
    delta = tfs.quaternions.qmult(unit_1, tfs.quaternions.qconjugate(unit_2))
    if np.allclose(delta, np.array([-1.0, 0.0, 0.0, 0.0])):
        return 2.0 * math.pi
    return float(2.0 * np.linalg.norm(quaternion_log(delta)))


def rotation_distance(t1: Transform, t2: Transform) -> float:
    q1 = (t1.rotation.w, t1.rotation.x, t1.rotation.y, t1.rotation.z)
    q2 = (t2.rotation.w, t2.rotation.x, t2.rotation.y, t2.rotation.z)
    return quaternion_distance(q1, q2)


def transform_distance(
    t1: Transform,
    t2: Transform,
    translation_scale: float = TRANSLATION_TOLERANCE_M,
    rotation_scale: float = ROTATION_TOLERANCE_RAD,
) -> float:
    translation_term = translation_distance(t1, t2) / translation_scale if translation_scale else 0.0
    rotation_term = rotation_distance(t1, t2) / rotation_scale if rotation_scale else 0.0
    return math.sqrt(translation_term ** 2 + rotation_term ** 2)


def sample_distance(sample_a: Sample, sample_b: Sample) -> float:
    robot = transform_distance(sample_a.robot, sample_b.robot)
    tracking = transform_distance(sample_a.tracking, sample_b.tracking)
    return math.sqrt(robot ** 2 + tracking ** 2)


__all__ = [
    'TRANSLATION_TOLERANCE_M',
    'ROTATION_TOLERANCE_RAD',
    'translation_distance',
    'rotation_distance',
    'transform_distance',
    'sample_distance',
]
