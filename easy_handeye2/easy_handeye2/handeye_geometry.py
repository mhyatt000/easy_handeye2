from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import transforms3d as tfs
from geometry_msgs.msg import Quaternion, Transform, Vector3


def _quat_to_tuple(quat: Quaternion) -> tuple[float, float, float, float]:
    return quat.w, quat.x, quat.y, quat.z


def _tuple_to_quat(quat: tuple[float, float, float, float]) -> Quaternion:
    w, x, y, z = quat
    return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))


def _vector_to_tuple(vec: Vector3) -> tuple[float, float, float]:
    return vec.x, vec.y, vec.z


def _tuple_to_vector(vec: tuple[float, float, float]) -> Vector3:
    x, y, z = vec
    return Vector3(x=float(x), y=float(y), z=float(z))


def transform_to_matrix(transform: Transform) -> np.ndarray:
    rot = tfs.quaternions.quat2mat(_quat_to_tuple(transform.rotation))
    tr = np.array(_vector_to_tuple(transform.translation))
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = tr
    return mat


def matrix_to_transform(matrix: np.ndarray) -> Transform:
    rot = matrix[:3, :3]
    tr = matrix[:3, 3]
    quat = tfs.quaternions.mat2quat(rot)
    return Transform(translation=_tuple_to_vector(tuple(tr)), rotation=_tuple_to_quat(tuple(quat)))


def compose_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = rotation
    mat[:3, 3] = translation
    return mat


def translation_distance(lhs: Transform, rhs: Transform) -> float:
    lv = np.array(_vector_to_tuple(lhs.translation))
    rv = np.array(_vector_to_tuple(rhs.translation))
    return float(np.linalg.norm(lv - rv))


def rotation_distance_deg(lhs: Transform, rhs: Transform) -> float:
    lq = np.array(_quat_to_tuple(lhs.rotation))
    rq = np.array(_quat_to_tuple(rhs.rotation))
    dot = float(np.clip(np.dot(lq, rq), -1.0, 1.0))
    angle = 2.0 * math.acos(abs(dot))
    return math.degrees(angle)


def transform_translation(transform: Transform) -> np.ndarray:
    return np.array(_vector_to_tuple(transform.translation))


def transform_rotation(transform: Transform) -> np.ndarray:
    return tfs.quaternions.quat2mat(_quat_to_tuple(transform.rotation))


def rotation_vector_from_matrix(matrix: np.ndarray) -> np.ndarray:
    rot_vec, _ = cv2.Rodrigues(matrix)
    return rot_vec.reshape(3)


def rotation_vector_from_transform(transform: Transform) -> np.ndarray:
    rot = transform_rotation(transform)
    return rotation_vector_from_matrix(rot)


def rotation_angle_deg_from_matrix(matrix: np.ndarray) -> float:
    vec = rotation_vector_from_matrix(matrix)
    return math.degrees(np.linalg.norm(vec))


def rotation_angle_deg_from_transform(transform: Transform) -> float:
    return rotation_angle_deg_from_matrix(transform_rotation(transform))


def se3_error(lhs: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.linalg.inv(rhs) @ lhs
    rot_vec = rotation_vector_from_matrix(delta[:3, :3])
    tr = delta[:3, 3]
    return rot_vec, tr


@dataclass
class PoseDiff:
    rotation_deg: float
    translation: float


def pose_difference(lhs: Transform, rhs: Transform) -> PoseDiff:
    return PoseDiff(rotation_deg=rotation_distance_deg(lhs, rhs), translation=translation_distance(lhs, rhs))


try:
    import cv2  # noqa: WPS433, used for Rodrigues
except ImportError as err:  # pragma: no cover - cv2 is required at runtime
    raise RuntimeError('OpenCV is required for geometry utilities') from err
