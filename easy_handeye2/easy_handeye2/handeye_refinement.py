from __future__ import annotations

import copy
from dataclasses import dataclass

import cv2
import numpy as np

from easy_handeye2_msgs.msg import Sample, SampleList

from .handeye_geometry import compose_matrix, matrix_to_transform, transform_to_matrix


@dataclass
class CornerPnPConfig:
    board_width: int = 4
    board_height: int = 4
    square_size: float = 0.03
    fx: float = 800.0
    fy: float = 800.0
    cx: float = 640.0
    cy: float = 480.0


class CornerPnPRefiner:
    def __init__(self, config: CornerPnPConfig | None = None):
        self.config = config or CornerPnPConfig()
        self.object_points = self._generate_object_points()
        self.camera_matrix = np.array([
            [self.config.fx, 0.0, self.config.cx],
            [0.0, self.config.fy, self.config.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    def _generate_object_points(self) -> np.ndarray:
        pts = []
        for y in range(self.config.board_height):
            for x in range(self.config.board_width):
                pts.append((x * self.config.square_size, y * self.config.square_size, 0.0))
        return np.array(pts, dtype=np.float64)

    def refine(self, sample: Sample) -> Sample:
        tracking_matrix = transform_to_matrix(sample.tracking)
        rot, _ = cv2.Rodrigues(tracking_matrix[:3, :3])
        tvec = tracking_matrix[:3, 3].reshape(3, 1)
        image_points, _ = cv2.projectPoints(self.object_points, rot, tvec, self.camera_matrix, self.dist_coeffs)
        success, rvec, tvec_refined = cv2.solvePnP(self.object_points, image_points, self.camera_matrix,
                                                   self.dist_coeffs, rvec=rot, tvec=tvec, useExtrinsicGuess=True)
        if success:
            rot_refined, _ = cv2.Rodrigues(rvec)
            matrix = compose_matrix(rot_refined, tvec_refined.reshape(3))
            sample.tracking = matrix_to_transform(matrix)
        return sample

    def refine_samples(self, samples: SampleList) -> SampleList:
        refined = SampleList()
        for sample in samples.samples:
            refined_sample = Sample()
            refined_sample.robot = copy.deepcopy(sample.robot)
            refined_sample.tracking = copy.deepcopy(sample.tracking)
            refined.samples.append(self.refine(refined_sample))
        return refined


@dataclass
class BundleAdjustConfig:
    max_iterations: int = 15
    rotation_step: float = 0.5
    translation_step: float = 0.5
    tolerance: float = 1e-5


class DaniilidisBundleAdjuster:
    def __init__(self, config: BundleAdjustConfig | None = None):
        self.config = config or BundleAdjustConfig()

    def refine(self, initial_transform, samples: SampleList):
        matrix = transform_to_matrix(initial_transform)
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        for _ in range(self.config.max_iterations):
            rot_updates = []
            trans_updates = []
            for sample in samples.samples:
                a_matrix = transform_to_matrix(sample.robot)
                b_matrix = transform_to_matrix(sample.tracking)
                lhs_rot = a_matrix[:3, :3] @ rotation
                rhs_rot = rotation @ b_matrix[:3, :3]
                delta_rot = rhs_rot.T @ lhs_rot
                rot_vec, _ = cv2.Rodrigues(delta_rot)
                rot_updates.append(rot_vec.reshape(3))
                lhs_tr = a_matrix[:3, :3] @ translation + a_matrix[:3, 3]
                rhs_tr = rotation @ b_matrix[:3, 3] + translation
                trans_updates.append(lhs_tr - rhs_tr)
            if not rot_updates:
                break
            avg_rot = np.mean(rot_updates, axis=0)
            avg_tr = np.mean(trans_updates, axis=0)
            if np.linalg.norm(avg_rot) < self.config.tolerance and np.linalg.norm(avg_tr) < self.config.tolerance:
                break
            step_rot = avg_rot * self.config.rotation_step
            step_tr = avg_tr * self.config.translation_step
            rot_update, _ = cv2.Rodrigues(step_rot)
            rotation = rot_update @ rotation
            translation = translation - step_tr
        return matrix_to_transform(compose_matrix(rotation, translation))
