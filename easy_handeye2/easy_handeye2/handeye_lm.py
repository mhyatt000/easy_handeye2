import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass
class MotionPair:
    """Paired robot and camera motions for AX = XB."""
    A: np.ndarray  # 4x4 robot motion
    B: np.ndarray  # 4x4 camera motion
    sqrt_info: Optional[np.ndarray] = None  # 6x6 sqrt information


@dataclass
class CornerBatch:
    """Corner reprojection block tied to one capture."""
    base_T_ee: np.ndarray  # 4x4
    base_T_tag: np.ndarray  # 4x4
    tag_corners: np.ndarray  # Nx3
    pixel_corners: np.ndarray  # Nx2
    K: np.ndarray  # 3x3 intrinsics
    weight: float = 1.0


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _left_jacobian(phi: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(phi)
    if theta < 1e-9:
        K = _skew(phi)
        return np.eye(3) + 0.5 * K + (1.0 / 12.0) * K @ K
    K = _skew(phi / theta)
    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / (theta * theta)
    C = (1.0 - A) / (theta * theta)
    return np.eye(3) + B * K + C * (K @ K)


def _left_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(phi)
    if theta < 1e-9:
        K = _skew(phi)
        return np.eye(3) - 0.5 * K + (1.0 / 12.0) * K @ K
    K = _skew(phi)
    half = 0.5 * theta
    sin_half = np.sin(half)
    if abs(sin_half) < 1e-9:
        sin_half = 1e-9
    cot_half = np.cos(half) / sin_half
    coeff = (1.0 - 0.5 * theta * cot_half) / (theta * theta)
    return np.eye(3) - 0.5 * K + coeff * (K @ K)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    rho = xi[:3]
    phi = xi[3:]
    R = Rotation.from_rotvec(phi).as_matrix()
    V = _left_jacobian(phi)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    phi = Rotation.from_matrix(R).as_rotvec()
    V_inv = _left_jacobian_inv(phi)
    rho = V_inv @ T[:3, 3]
    return np.hstack((rho, phi))


def _weighted_residual(residual: np.ndarray, sqrt_info: Optional[np.ndarray]) -> np.ndarray:
    if sqrt_info is None:
        return residual
    return sqrt_info @ residual


def axb_residuals(X: np.ndarray, pairs: Sequence[MotionPair]) -> np.ndarray:
    T = se3_exp(X)
    res = []
    T_inv = np.linalg.inv(T)
    for pair in pairs:
        err = pair.A @ T @ pair.B @ T_inv
        res.append(_weighted_residual(se3_log(err), pair.sqrt_info))
    return np.concatenate(res) if res else np.zeros(0)


def _project_corners(T: np.ndarray, corners: np.ndarray, K: np.ndarray) -> np.ndarray:
    pts = (T @ np.hstack((corners, np.ones((corners.shape[0], 1)))).T).T
    pts = pts[:, :3]
    zs = pts[:, 2:3]
    zs[zs == 0.0] = 1e-9
    pix = (pts[:, :2] / zs) @ K[:2, :2].T + K[:2, 2]
    return pix


def corner_residuals(X: np.ndarray, batches: Sequence[CornerBatch]) -> np.ndarray:
    T = se3_exp(X)
    res = []
    for batch in batches:
        base_T_cam = batch.base_T_ee @ T
        cam_T_tag = np.linalg.inv(base_T_cam) @ batch.base_T_tag
        proj = _project_corners(cam_T_tag, batch.tag_corners, batch.K)
        diff = (proj - batch.pixel_corners).reshape(-1)
        res.append(np.sqrt(batch.weight) * diff)
    return np.concatenate(res) if res else np.zeros(0)


def residual_stack(x: np.ndarray, pairs: Sequence[MotionPair], batches: Sequence[CornerBatch]) -> np.ndarray:
    return np.concatenate((axb_residuals(x, pairs), corner_residuals(x, batches)))


def optimize_handeye(
    seed: np.ndarray,
    pairs: Sequence[MotionPair],
    batches: Sequence[CornerBatch],
    max_nfev: int = 50,
    verbose: int = 1,
):
    def fun(x):
        return residual_stack(x, pairs, batches)

    return least_squares(fun, seed, method='lm', max_nfev=max_nfev, verbose=verbose)
